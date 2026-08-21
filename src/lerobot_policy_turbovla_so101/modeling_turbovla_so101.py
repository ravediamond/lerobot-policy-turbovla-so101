#!/usr/bin/env python

# Copyright 2026 The TurboVLA-LeRobot contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TurboVLA Policy

A direct V + L -> A map, as per TurboVLA (paper: https://arxiv.org/abs/2607.27205, code:
https://github.com/H-EmbodVis/TurboVLA). There is no LLM in the middle, no action tokenization and
no autoregressive decode:

    images (N cams) --> DINOv3 ViT --> proj to d ------.
                        + patch pos emb                |
                        + camera-view emb              +--> N bidirectional cross-attention layers
                                                       |    (LN + xattn + FFN, residual)
    instruction ------> BERT (token-level) --> proj ---'                |
                                                                        v
                                                        ACT-style parallel action decoder
                                                                        |
                                                                        v
                                                          continuous chunk (H, action_dim)

Only the modules were ported from upstream, not the harness: upstream is built around its own
trainer, TFDS/RLDS for LIBERO, and a `flash-attn` dependency. Here `lerobot-train` drives
everything and all attention goes through `scaled_dot_product_attention`.
"""

from collections import deque
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .action_decoder import TurboVLAActionDecoder
from .configuration_turbovla_so101 import TurboVLAConfig
from .temporal_ensemble import TemporalEnsembler
from .vl_interaction import TurboVLAVLInteraction

# DINOv3 (like DINOv2 and most ImageNet-pretrained ViTs) expects ImageNet-normalized inputs. The
# dataset normalizer leaves the visual stream alone (`NormalizationMode.IDENTITY`), so we apply this
# here, right where the backbone that needs it lives.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_GATED_BACKBONE_HINT = (
    "The `facebook/dinov3-*` repos are gated. Accept the terms on the model page, then "
    "authenticate with `hf auth login` or set the HF_TOKEN environment variable. To build the "
    "policy without any Hub download (random init, e.g. for a smoke test), set "
    "`--policy.load_pretrained_backbones=false`."
)


class TurboVLAPolicy(PreTrainedPolicy):
    """TurboVLA: language-conditioned action chunking with a bidirectional V-L trunk.

    A drop-in alternative to ACT with the same ergonomics (chunked continuous actions, parallel
    decode, L1 loss). The difference that matters in practice: ACT ignores the task string, so a
    multi-task dataset needs one checkpoint per task. TurboVLA conditions on it, so one checkpoint
    can cover many tasks in the same dataset.
    """

    config_class = TurboVLAConfig
    name = "turbovla_so101"
    # FSDP2 wrap units: one unit per transformer layer of both stacks.
    _fsdp_wrap_modules = ["TurboVLAFusionLayer", "TurboVLADecoderLayer"]

    def __init__(self, config: TurboVLAConfig, **kwargs):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default
                    instantiation of the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = TurboVLA(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = TemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        """Two param groups: the freshly initialized trunk, and the pretrained backbones.

        Frozen backbones contribute no parameters at all (`requires_grad` is False), so the second
        group is simply empty in the default configuration.
        """
        backbone_prefixes = ("model.vision_encoder", "model.language_encoder")
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith(backbone_prefixes) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith(backbone_prefixes) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `predict_action_chunk` in order to return one action at a time for
        execution in the environment. With `temporal_ensemble_coeff` set, every call queries the
        model for a fresh chunk and the temporal ensembler blends it with the previous chunks'
        overlapping predictions rather than draining a queue — this is what removes the visible
        jitter at chunk boundaries from a single-shot decoder. Otherwise it works by managing the
        actions in a queue and only calling `predict_action_chunk` when the queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but
            # the queue effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        return self.model(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if batch.get("task") is None:
            # Without this the policy silently degenerates into a worse ACT: the whole point is the
            # language conditioning, so fail loudly rather than train a mute model.
            raise ValueError(
                "TurboVLA requires a task description, but the batch has no `task` key. Record or "
                "re-export the dataset with a task string per episode, or use `--policy.type=act` "
                "if the data genuinely has no language annotation."
            )

        actions_hat = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        # Plain L1 on the action chunk. No VAE and no KL term, unlike ACT.
        return l1_loss, {"l1_loss": l1_loss.item()}


class TurboVLA(nn.Module):
    """The underlying neural network for `TurboVLAPolicy`."""

    def __init__(self, config: TurboVLAConfig):
        super().__init__()
        self.config = config
        # Fixed camera order. It has to be stable between training and rollout, since it indexes the
        # camera-view embedding below.
        self.camera_keys = list(config.image_features)

        self.vision_encoder, vision_width, self.vision_patch_size = _load_vision_backbone(config)
        self.language_encoder, language_width = _load_language_backbone(config)
        self.tokenizer = _load_tokenizer(config)

        if config.image_size % self.vision_patch_size != 0:
            raise ValueError(
                f"`image_size` ({config.image_size}) must be divisible by the vision backbone's "
                f"patch size ({self.vision_patch_size})."
            )
        self.n_patches_per_side = config.image_size // self.vision_patch_size
        self.n_patches = self.n_patches_per_side**2

        if config.freeze_vision_backbone:
            self.vision_encoder.requires_grad_(False)
        if config.freeze_language_backbone:
            self.language_encoder.requires_grad_(False)

        # Project both modalities into the shared width `d`.
        self.vision_proj = nn.Linear(vision_width, config.dim_model)
        self.language_proj = nn.Linear(language_width, config.dim_model)

        # Where a patch is in the frame, and which camera it came from. The camera-view embedding is
        # what keeps views apart as soon as there is more than one.
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, config.dim_model))
        nn.init.normal_(self.patch_pos_embed, std=0.02)
        self.camera_embed = nn.Embedding(len(self.camera_keys), config.dim_model)

        # Optional proprioceptive state, carried as one extra token alongside the patches.
        self.state_proj = None
        if config.robot_state_feature is not None:
            self.state_proj = nn.Linear(config.robot_state_feature.shape[0], config.dim_model)
            self.state_embed = nn.Parameter(torch.zeros(1, 1, config.dim_model))
            nn.init.normal_(self.state_embed, std=0.02)

        self.vl_interaction = TurboVLAVLInteraction(config)
        self.action_decoder = TurboVLAActionDecoder(config, config.action_feature.shape[0])

        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self._set_frozen_backbones_to_eval()

        # Only the fusion + decoder stack is compiled: the backbones are separate pretrained
        # `transformers` modules, and DINOv3/BERT are unlikely to benefit as much (fixed shapes
        # already, and re-tracing on every unfrozen fine-tune step would fight the whole point of
        # caching a compiled graph). `forward` covers both training and inference — there's a single
        # decode path, unlike a flow-matching policy with separate one-shot vs. iterative-denoise
        # forwards to compile independently.
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.forward = torch.compile(self.forward)

    def train(self, mode: bool = True):
        """Keep frozen backbones in eval mode so their dropout / norm statistics stay put."""
        super().train(mode)
        self._set_frozen_backbones_to_eval()
        return self

    def _set_frozen_backbones_to_eval(self) -> None:
        if self.config.freeze_vision_backbone:
            self.vision_encoder.eval()
        if self.config.freeze_language_backbone:
            self.language_encoder.eval()

    def _encode_images(self, batch: dict[str, Tensor]) -> Tensor:
        """All camera views -> (B, n_cameras * n_patches, dim_model)."""
        tokens = []
        for camera_index, key in enumerate(self.camera_keys):
            images = batch[key]
            if images.ndim == 5:
                # (B, T, C, H, W) with n_obs_steps == 1 -> take the current frame.
                images = images[:, 0]

            if images.shape[-2:] != (self.config.image_size, self.config.image_size):
                images = F.interpolate(
                    images,
                    size=(self.config.image_size, self.config.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            images = (images - self.image_mean) / self.image_std

            if self.config.freeze_vision_backbone:
                with torch.no_grad():
                    features = self.vision_encoder(pixel_values=images).last_hidden_state
            else:
                features = self.vision_encoder(pixel_values=images).last_hidden_state

            # Drop the leading non-patch tokens (CLS, plus DINOv3's register tokens) and keep the
            # patch grid, which is what carries the spatial detail the decoder needs.
            n_prefix = features.shape[1] - self.n_patches
            if n_prefix < 0:
                raise ValueError(
                    f"Vision backbone returned {features.shape[1]} tokens, fewer than the "
                    f"{self.n_patches} patches expected for image_size={self.config.image_size} and "
                    f"patch_size={self.vision_patch_size}."
                )
            features = features[:, n_prefix:]

            projected = self.vision_proj(features) + self.patch_pos_embed
            projected = projected + self.camera_embed.weight[camera_index].view(1, 1, -1)
            tokens.append(projected)

        return torch.cat(tokens, dim=1)

    def _encode_language(self, tasks: str | list[str], batch_size: int, device) -> tuple[Tensor, Tensor]:
        """Instruction -> (B, n_text_tokens, dim_model) plus its (B, n_text_tokens) validity mask."""
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        else:
            tasks = list(tasks)
        if len(tasks) != batch_size:
            raise ValueError(
                f"Got {len(tasks)} task strings for a batch of {batch_size}. They must correspond "
                "one-to-one."
            )

        encoded = self.tokenizer(
            tasks,
            padding=True,
            truncation=True,
            max_length=self.config.max_language_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if self.config.freeze_language_backbone:
            with torch.no_grad():
                out = self.language_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.language_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Token-level, deliberately *not* pooled — pooling is what loses the object / attribute /
        # spatial-relation grounding this policy depends on.
        text_tokens = self.language_proj(out.last_hidden_state)
        return text_tokens, attention_mask.to(torch.bool)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns the predicted action chunk, shape (B, chunk_size, action_dim)."""
        vision_tokens = self._encode_images(batch)
        batch_size = vision_tokens.shape[0]
        device = vision_tokens.device

        if self.state_proj is not None:
            state = batch[OBS_STATE]
            if state.ndim == 3:
                state = state[:, 0]
            state_token = self.state_proj(state).unsqueeze(1) + self.state_embed
            vision_tokens = torch.cat([state_token, vision_tokens], dim=1)

        tasks = batch.get("task")
        if tasks is None:
            raise ValueError(
                "TurboVLA requires a task description, but the batch has no `task` key."
            )
        text_tokens, text_mask = self._encode_language(tasks, batch_size, device)

        vision_tokens, text_tokens = self.vl_interaction(
            vision_tokens, text_tokens, text_padding_mask=text_mask
        )

        # The decoder cross-attends over both fused streams. Vision tokens are always valid; the
        # text tokens carry the tokenizer's padding mask.
        memory = torch.cat([vision_tokens, text_tokens], dim=1)
        memory_mask = torch.cat(
            [
                torch.ones(
                    vision_tokens.shape[:2], dtype=torch.bool, device=device
                ),
                text_mask,
            ],
            dim=1,
        )

        return self.action_decoder(memory, memory_padding_mask=memory_mask)


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TurboVLA needs `transformers` for its DINOv3 and BERT backbones. Install it with "
            "`pip install 'transformers>=5.4.0,<5.6.0'`."
        ) from e


def _load_vision_backbone(config: TurboVLAConfig) -> tuple[nn.Module, int, int]:
    """Returns `(module, hidden_width, patch_size)`."""
    _require_transformers()
    from transformers import AutoConfig, AutoModel

    try:
        backbone_config = AutoConfig.from_pretrained(config.vision_backbone)
        if config.load_pretrained_backbones:
            model = AutoModel.from_pretrained(config.vision_backbone)
        else:
            model = AutoModel.from_config(backbone_config)
    except OSError as e:
        raise OSError(f"Could not load vision backbone '{config.vision_backbone}'. {_GATED_BACKBONE_HINT}") from e

    patch_size = getattr(backbone_config, "patch_size", None)
    if patch_size is None:
        raise ValueError(
            f"Vision backbone '{config.vision_backbone}' has no `patch_size` in its config; "
            "TurboVLA expects a patch-based vision transformer such as DINOv3."
        )
    return model, backbone_config.hidden_size, patch_size


def _load_language_backbone(config: TurboVLAConfig) -> tuple[nn.Module, int]:
    """Returns `(module, hidden_width)`."""
    _require_transformers()
    from transformers import AutoConfig, AutoModel

    backbone_config = AutoConfig.from_pretrained(config.language_backbone)
    if config.load_pretrained_backbones:
        model = AutoModel.from_pretrained(config.language_backbone)
    else:
        model = AutoModel.from_config(backbone_config)
    return model, backbone_config.hidden_size


def _load_tokenizer(config: TurboVLAConfig):
    _require_transformers()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.language_backbone)
