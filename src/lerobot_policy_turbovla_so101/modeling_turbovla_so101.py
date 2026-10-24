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
"""TurboVLA policy: upstream's network, LeRobot's harness.

The network (`self.model`) is `H-EmbodVis/TurboVLA`'s own `TurboVLA` module, vendored in `upstream/`,
so its parameter names are upstream's and a released checkpoint loads into it directly. This file only
does what upstream's data loader and trainer did around it:

- images: resize to `image_size` (bilinear, antialiased), ImageNet-normalise, stack the cameras in
  `camera_order` -> `{"dinov3": (B, V, 3, H, W)}` (upstream feeds pre-resized 256-px frames through the
  DINOv3 processor with resize disabled, which is the same rescale + normalise);
- state and actions arrive normalised by LeRobot's processor (MEAN_STD state, MIN_MAX actions to
  [-1, 1]), as upstream's loader normalises them;
- loss: upstream's `masked_l1_loss`;
- optimiser groups: upstream's `build_param_group_optimizer` (DINOv3 vs head, weight decay only on
  weight matrices).
"""

import contextvars
import logging
from collections import deque
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor, nn

from .configuration_turbovla_so101 import UPSTREAM_COMMIT, TurboVLAConfig
from .temporal_ensemble import TemporalEnsembler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Set while `from_pretrained` builds the policy: the saved weights replace everything right after, so
# neither the upstream init checkpoint nor the pretrained backbone weights need to be fetched.
_LOADING_SAVED_POLICY = contextvars.ContextVar("turbovla_loading_saved_policy", default=False)


def masked_l1_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Upstream `turbovla/training/trainer.py` `masked_l1_loss`, unchanged."""
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def _use_weight_decay(name: str, param: Tensor) -> bool:
    """Upstream `turbovla/training/trainer.py` `_use_weight_decay`, unchanged."""
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def build_upstream_config(config: TurboVLAConfig, pretrained_backbones: bool):
    """The upstream `TurboVLAConfig` this LeRobot config describes."""
    from .upstream.configuration import (
        ActionHeadConfig,
        InteractionConfig,
        TextEncoderConfig,
        VisionEncoderConfig,
    )
    from .upstream.configuration import TurboVLAConfig as UpstreamConfig

    return UpstreamConfig(
        compatibility_profile="libero",
        text=TextEncoderConfig(
            model_name_or_path=config.language_backbone,
            max_length=config.max_language_tokens,
            padding_length=config.text_padding_length,
            sub_sentence_present=config.sub_sentence_present,
            frozen=config.freeze_language_backbone,
            force_eval_when_frozen=True,
            zero_padded_tokens=config.zero_padded_text,
            pretrained=pretrained_backbones,
        ),
        vision=VisionEncoderConfig(
            model_name_or_path=config.vision_backbone,
            image_size=config.image_size,
            num_views=len(config.image_features),
            position_embedding="view",
            encode_views_separately=True,
            frozen=config.freeze_vision_backbone,
            compute_precision=config.dinov3_precision,
            dropout=config.vision_dropout,
            pretrained=pretrained_backbones,
        ),
        interaction=InteractionConfig(
            hidden_dim=config.dim_model,
            nheads=config.n_heads,
            num_layers=config.n_interaction_layers,
            dim_feedforward=config.dim_feedforward,
            enhancer_inner_dim=config.enhancer_inner_dim,
            text_dropout=config.text_dropout,
            fusion_dropout=config.fusion_dropout,
            fusion_droppath=config.fusion_droppath,
            compute_precision=config.interaction_precision,
        ),
        action=ActionHeadConfig(
            action_dim=config.action_feature.shape[0],
            state_dim=config.robot_state_feature.shape[0],
            horizon=config.chunk_size,
            num_state_tokens=config.num_state_tokens,
            num_layers=config.n_decoder_layers,
            mlp_hidden_dim=config.action_mlp_hidden_dim,
            state_hidden_dim=config.state_hidden_dim,
            dropout=config.action_dropout,
        ),
    )


def build_upstream_model(config: TurboVLAConfig, pretrained_backbones: bool) -> nn.Module:
    from .upstream.turbovla import TurboVLA

    return TurboVLA(build_upstream_config(config, pretrained_backbones))


def read_upstream_state_dict(path: str) -> dict[str, Tensor]:
    """An upstream `.pth` (local path or `hf://<repo>/<file>`) -> its model state dict, read the way
    upstream's evaluator reads it (`ema_model_state_dict` first, then `model_state_dict`, `module.`
    prefixes stripped)."""
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        org, name, filename = path[len("hf://") :].split("/", 2)
        path = hf_hub_download(repo_id=f"{org}/{name}", filename=filename)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    state = checkpoint.get("ema_model_state_dict")
    if not isinstance(state, dict):
        state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError("expected `ema_model_state_dict` or `model_state_dict` in the upstream checkpoint")
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}


# transformers 5 nests DINOv3's blocks one level deeper (`DINOv3ViTModel.layer` -> `.model.layer`).
# Upstream checkpoints (and the released LIBERO one) were saved under transformers 4.56's names.
_BACKBONE_RENAMES = (("vision_encoder.backbone.layer.", "vision_encoder.backbone.model.layer."),)


def align_backbone_names(source: dict[str, Tensor], target_keys) -> tuple[dict[str, Tensor], int]:
    """Rename source keys between transformers 4.x and 5.x backbone layouts, in whichever direction
    the installed transformers needs. Returns the renamed dict and how many keys changed."""
    target_keys = set(target_keys)
    renamed, n = {}, 0
    for key, value in source.items():
        new = key
        if key not in target_keys:
            for old_prefix, new_prefix in _BACKBONE_RENAMES:
                for a, b in ((old_prefix, new_prefix), (new_prefix, old_prefix)):
                    if key.startswith(a) and b + key[len(a) :] in target_keys:
                        new = b + key[len(a) :]
        n += new != key
        renamed[new] = value
    return renamed, n


def load_upstream_weights(model: nn.Module, source: dict[str, Tensor]) -> dict:
    """Load every tensor whose name and shape match; tensors whose shape depends on the robot (state /
    action width, number of action queries) keep their fresh init. Any other mismatch, missing or unused
    tensor is an error: no silent partial load."""
    target = model.state_dict()
    source, n_renamed = align_backbone_names(source, target)
    reinitialised = sorted(k for k in target if k in source and source[k].shape != target[k].shape)
    missing = sorted(k for k in target if k not in source)
    unused = sorted(k for k in source if k not in target)
    robot_dependent = (
        "action_head.state_projection.",
        "action_head.decoder.action_projection.layers.2.",
        "action_head.decoder.action_queries.",
    )
    unexpected = [k for k in reinitialised if not k.startswith(robot_dependent)]
    if missing or unused or unexpected:
        raise RuntimeError(
            f"upstream checkpoint does not fit: missing {missing[:10]}, unused {unused[:10]}, "
            f"shape mismatch outside the robot-dependent tensors {unexpected[:10]}"
        )
    merged = {k: (target[k] if k in reinitialised else source[k]) for k in target}
    model.load_state_dict(merged, strict=True)
    return {
        "loaded": len(target) - len(reinitialised),
        "reinitialised": reinitialised,
        "total": len(target),
        "renamed_backbone_keys": n_renamed,
    }


class TurboVLAPolicy(PreTrainedPolicy):
    """TurboVLA (arXiv 2607.27205), upstream code @ a3365e8, behind LeRobot's policy interface."""

    config_class = TurboVLAConfig
    name = "turbovla_so101"
    _fsdp_wrap_modules = ["BiAttentionBlock", "TransformerEncoderLayer", "TransformerDecoderLayer"]

    def __init__(self, config: TurboVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        loading_saved = _LOADING_SAVED_POLICY.get()
        self.camera_keys = config.ordered_camera_keys

        self.model = build_upstream_model(
            config, pretrained_backbones=config.load_pretrained_backbones and not loading_saved
        )
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.init_report = None
        if config.upstream_init_checkpoint and not loading_saved:
            self.init_report = load_upstream_weights(
                self.model, read_upstream_state_dict(config.upstream_init_checkpoint)
            )
            self.init_report["checkpoint"] = config.upstream_init_checkpoint
            logging.info(
                f"TurboVLA init from {config.upstream_init_checkpoint} (upstream {UPSTREAM_COMMIT[:7]}): "
                f"{self.init_report['loaded']}/{self.init_report['total']} tensors loaded, re-initialised "
                f"{self.init_report['reinitialised']}"
            )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = TemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        token = _LOADING_SAVED_POLICY.set(True)
        try:
            return super().from_pretrained(*args, **kwargs)
        finally:
            _LOADING_SAVED_POLICY.reset(token)

    def get_optim_params(self) -> list[dict]:
        """Upstream's `build_param_group_optimizer`: DINOv3 vs everything else, weight decay only on
        weight matrices (not biases or norms). Frozen parameters (BERT) are left out, as upstream does."""
        c = self.config
        groups = {
            "dinov3_decay": (c.optimizer_lr_backbone, c.optimizer_weight_decay),
            "dinov3_no_decay": (c.optimizer_lr_backbone, 0.0),
            "head_decay": (c.optimizer_lr, c.optimizer_weight_decay),
            "head_no_decay": (c.optimizer_lr, 0.0),
        }
        params = {k: [] for k in groups}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            prefix = "dinov3" if name.startswith("vision_encoder.backbone") else "head"
            params[f"{prefix}_{'decay' if _use_weight_decay(name, param) else 'no_decay'}"].append(param)
        return [
            {"params": params[k], "lr": lr, "weight_decay": wd, "name": k}
            for k, (lr, wd) in groups.items()
            if params[k]
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _images(self, batch: dict[str, Tensor]) -> Tensor:
        views = []
        for key in self.camera_keys:
            images = batch[key]
            if images.ndim == 5:  # (B, T, C, H, W) with n_obs_steps == 1
                images = images[:, -1]
            if images.shape[-2:] != (self.config.image_size, self.config.image_size):
                images = F.interpolate(
                    images.float(),
                    size=(self.config.image_size, self.config.image_size),
                    mode="bilinear",
                    antialias=True,
                    align_corners=False,
                )
            views.append((images.float().clamp(0.0, 1.0) - self.image_mean) / self.image_std)
        return torch.stack(views, dim=1)

    def _tasks(self, batch: dict[str, Tensor], batch_size: int) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            raise ValueError(
                "TurboVLA requires a task description, but the batch has no `task` key. Record or "
                "re-export the dataset with a task string per episode."
            )
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        if len(tasks) != batch_size:
            raise ValueError(f"got {len(tasks)} task strings for a batch of {batch_size}")
        return [str(t) for t in tasks]

    def _state(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state[:, -1]
        p = self.config.state_dropout_prob
        if self.training and p > 0.0:
            keep = (torch.rand(state.shape[0], 1, device=state.device) >= p).to(state.dtype)
            state = state * keep  # normalised state: 0 is the dataset mean
        return state

    def predict_normalized_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        pixels = self._images(batch)
        return self.model(self._tasks(batch, pixels.shape[0]), {"dinov3": pixels}, self._state(batch))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """One action per call: from a queue refilled every `n_action_steps`, or, with
        `temporal_ensemble_coeff` set, a fresh chunk every call blended by ACT's temporal ensembling."""
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            return self.temporal_ensembler.update(self.predict_action_chunk(batch))
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        return self.predict_normalized_chunk(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        pred = self.predict_normalized_chunk(batch)
        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"]
        else:
            mask = torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool)
        # Upstream's loader clamps the min-max-normalised targets to [-1, 1] (the range of the tanh head).
        target = batch[ACTION].float().clamp(-1.0, 1.0)
        loss = masked_l1_loss(pred.float(), target, mask)
        return loss, {"l1_loss": loss.item()}
