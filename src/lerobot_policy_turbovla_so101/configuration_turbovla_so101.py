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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("turbovla_so101")
@dataclass
class TurboVLAConfig(PreTrainedConfig):
    """Configuration class for the TurboVLA policy.

    Defaults follow the paper's LIBERO recipe, which is also the recommended starting point for
    SO-101 real-arm data (6-DoF + gripper is close enough to the 7-D case, and `chunk_size=12` at
    30 fps is a sane chunk).

    TurboVLA replaces the usual V -> L -> A pathway (an LLM in the middle) with a direct V + L -> A
    map: DINOv3 patch tokens and BERT *token-level* embeddings are fused by N bidirectional
    cross-attention layers, then an ACT-style decoder emits the whole action chunk in one parallel
    forward pass. There is no action tokenization and no autoregressive decode.

    The parameters you will most likely need to change are the ones which depend on the environment
    / sensors. Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - At least one key starting with "observation.images." is required as an input. If there are
          several, they are treated as multiple camera views and each gets its own camera-view
          embedding. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.
        - The task string is required. It arrives in the batch as `task` and is the whole point of
          this policy: without it TurboVLA is just a worse ACT.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy. Only
            1 is supported (upstream TurboVLA is single-frame, as is ACT).
        chunk_size: The size of the action prediction "chunks" in units of environment steps. This
            is `H`, the number of learnable action queries in the decoder.
        n_action_steps: The number of action steps to run in the environment for one invocation of
            the policy. This should be no greater than the chunk size.
        vision_backbone: HF repo id of the DINOv3 vision encoder. NOTE: the `facebook/dinov3-*`
            repos are *gated* — accept the terms on the model page and log in with `hf auth login`
            (or set `HF_TOKEN`) before training, otherwise loading fails with a 401.
        language_backbone: HF repo id of the language encoder. Token-level embeddings are used, not
            a pooled sentence embedding — that is what preserves object / attribute / spatial
            grounding. The paper reports the encoder is swappable (T5-small 97.1%, SigLIP-B 95.5%).
        load_pretrained_backbones: Whether to download pretrained backbone weights. Set to False to
            build both backbones from their configs with random init, which is the only way to
            construct the policy without Hub access (used by the tests).
        freeze_vision_backbone: Freeze DINOv3. Dominates both VRAM and final quality.
        freeze_language_backbone: Freeze the language encoder.
        image_size: Images are resized to this square resolution before the vision backbone. Must be
            divisible by the backbone's patch size (16 for the `dinov3-vit*16` family).
        max_language_tokens: Max instruction length in tokens; longer instructions are truncated.
        dim_model: The shared width `d` of the fusion trunk and the action decoder.
        n_heads: Number of attention heads in the fusion and decoder blocks.
        dim_feedforward: Hidden width of the feed-forward layers.
        feedforward_activation: Activation used in the feed-forward layers.
        dropout: Dropout used in the fusion and decoder blocks.
        n_fusion_layers: `N`, the number of bidirectional vision-language cross-attention layers.
        n_decoder_layers: Number of ACT-style action-decoder layers.
        optimizer_lr: Peak learning rate for the newly initialized trunk.
        optimizer_lr_backbone: Learning rate for the pretrained backbones. Ignored for whichever
            backbones are frozen.
        temporal_ensemble_coeff: Coefficient for exponentially-weighted temporal ensembling across
            overlapping action chunks (as in ACT's Algorithm 2), which smooths out the visible
            per-chunk jitter a single-shot decoder like this one otherwise has at rollout. `None`
            (default) disables it and uses a plain action queue instead. Requires
            `n_action_steps == 1` when set, since the ensembler needs a new chunk every step to
            average over.
        compile_model: Whether to wrap the fusion + action-decoder forward pass in `torch.compile`.
            Off by default since it adds a warmup-compilation cost on the first call; worth turning
            on for repeated rollout on the same shapes, since inference speed is the whole point of
            this architecture.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 12
    n_action_steps: int = 12

    # Images are fed to DINOv3, which expects its own ImageNet-style normalization. We therefore
    # keep the visual stream untouched by the dataset normalizer and apply the backbone's
    # normalization inside the model, where it belongs.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture: backbones.
    vision_backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    language_backbone: str = "google-bert/bert-base-uncased"
    load_pretrained_backbones: bool = True
    freeze_vision_backbone: bool = True
    freeze_language_backbone: bool = True
    image_size: int = 224
    max_language_tokens: int = 32

    # Architecture: shared trunk.
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 1024
    feedforward_activation: str = "relu"
    dropout: float = 0.1
    n_fusion_layers: int = 6
    n_decoder_layers: int = 4

    # Training preset.
    optimizer_lr: float = 5e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 5e-6
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 10_000
    scheduler_decay_steps: int = 80_000
    scheduler_decay_lr: float = 5e-7

    # Rollout smoothing and inference speed.
    temporal_ensemble_coeff: float | None = None
    compile_model: bool = False

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise ValueError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is because the "
                "ensembling algorithm needs to compute the mean over all predictions for a given "
                "action, so we need to query the policy every step and produce a single action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model "
                f"invocation. Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} "
                f"for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )
        if self.dim_model % self.n_heads != 0:
            raise ValueError(
                f"`dim_model` must be divisible by `n_heads`. Got {self.dim_model} and {self.n_heads}."
            )
        if self.image_size <= 0:
            raise ValueError(f"`image_size` must be positive. Got {self.image_size}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError(
                "TurboVLA requires at least one camera: no `observation.images.*` key was found in "
                "`input_features`."
            )
        if self.action_feature is None:
            raise ValueError("TurboVLA requires an `action` key in `output_features`.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
