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
import math
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

UPSTREAM_COMMIT = "a3365e8f1787d30bafd99d2806edbf4dc82b5370"
LIBERO_CHECKPOINT = "hf://H-EmbodVis/TurboVLA/checkpoints/libero/turbovla_libero.pth"


@LRSchedulerConfig.register_subclass("turbovla_warmup_cosine")
@dataclass
class TurboVLASchedulerConfig(LRSchedulerConfig):
    """Upstream's `build_scheduler` (turbovla/training/trainer.py): the LR multiplier ramps linearly
    from 0.1 to 1.0 over the warmup, then follows a cosine from 1.0 down to `min_lr_ratio`. The released
    LIBERO recipe uses `min_lr_ratio=1.0`, i.e. a constant LR after warmup."""

    num_warmup_steps: int = 10_000
    min_lr_ratio: float = 1.0

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        warmup, min_ratio = self.num_warmup_steps, self.min_lr_ratio

        def lr_lambda(step):
            if step < warmup:
                return 0.1 + 0.9 * float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, num_training_steps - warmup))
            progress = min(max(progress, 0.0), 1.0)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(progress * math.pi))

        return LambdaLR(optimizer, lr_lambda)


@PreTrainedConfig.register_subclass("turbovla_so101")
@dataclass
class TurboVLAConfig(PreTrainedConfig):
    """TurboVLA (arXiv 2607.27205) as a LeRobot policy, faithful to the authors' code.

    The network is upstream's own code, vendored unchanged from `H-EmbodVis/TurboVLA` @ `a3365e8` in
    `upstream/` (see that package's file headers for the few compatibility patches). Every default here
    is upstream's released LIBERO recipe (`experiments/libero/README.md` + trainer defaults), so the
    released LIBERO checkpoint loads into this policy tensor for tensor (`upstream_init_checkpoint`).

    Architecture: DINOv3 ViT-B/16 per camera (trainable, 256 px) -> VisionProjection (LN, MLP + linear
    skip, LN) + view embedding; frozen BERT-base with Grounding-DINO sub-sentence masks -> linear
    projection; 6 x [BiAttentionBlock (one shared affinity, layer-scale 1e-4, drop-path 0.1) + text
    self-attention layer]; robot state -> 2 tokens appended to the fused memory; ACT decoder (3 pre-norm
    layers, FFN 2048, 12 learned queries) -> 3-layer MLP -> tanh. Actions are min-max normalised to
    [-1, 1] (hence the tanh), state is z-scored, images get ImageNet normalisation inside the model.
    Loss: masked L1.

    Plugin-only options, off by default (not in the paper): `state_dropout_prob`.

    Camera order matters: view 0 / view 1 index the view embedding. LIBERO's checkpoint was trained with
    view 0 = the third-person camera and view 1 = the wrist camera; set `camera_order` so the same holds.

    Args (beyond the LeRobot standard ones):
        upstream_init_checkpoint: Upstream TurboVLA checkpoint (`.pth`) to initialise from, as a local
            path or `hf://<repo>/<path>`. Every tensor whose shape matches is loaded; the ones that depend
            on the robot (state / action width, number of action queries when `chunk_size` != 12) keep
            their fresh init and are listed in the log. The paper's real-robot runs start from
            `LIBERO_CHECKPOINT`. Only used when training starts, never when a trained policy is loaded.
        state_dropout_prob: Plugin-only (not in the paper): during training, replace the normalised
            state by zeros (the dataset mean) for this fraction of samples, so the arm's own pose cannot
            stand in for the instruction. 0 = upstream behaviour.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 12
    n_action_steps: int = 12
    camera_order: list[str] | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Backbones.
    vision_backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    language_backbone: str = "google-bert/bert-base-uncased"
    load_pretrained_backbones: bool = True
    freeze_vision_backbone: bool = False
    freeze_language_backbone: bool = True
    image_size: int = 256
    vision_dropout: float = 0.1
    dinov3_precision: str = "bf16_autocast"

    # Text.
    max_language_tokens: int = 256
    text_padding_length: int | None = 21
    sub_sentence_present: bool = True
    zero_padded_text: bool = False

    # Vision-language interaction.
    dim_model: int = 256
    n_heads: int = 8
    n_interaction_layers: int = 6
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    interaction_precision: str = "fp32"

    # Action head.
    dim_feedforward: int = 2048
    n_decoder_layers: int = 3
    num_state_tokens: int = 2
    state_hidden_dim: int = 256
    action_mlp_hidden_dim: int = 512
    action_dropout: float = 0.1

    # Initialisation and plugin-only options.
    upstream_init_checkpoint: str | None = None
    state_dropout_prob: float = 0.0

    # Training preset: upstream's trainer defaults. AdamW keeps torch's default betas (upstream passes
    # none; the paper states none). Weight decay 1e-10 on weight matrices, 0 on biases / norms.
    optimizer_lr: float = 5e-5
    optimizer_lr_backbone: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 10_000
    scheduler_min_lr_ratio: float = 1.0

    # Rollout smoothing.
    temporal_ensemble_coeff: float | None = None

    def __post_init__(self):
        super().__post_init__()
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
        if not 0.0 <= self.state_dropout_prob < 1.0:
            raise ValueError(f"`state_dropout_prob` must be in [0, 1). Got {self.state_dropout_prob}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        # Per-group LR and weight decay come from `TurboVLAPolicy.get_optim_params`.
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> TurboVLASchedulerConfig:
        return TurboVLASchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps, min_lr_ratio=self.scheduler_min_lr_ratio
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError(
                "TurboVLA requires at least one camera: no `observation.images.*` key was found in "
                "`input_features`."
            )
        if self.robot_state_feature is None:
            raise ValueError(
                "TurboVLA conditions its action decoder on the robot state: add `observation.state`."
            )
        if self.action_feature is None:
            raise ValueError("TurboVLA requires an `action` key in `output_features`.")
        if self.camera_order is not None and sorted(self.camera_order) != sorted(self.image_features):
            raise ValueError(
                f"`camera_order` {self.camera_order} must list exactly the image features "
                f"{list(self.image_features)}."
            )

    @property
    def ordered_camera_keys(self) -> list[str]:
        return list(self.camera_order) if self.camera_order is not None else list(self.image_features)

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
