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
"""Tests for the TurboVLA LeRobot policy plugin.

Everything here runs on CPU against tiny randomly initialized backbones, built from locally
constructed `transformers` configs. No Hub access, no GPU, no gated weights.
"""

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from transformers import AutoModel, BertConfig, DINOv3ViTConfig

from lerobot_policy_turbovla_so101 import modeling_turbovla_so101
from lerobot_policy_turbovla_so101.configuration_turbovla_so101 import TurboVLAConfig
from lerobot_policy_turbovla_so101.modeling_turbovla_so101 import TurboVLAPolicy
from lerobot_policy_turbovla_so101.processor_turbovla_so101 import make_turbovla_so101_pre_post_processors

IMAGE_SIZE = 64
PATCH_SIZE = 16
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
VISION_WIDTH = 32
LANGUAGE_WIDTH = 24
VOCAB_SIZE = 40
STATE_DIM = 6
ACTION_DIM = 7
CHUNK = 4


class _StubTokenizer:
    """Deterministic stand-in for a BERT tokenizer: hashes words into ids, pads to the longest."""

    def __call__(self, texts, padding=True, truncation=True, max_length=8, return_tensors="pt"):
        batch = [[(hash(w) % (VOCAB_SIZE - 2)) + 2 for w in t.split()][:max_length] or [1] for t in texts]
        width = max(len(ids) for ids in batch)
        input_ids = torch.zeros(len(batch), width, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), width, dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture(autouse=True)
def tiny_backbones(monkeypatch):
    """Swap the three Hub loaders for tiny local models so the suite runs offline."""

    def fake_vision(config):
        cfg = DINOv3ViTConfig(
            hidden_size=VISION_WIDTH,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=2 * VISION_WIDTH,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
        )
        return AutoModel.from_config(cfg), cfg.hidden_size, cfg.patch_size

    def fake_language(config):
        cfg = BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=LANGUAGE_WIDTH,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=2 * LANGUAGE_WIDTH,
            max_position_embeddings=32,
        )
        return AutoModel.from_config(cfg), cfg.hidden_size

    monkeypatch.setattr(modeling_turbovla_so101, "_load_vision_backbone", fake_vision)
    monkeypatch.setattr(modeling_turbovla_so101, "_load_language_backbone", fake_language)
    monkeypatch.setattr(modeling_turbovla_so101, "_load_tokenizer", lambda config: _StubTokenizer())


def make_config(n_cameras: int = 2, **overrides) -> TurboVLAConfig:
    image_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    input_features = {
        f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
        for i in range(n_cameras)
    }
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    kwargs = {
        "input_features": input_features,
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "device": "cpu",
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_fusion_layers": 2,
        "n_decoder_layers": 2,
        "image_size": IMAGE_SIZE,
        "max_language_tokens": 8,
    }
    kwargs.update(overrides)
    return TurboVLAConfig(**kwargs)


def make_batch(batch_size: int = 2, n_cameras: int = 2, with_action: bool = True) -> dict:
    batch = {
        f"observation.images.cam{i}": torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        for i in range(n_cameras)
    }
    batch["observation.state"] = torch.randn(batch_size, STATE_DIM)
    batch["task"] = [f"pick up the block number {i}" for i in range(batch_size)]
    if with_action:
        batch["action"] = torch.randn(batch_size, CHUNK, ACTION_DIM)
        batch["action_is_pad"] = torch.zeros(batch_size, CHUNK, dtype=torch.bool)
    return batch


# --------------------------------------------------------------------------------------------
# Plugin wiring
# --------------------------------------------------------------------------------------------


def test_name_is_spelled_identically_everywhere():
    """The registered name, the policy `name`, and the processor factory must agree."""
    from lerobot.configs import PreTrainedConfig

    assert "turbovla_so101" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("turbovla_so101") is TurboVLAConfig
    assert TurboVLAPolicy.name == "turbovla_so101"
    assert TurboVLAPolicy.config_class is TurboVLAConfig
    assert make_turbovla_so101_pre_post_processors.__name__ == "make_turbovla_so101_pre_post_processors"


def test_factory_resolves_policy_class_by_convention():
    """`configuration_turbovla_so101.TurboVLAConfig` -> `modeling_turbovla_so101.TurboVLAPolicy`."""
    from lerobot.policies.factory import get_policy_class

    assert get_policy_class("turbovla_so101") is TurboVLAPolicy


def test_distribution_name_carries_the_discovery_prefix():
    """LeRobot finds plugins by distribution-name prefix, so this name is load-bearing."""
    import importlib.metadata as md

    names = {d.metadata.get("Name") for d in md.distributions()}
    assert "lerobot_policy_turbovla_so101" in names


def test_processor_factory_builds_pipelines():
    config = make_config()
    pre, post = make_turbovla_so101_pre_post_processors(config, dataset_stats=None)
    assert pre is not None and post is not None


# --------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------


def test_action_delta_indices_span_the_chunk():
    assert make_config().action_delta_indices == list(range(CHUNK))
    assert make_config().observation_delta_indices is None
    assert make_config().reward_delta_indices is None


def test_n_action_steps_cannot_exceed_chunk_size():
    with pytest.raises(ValueError, match="chunk size is the upper bound"):
        make_config(chunk_size=4, n_action_steps=8)


def test_multiple_observation_steps_rejected():
    with pytest.raises(ValueError, match="Multiple observation steps"):
        make_config(n_obs_steps=2)


def test_dim_model_must_divide_by_n_heads():
    with pytest.raises(ValueError, match="divisible"):
        make_config(dim_model=30, n_heads=4)


def test_validate_features_requires_a_camera():
    config = make_config(n_cameras=0)
    with pytest.raises(ValueError, match="at least one camera"):
        config.validate_features()


def test_image_size_must_divide_by_patch_size():
    with pytest.raises(ValueError, match="divisible by the vision backbone"):
        TurboVLAPolicy(make_config(image_size=72))


def test_optimizer_and_scheduler_presets():
    config = make_config()
    assert config.get_optimizer_preset().lr == config.optimizer_lr
    scheduler = config.get_scheduler_preset()
    assert scheduler.num_warmup_steps == config.scheduler_warmup_steps
    assert scheduler.peak_lr == config.optimizer_lr


# --------------------------------------------------------------------------------------------
# Forward / loss
# --------------------------------------------------------------------------------------------


def test_forward_returns_scalar_l1_loss():
    policy = TurboVLAPolicy(make_config())
    loss, loss_dict = policy.forward(make_batch())
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert set(loss_dict) == {"l1_loss"}


def test_forward_requires_the_task_string():
    """Without language this is just a worse ACT, so it must fail loudly."""
    policy = TurboVLAPolicy(make_config())
    batch = make_batch()
    del batch["task"]
    with pytest.raises(ValueError, match="requires a task description"):
        policy.forward(batch)


def test_padded_actions_are_excluded_from_the_loss():
    policy = TurboVLAPolicy(make_config())
    torch.manual_seed(0)
    batch = make_batch()

    all_valid, _ = policy.forward(batch)

    # Corrupt the tail of the chunk, then mark it as padding: the loss must not move.
    padded = dict(batch)
    padded["action"] = batch["action"].clone()
    padded["action"][:, -2:] = 1e4
    pad_mask = torch.zeros_like(batch["action_is_pad"])
    pad_mask[:, -2:] = True
    padded["action_is_pad"] = pad_mask

    masked_ref = dict(batch)
    masked_ref["action_is_pad"] = pad_mask

    torch.manual_seed(0)
    loss_padded, _ = policy.forward(padded)
    torch.manual_seed(0)
    loss_ref, _ = policy.forward(masked_ref)
    assert torch.allclose(loss_padded, loss_ref)
    assert not torch.isclose(loss_padded, all_valid)


def test_predicted_chunk_has_the_right_shape():
    policy = TurboVLAPolicy(make_config())
    chunk = policy.predict_action_chunk(make_batch(batch_size=3, with_action=False))
    assert chunk.shape == (3, CHUNK, ACTION_DIM)


def test_works_without_a_state_feature():
    config = make_config()
    del config.input_features["observation.state"]
    policy = TurboVLAPolicy(config)
    batch = make_batch()
    del batch["observation.state"]
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)


def test_single_camera_and_three_cameras_both_work():
    for n_cameras in (1, 3):
        policy = TurboVLAPolicy(make_config(n_cameras=n_cameras))
        loss, _ = policy.forward(make_batch(n_cameras=n_cameras))
        assert torch.isfinite(loss)


def test_images_are_resized_to_the_configured_resolution():
    policy = TurboVLAPolicy(make_config())
    batch = make_batch(with_action=False)
    for i in range(2):
        batch[f"observation.images.cam{i}"] = torch.rand(2, 3, 96, 120)
    assert policy.predict_action_chunk(batch).shape == (2, CHUNK, ACTION_DIM)


def test_a_task_string_may_be_shared_across_the_batch():
    policy = TurboVLAPolicy(make_config())
    batch = make_batch(batch_size=3, with_action=False)
    batch["task"] = "pick up the red block"
    assert policy.predict_action_chunk(batch).shape == (3, CHUNK, ACTION_DIM)


# --------------------------------------------------------------------------------------------
# The parts that make this TurboVLA and not ACT
# --------------------------------------------------------------------------------------------


def test_predictions_depend_on_the_instruction():
    """The whole point: same pixels, different task -> different actions."""
    policy = TurboVLAPolicy(make_config())
    policy.eval()
    batch = make_batch(batch_size=1, with_action=False)

    batch["task"] = ["pick up the red block"]
    first = policy.predict_action_chunk(batch)
    batch["task"] = ["push the blue drawer shut"]
    second = policy.predict_action_chunk(batch)

    assert not torch.allclose(first, second)


def test_fusion_runs_in_both_directions():
    """Vision must respond to text *and* text must respond to vision."""
    from lerobot_policy_turbovla_so101.vl_interaction import TurboVLAVLInteraction

    config = make_config()
    fusion = TurboVLAVLInteraction(config).eval()

    vision = torch.randn(1, 5, config.dim_model)
    text_a = torch.randn(1, 3, config.dim_model)
    text_b = torch.randn(1, 3, config.dim_model)
    mask = torch.ones(1, 3, dtype=torch.bool)

    vision_a, out_text_a = fusion(vision, text_a, text_padding_mask=mask)
    vision_b, _ = fusion(vision, text_b, text_padding_mask=mask)
    # text -> vision: swapping the text changes the fused vision tokens.
    assert not torch.allclose(vision_a, vision_b)

    other_vision = torch.randn(1, 5, config.dim_model)
    _, out_text_c = fusion(other_vision, text_a, text_padding_mask=mask)
    # vision -> text: swapping the vision changes the fused text tokens.
    assert not torch.allclose(out_text_a, out_text_c)


def test_camera_views_are_distinguished():
    """Two cameras must not be interchangeable, otherwise the view embedding is doing nothing."""
    policy = TurboVLAPolicy(make_config(n_cameras=2))
    policy.eval()
    batch = make_batch(batch_size=1, with_action=False)
    cam0, cam1 = batch["observation.images.cam0"], batch["observation.images.cam1"]

    straight = policy.predict_action_chunk(batch)
    batch["observation.images.cam0"], batch["observation.images.cam1"] = cam1, cam0
    swapped = policy.predict_action_chunk(batch)

    assert not torch.allclose(straight, swapped)


def test_padding_mask_makes_trailing_text_tokens_inert():
    """Masked-out instruction tokens must not leak into the prediction."""
    from lerobot_policy_turbovla_so101.vl_interaction import TurboVLAVLInteraction

    config = make_config()
    fusion = TurboVLAVLInteraction(config).eval()

    vision = torch.randn(1, 5, config.dim_model)
    text = torch.randn(1, 4, config.dim_model)
    mask = torch.tensor([[True, True, False, False]])

    vision_a, _ = fusion(vision, text, text_padding_mask=mask)

    clobbered = text.clone()
    clobbered[:, 2:] = 1e3  # only the masked-out positions
    vision_b, _ = fusion(vision, clobbered, text_padding_mask=mask)

    assert torch.allclose(vision_a, vision_b, atol=1e-5)


def test_no_vae_head():
    """TurboVLA trains on plain L1 — ACT's VAE machinery should be absent."""
    policy = TurboVLAPolicy(make_config())
    names = [n for n, _ in policy.named_parameters()]
    assert not any("vae" in n for n in names)
    assert not hasattr(policy.config, "use_vae")


# --------------------------------------------------------------------------------------------
# Training mechanics
# --------------------------------------------------------------------------------------------


def test_frozen_backbones_have_no_gradients_and_the_trunk_does():
    policy = TurboVLAPolicy(make_config())
    policy.train()
    loss, _ = policy.forward(make_batch())
    loss.backward()

    vision_grads = [p.grad for p in policy.model.vision_encoder.parameters()]
    language_grads = [p.grad for p in policy.model.language_encoder.parameters()]
    assert all(g is None for g in vision_grads)
    assert all(g is None for g in language_grads)

    trunk_grads = [p.grad for p in policy.model.vl_interaction.parameters()]
    assert all(g is not None for g in trunk_grads)
    assert any(g.abs().sum() > 0 for g in trunk_grads)
    assert policy.model.action_decoder.query_embed.weight.grad is not None


def test_unfrozen_backbones_do_get_gradients():
    policy = TurboVLAPolicy(make_config(freeze_vision_backbone=False, freeze_language_backbone=False))
    policy.train()
    loss, _ = policy.forward(make_batch())
    loss.backward()
    assert any(p.grad is not None for p in policy.model.vision_encoder.parameters())
    assert any(p.grad is not None for p in policy.model.language_encoder.parameters())


def test_frozen_backbones_stay_in_eval_mode_after_train():
    policy = TurboVLAPolicy(make_config())
    policy.train()
    assert not policy.model.vision_encoder.training
    assert not policy.model.language_encoder.training
    assert policy.model.vl_interaction.training


def test_optim_params_split_trunk_from_backbones():
    config = make_config(freeze_vision_backbone=False, freeze_language_backbone=False)
    policy = TurboVLAPolicy(config)
    groups = policy.get_optim_params()
    assert len(groups) == 2
    assert "lr" not in groups[0]
    assert groups[1]["lr"] == config.optimizer_lr_backbone
    assert len(groups[1]["params"]) > 0

    total = sum(len(g["params"]) for g in groups)
    assert total == len([p for p in policy.parameters() if p.requires_grad])


def test_optim_params_backbone_group_is_empty_when_frozen():
    policy = TurboVLAPolicy(make_config())
    groups = policy.get_optim_params()
    assert len(groups[1]["params"]) == 0


def test_one_training_step_reduces_the_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    policy = TurboVLAPolicy(make_config())
    policy.train()
    batch = make_batch()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)

    first, _ = policy.forward(batch)
    for _ in range(12):
        optimizer.zero_grad()
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
    final, _ = policy.forward(batch)
    assert final.item() < first.item()


# --------------------------------------------------------------------------------------------
# Rollout mechanics
# --------------------------------------------------------------------------------------------


def test_select_action_serves_one_action_at_a_time_from_the_queue():
    config = make_config(n_action_steps=CHUNK)
    policy = TurboVLAPolicy(config)
    batch = make_batch(batch_size=1, with_action=False)

    actions = [policy.select_action(batch) for _ in range(CHUNK)]
    assert all(a.shape == (1, ACTION_DIM) for a in actions)
    assert len(policy._action_queue) == 0


def test_reset_clears_the_queue_between_episodes():
    policy = TurboVLAPolicy(make_config())
    batch = make_batch(batch_size=1, with_action=False)

    policy.select_action(batch)
    assert len(policy._action_queue) == CHUNK - 1
    policy.reset()
    assert len(policy._action_queue) == 0


def test_n_action_steps_smaller_than_chunk_truncates_the_queue():
    policy = TurboVLAPolicy(make_config(chunk_size=CHUNK, n_action_steps=2))
    batch = make_batch(batch_size=1, with_action=False)
    policy.select_action(batch)
    assert len(policy._action_queue) == 1


def test_temporal_ensemble_requires_single_action_step():
    with pytest.raises(ValueError, match="n_action_steps"):
        make_config(temporal_ensemble_coeff=0.01, n_action_steps=2)


def test_temporal_ensemble_returns_one_action_per_call_without_draining_a_queue():
    policy = TurboVLAPolicy(make_config(temporal_ensemble_coeff=0.01, n_action_steps=1))
    batch = make_batch(batch_size=1, with_action=False)

    action = policy.select_action(batch)
    assert action.shape == (1, ACTION_DIM)
    assert not hasattr(policy, "_action_queue") or len(policy._action_queue) == 0

    # Should keep returning single actions across repeated calls, no queue to drain.
    for _ in range(CHUNK * 2):
        action = policy.select_action(batch)
        assert action.shape == (1, ACTION_DIM)


def test_temporal_ensemble_resets_between_episodes():
    policy = TurboVLAPolicy(make_config(temporal_ensemble_coeff=0.01, n_action_steps=1))
    batch = make_batch(batch_size=1, with_action=False)

    policy.select_action(batch)
    assert policy.temporal_ensembler.ensembled_actions is not None
    policy.reset()
    assert policy.temporal_ensembler.ensembled_actions is None


def test_select_action_is_deterministic_in_eval():
    policy = TurboVLAPolicy(make_config())
    policy.eval()
    batch = make_batch(batch_size=1, with_action=False)

    first = policy.select_action(batch)
    policy.reset()
    second = policy.select_action(batch)
    assert torch.allclose(first, second)


# --------------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    policy = TurboVLAPolicy(make_config())
    policy.eval()
    batch = make_batch(batch_size=1, with_action=False)
    before = policy.predict_action_chunk(batch)

    policy.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").exists()

    reloaded = TurboVLAPolicy.from_pretrained(tmp_path, config=policy.config)
    reloaded.eval()
    after = reloaded.predict_action_chunk(batch)
    assert torch.allclose(before, after, atol=1e-6)
