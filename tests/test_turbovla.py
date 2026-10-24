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
"""Tests for the TurboVLA LeRobot policy plugin (LeRobot wiring, recipe, rollout mechanics).

That the network computes what upstream computes is `test_upstream_parity.py`'s job. Everything here
runs on CPU against tiny local backbones (`tiny_backbones.py`): no Hub access, no GPU, no gated weights.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from lerobot.configs import NormalizationMode
from test_upstream_parity import TINY, tiny_plugin_config
from tiny_backbones import IMAGE_SIZE, make_backbone_dirs

from lerobot_policy_turbovla_so101.configuration_turbovla_so101 import TurboVLAConfig, TurboVLASchedulerConfig
from lerobot_policy_turbovla_so101.modeling_turbovla_so101 import TurboVLAPolicy
from lerobot_policy_turbovla_so101.processor_turbovla_so101 import make_turbovla_so101_pre_post_processors

STATE_DIM = TINY["action"]["state_dim"]
ACTION_DIM = TINY["action"]["action_dim"]
CHUNK = TINY["action"]["horizon"]
CAMS = ("observation.images.front", "observation.images.wrist")
BOWLS = ("Put the blue bowl in the pink bowl.", "Put the pink bowl in the blue bowl.")


@pytest.fixture(scope="session")
def dirs(tmp_path_factory):
    return make_backbone_dirs(tmp_path_factory.mktemp("backbones"))


@pytest.fixture
def make_config(dirs):
    def _make(**overrides) -> TurboVLAConfig:
        return tiny_plugin_config(*dirs, **overrides)

    return _make


@pytest.fixture
def make_policy(make_config):
    def _make(seed: int = 0, **overrides) -> TurboVLAPolicy:
        torch.manual_seed(seed)
        return TurboVLAPolicy(make_config(**overrides))

    return _make


def make_batch(batch_size: int = 2, with_action: bool = True, size=(IMAGE_SIZE, IMAGE_SIZE), seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    batch = {cam: torch.rand(batch_size, 3, *size, generator=g) for cam in CAMS}
    batch["observation.state"] = torch.randn(batch_size, STATE_DIM, generator=g)
    batch["task"] = [BOWLS[i % 2] for i in range(batch_size)]
    if with_action:
        batch["action"] = torch.rand(batch_size, CHUNK, ACTION_DIM, generator=g) * 2 - 1
        batch["action_is_pad"] = torch.zeros(batch_size, CHUNK, dtype=torch.bool)
    return batch


# --------------------------------------------------------------------------------------------
# Plugin wiring
# --------------------------------------------------------------------------------------------


def test_name_is_spelled_identically_everywhere():
    from lerobot.configs import PreTrainedConfig

    assert "turbovla_so101" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("turbovla_so101") is TurboVLAConfig
    assert TurboVLAPolicy.name == "turbovla_so101"
    assert TurboVLAPolicy.config_class is TurboVLAConfig


def test_factory_resolves_policy_class_by_convention():
    from lerobot.policies.factory import get_policy_class

    assert get_policy_class("turbovla_so101") is TurboVLAPolicy


def test_distribution_name_carries_the_discovery_prefix():
    import importlib.metadata as md

    assert "lerobot_policy_turbovla_so101" in {d.metadata.get("Name") for d in md.distributions()}


def test_scheduler_is_registered_for_resume():
    from lerobot.optim.schedulers import LRSchedulerConfig

    assert LRSchedulerConfig.get_choice_class("turbovla_warmup_cosine") is TurboVLASchedulerConfig


def test_processor_factory_builds_pipelines(make_config):
    pre, post = make_turbovla_so101_pre_post_processors(make_config(), dataset_stats=None)
    assert pre is not None and post is not None


def test_processor_normalises_like_upstreams_loader(make_config):
    """Actions min-max to [-1, 1] (the tanh head's range), state z-scored, images left to the model."""
    config = make_config()
    stats = {
        "action": {"min": torch.full((ACTION_DIM,), -10.0), "max": torch.full((ACTION_DIM,), 30.0)},
        "observation.state": {"mean": torch.full((STATE_DIM,), 2.0), "std": torch.full((STATE_DIM,), 4.0)},
    }
    pre, post = make_turbovla_so101_pre_post_processors(config, dataset_stats=stats)
    out = pre(
        {
            "observation.state": torch.full((1, STATE_DIM), 6.0),
            "action": torch.full((1, ACTION_DIM), 30.0),
            CAMS[0]: torch.rand(1, 3, 8, 8),
            CAMS[1]: torch.rand(1, 3, 8, 8),
            "task": "x",
        }
    )
    torch.testing.assert_close(out["observation.state"], torch.full((1, STATE_DIM), 1.0))
    torch.testing.assert_close(out["action"], torch.full((1, ACTION_DIM), 1.0))
    torch.testing.assert_close(post(torch.full((1, ACTION_DIM), -1.0)), torch.full((1, ACTION_DIM), -10.0))


# --------------------------------------------------------------------------------------------
# Defaults are upstream's released LIBERO recipe
# --------------------------------------------------------------------------------------------


def test_defaults_are_upstreams_libero_recipe():
    c = TurboVLAConfig()
    assert (c.chunk_size, c.image_size, c.dim_model, c.n_heads, c.n_interaction_layers) == (
        12,
        256,
        256,
        8,
        6,
    )
    assert (c.enhancer_inner_dim, c.dim_feedforward, c.n_decoder_layers, c.num_state_tokens) == (
        1024,
        2048,
        3,
        2,
    )
    assert (c.state_hidden_dim, c.action_mlp_hidden_dim, c.text_padding_length) == (256, 512, 21)
    assert (c.fusion_dropout, c.fusion_droppath, c.text_dropout, c.vision_dropout, c.action_dropout) == (
        0.0,
        0.1,
        0.0,
        0.1,
        0.1,
    )
    assert not c.freeze_vision_backbone and c.freeze_language_backbone and c.sub_sentence_present
    assert (c.optimizer_lr, c.optimizer_lr_backbone, c.optimizer_weight_decay) == (5e-5, 5e-5, 1e-10)
    assert c.optimizer_betas == (0.9, 0.999)  # torch's default: upstream passes none, the paper states none
    assert (c.optimizer_grad_clip_norm, c.scheduler_warmup_steps, c.scheduler_min_lr_ratio) == (
        1.0,
        10_000,
        1.0,
    )
    assert c.normalization_mapping == {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MIN_MAX,
    }
    assert c.state_dropout_prob == 0.0 and c.upstream_init_checkpoint is None


def test_scheduler_matches_upstreams_build_scheduler():
    def upstream(step, max_steps, warmup, min_ratio):  # turbovla/training/trainer.py build_scheduler
        if step < warmup:
            return 0.1 + 0.9 * float(step + 1) / float(max(1, warmup))
        progress = min(max(float(step - warmup) / float(max(1, max_steps - warmup)), 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(progress * math.pi))

    for min_ratio in (1.0, 0.1):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = TurboVLASchedulerConfig(num_warmup_steps=10, min_lr_ratio=min_ratio).build(opt, 40)
        for step in range(45):
            assert opt.param_groups[0]["lr"] == pytest.approx(upstream(step, 40, 10, min_ratio))
            opt.step()
            sched.step()


# --------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------


def test_action_delta_indices_span_the_chunk(make_config):
    c = make_config()
    assert c.action_delta_indices == list(range(CHUNK))
    assert c.observation_delta_indices is None and c.reward_delta_indices is None


def test_n_action_steps_cannot_exceed_chunk_size(make_config):
    with pytest.raises(ValueError, match="chunk size is the upper bound"):
        make_config(n_action_steps=CHUNK + 1)


def test_multiple_observation_steps_rejected(make_config):
    with pytest.raises(ValueError, match="Multiple observation steps"):
        make_config(n_obs_steps=2)


def test_dim_model_must_divide_by_n_heads(make_config):
    with pytest.raises(ValueError, match="divisible"):
        make_config(dim_model=30, n_heads=4)


def test_state_dropout_must_be_a_probability(make_config):
    with pytest.raises(ValueError, match="state_dropout_prob"):
        make_config(state_dropout_prob=1.0)


def test_validate_features_requires_camera_state_and_matching_camera_order(make_config):
    c = make_config()
    c.input_features = {k: v for k, v in c.input_features.items() if "images" not in k}
    with pytest.raises(ValueError, match="camera"):
        c.validate_features()
    c = make_config()
    c.input_features.pop("observation.state")
    with pytest.raises(ValueError, match="observation.state"):
        c.validate_features()
    with pytest.raises(ValueError, match="camera_order"):
        make_config(camera_order=["observation.images.front"]).validate_features()


# --------------------------------------------------------------------------------------------
# Forward pass and loss
# --------------------------------------------------------------------------------------------


def test_forward_returns_scalar_l1_loss(make_policy):
    loss, info = make_policy().forward(make_batch())
    assert loss.ndim == 0 and loss.item() > 0 and info["l1_loss"] == pytest.approx(loss.item())


def test_forward_requires_the_task_string(make_policy):
    batch = make_batch()
    del batch["task"]
    with pytest.raises(ValueError, match="task"):
        make_policy().forward(batch)


def test_padded_actions_are_excluded_from_the_loss(make_policy):
    policy = make_policy().eval()  # no dropout: compare two forward passes
    batch = make_batch()
    batch["action_is_pad"][:, -4:] = True
    loss_a, _ = policy.forward(batch)
    batch["action"][:, -4:] = 123.0
    loss_b, _ = policy.forward(batch)
    torch.testing.assert_close(loss_a, loss_b)


def test_targets_are_clamped_to_the_tanh_range(make_policy):
    policy = make_policy().eval()
    batch = make_batch()
    loss_a, _ = policy.forward(batch)
    batch["action"] = batch["action"].sign() * 5.0
    loss_b, _ = policy.forward(batch)
    batch["action"] = batch["action"].sign()
    loss_c, _ = policy.forward(batch)
    torch.testing.assert_close(loss_b, loss_c)
    assert not torch.allclose(loss_a, loss_b)


def test_predicted_chunk_has_the_right_shape_and_range(make_policy):
    chunk = make_policy().predict_action_chunk(make_batch(with_action=False))
    assert chunk.shape == (2, CHUNK, ACTION_DIM)
    assert chunk.abs().max() <= 1.0


def test_native_camera_frames_are_resized(make_policy):
    """LeRobot hands over 480 x 640 frames; upstream trains on 256 x 256 (here: the tiny size)."""
    policy = make_policy()
    small = make_batch(with_action=False)
    native = make_batch(with_action=False, size=(48, 80))
    assert policy._images(native).shape == policy._images(small).shape == (2, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert policy.predict_action_chunk(native).shape == (2, CHUNK, ACTION_DIM)


def test_a_task_string_may_be_shared_across_the_batch(make_policy):
    policy = make_policy()
    batch = make_batch(with_action=False)
    batch["task"] = BOWLS[0]
    shared = policy.predict_action_chunk(batch)
    batch["task"] = [BOWLS[0], BOWLS[0]]
    torch.testing.assert_close(shared, policy.predict_action_chunk(batch))


def test_predictions_depend_on_the_instruction(make_policy):
    policy = make_policy()
    batch = make_batch(batch_size=1, with_action=False)
    batch["task"] = [BOWLS[0]]
    a = policy.predict_action_chunk(batch)
    batch["task"] = [BOWLS[1]]
    assert not torch.allclose(a, policy.predict_action_chunk(batch))


def test_camera_order_indexes_the_view_embedding(make_policy):
    batch = make_batch(batch_size=1, with_action=False)
    a = make_policy(seed=0).predict_action_chunk(batch)
    b = make_policy(seed=0, camera_order=list(reversed(CAMS))).predict_action_chunk(batch)
    assert not torch.allclose(a, b)


def test_state_dropout_is_training_only(make_policy):
    policy = make_policy(state_dropout_prob=0.5)
    batch = make_batch(batch_size=256, with_action=False)
    policy.train()
    torch.manual_seed(0)
    dropped = (policy._state(batch) == 0).all(dim=1).float().mean().item()
    assert 0.35 < dropped < 0.65
    policy.eval()
    torch.testing.assert_close(policy._state(batch), batch["observation.state"])
    assert make_policy().train()._state(batch).equal(batch["observation.state"])


# --------------------------------------------------------------------------------------------
# What trains: upstream's split
# --------------------------------------------------------------------------------------------


def test_bert_is_frozen_and_in_eval_mode_dinov3_and_head_train(make_policy):
    policy = make_policy()
    policy.train()
    loss, _ = policy.forward(make_batch())
    loss.backward()
    model = policy.model
    assert all(p.grad is None for p in model.text_encoder.bert.parameters())
    assert not model.text_encoder.bert.training
    assert any(p.grad is not None for p in model.vision_encoder.backbone.parameters())
    assert model.text_encoder.text_projection.weight.grad is not None
    assert model.action_head.decoder.action_queries.weight.grad is not None


def test_optim_params_follow_upstreams_param_groups(make_policy):
    policy = make_policy()
    groups = {g["name"]: g for g in policy.get_optim_params()}
    assert set(groups) == {"dinov3_decay", "dinov3_no_decay", "head_decay", "head_no_decay"}
    assert {g["weight_decay"] for n, g in groups.items() if n.endswith("no_decay")} == {0.0}
    assert groups["head_decay"]["weight_decay"] == 1e-10
    in_groups = {id(p) for g in groups.values() for p in g["params"]}
    trainable = {id(p) for p in policy.model.parameters() if p.requires_grad}
    assert in_groups == trainable


def test_one_training_step_reduces_the_loss_on_a_fixed_batch(make_policy):
    policy = make_policy()
    batch = make_batch()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    policy.eval()
    first, _ = policy.forward(batch)
    policy.train()
    for _ in range(15):
        optimizer.zero_grad()
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
    policy.eval()
    final, _ = policy.forward(batch)
    assert final.item() < first.item()


# --------------------------------------------------------------------------------------------
# Initialising from an upstream checkpoint (the paper's real-robot recipe starts from LIBERO)
# --------------------------------------------------------------------------------------------


@pytest.fixture
def upstream_pth(tmp_path):
    """The parity reference's weights, saved the way upstream's trainer saves a checkpoint (DDP prefix)."""
    ref = torch.load(Path(__file__).parent / "parity" / "upstream_reference.pt", weights_only=False)
    path = tmp_path / "turbovla_step_1.pth"
    torch.save(
        {"model_state_dict": {f"module.{k}": v for k, v in ref["state_dict"].items()}, "model_config": {}},
        path,
    )
    return path, ref


def test_init_from_upstream_checkpoint_loads_everything_when_shapes_match(make_policy, upstream_pth):
    path, ref = upstream_pth
    policy = make_policy(upstream_init_checkpoint=str(path)).eval()
    assert policy.init_report["reinitialised"] == []
    batch = {
        CAMS[0]: ref["images"][:, 0],
        CAMS[1]: ref["images"][:, 1],
        "observation.state": ref["state"],
        "task": list(ref["instructions"]),
    }
    torch.testing.assert_close(policy.predict_action_chunk(batch), ref["actions"], atol=1e-5, rtol=1e-4)


def test_ema_weights_are_preferred_like_upstreams_evaluator(make_policy, upstream_pth, tmp_path):
    path, ref = upstream_pth
    ckpt = torch.load(path, weights_only=False)
    ckpt["ema_model_state_dict"] = {
        k: v + 1.0 if v.is_floating_point() else v for k, v in ref["state_dict"].items()
    }
    torch.save(ckpt, tmp_path / "ema.pth")
    policy = make_policy(upstream_init_checkpoint=str(tmp_path / "ema.pth"))
    torch.testing.assert_close(policy.model.view_embedding, ref["state_dict"]["view_embedding"] + 1.0)


def test_a_50_step_chunk_reinitialises_only_the_action_queries(make_policy, upstream_pth):
    path, ref = upstream_pth
    policy = make_policy(upstream_init_checkpoint=str(path), chunk_size=50, n_action_steps=50)
    assert policy.init_report["reinitialised"] == ["action_head.decoder.action_queries.weight"]
    assert policy.model.action_head.decoder.action_queries.weight.shape[0] == 50
    torch.testing.assert_close(policy.model.view_embedding, ref["state_dict"]["view_embedding"])


def test_a_mismatch_outside_the_robot_dependent_tensors_is_an_error(make_policy, upstream_pth, tmp_path):
    path, ref = upstream_pth
    bad = dict(ref["state_dict"])
    bad["view_embedding"] = torch.zeros(1, 3, bad["view_embedding"].shape[-1])
    torch.save({"model_state_dict": bad}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="does not fit"):
        make_policy(upstream_init_checkpoint=str(tmp_path / "bad.pth"))


def test_a_saved_policy_reloads_without_the_init_checkpoint(make_policy, upstream_pth, tmp_path):
    path, _ = upstream_pth
    policy = make_policy(upstream_init_checkpoint=str(path)).eval()
    batch = make_batch(batch_size=1, with_action=False)
    before = policy.predict_action_chunk(batch)
    policy.save_pretrained(tmp_path / "saved")
    path.unlink()  # the init checkpoint is gone: reloading must not need it
    reloaded = TurboVLAPolicy.from_pretrained(tmp_path / "saved")
    assert reloaded.init_report is None
    torch.testing.assert_close(reloaded.predict_action_chunk(batch), before, atol=1e-6, rtol=1e-5)
    saved = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert saved["type"] == "turbovla_so101"


# --------------------------------------------------------------------------------------------
# Rollout mechanics
# --------------------------------------------------------------------------------------------


def test_select_action_serves_one_action_at_a_time_from_the_queue(make_policy):
    policy = make_policy()
    batch = make_batch(batch_size=1, with_action=False)
    actions = [policy.select_action(batch) for _ in range(CHUNK)]
    assert all(a.shape == (1, ACTION_DIM) for a in actions)
    assert len(policy._action_queue) == 0


def test_reset_clears_the_queue_between_episodes(make_policy):
    policy = make_policy()
    policy.select_action(make_batch(batch_size=1, with_action=False))
    assert len(policy._action_queue) == CHUNK - 1
    policy.reset()
    assert len(policy._action_queue) == 0


def test_n_action_steps_smaller_than_chunk_truncates_the_queue(make_policy):
    policy = make_policy(n_action_steps=2)
    policy.select_action(make_batch(batch_size=1, with_action=False))
    assert len(policy._action_queue) == 1


def test_temporal_ensemble_requires_single_action_step(make_config):
    with pytest.raises(ValueError, match="n_action_steps"):
        make_config(temporal_ensemble_coeff=0.01, n_action_steps=2)


def test_temporal_ensemble_returns_one_action_per_call_and_resets(make_policy):
    policy = make_policy(temporal_ensemble_coeff=0.01, n_action_steps=1)
    batch = make_batch(batch_size=1, with_action=False)
    for _ in range(CHUNK * 2):
        assert policy.select_action(batch).shape == (1, ACTION_DIM)
    assert policy.temporal_ensembler.ensembled_actions is not None
    policy.reset()
    assert policy.temporal_ensembler.ensembled_actions is None


def test_select_action_is_deterministic(make_policy):
    policy = make_policy()
    policy.train()  # select_action must switch to eval itself
    batch = make_batch(batch_size=1, with_action=False)
    first = policy.select_action(batch)
    policy.reset()
    torch.testing.assert_close(first, policy.select_action(batch))
