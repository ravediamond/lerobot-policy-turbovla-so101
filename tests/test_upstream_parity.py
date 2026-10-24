"""The plugin must compute exactly what upstream TurboVLA computes.

`parity/upstream_reference.pt` was produced by `parity/make_upstream_reference.py` with upstream's own
code (H-EmbodVis/TurboVLA @ a3365e8) on transformers 4.56.2: a tiny TurboVLA, its full state dict
(backbones included, under 4.56's parameter names), fixed inputs and its eval-mode outputs. Here the
plugin builds the same architecture on the installed transformers, loads that state dict with the same
loader used for the released LIBERO checkpoint, and must reproduce the outputs.
"""

import json
from pathlib import Path

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from tiny_backbones import IMAGE_SIZE, make_backbone_dirs

from lerobot_policy_turbovla_so101.configuration_turbovla_so101 import TurboVLAConfig
from lerobot_policy_turbovla_so101.modeling_turbovla_so101 import (
    TurboVLAPolicy,
    align_backbone_names,
    load_upstream_weights,
)

PARITY = Path(__file__).resolve().parent / "parity"
REFERENCE = PARITY / "upstream_reference.pt"
TINY = json.loads((PARITY / "tiny_config.json").read_text())


def tiny_plugin_config(vision_dir: str, text_dir: str, **overrides) -> TurboVLAConfig:
    t, i, a = TINY["text"], TINY["interaction"], TINY["action"]
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    kwargs = dict(
        input_features={
            "observation.images.front": image,
            "observation.images.wrist": image,
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(a["state_dim"],)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(a["action_dim"],))},
        device="cpu",
        camera_order=["observation.images.front", "observation.images.wrist"],
        vision_backbone=vision_dir,
        language_backbone=text_dir,
        load_pretrained_backbones=False,
        freeze_vision_backbone=False,
        freeze_language_backbone=t["frozen"],
        image_size=IMAGE_SIZE,
        max_language_tokens=t["max_length"],
        text_padding_length=t["padding_length"],
        sub_sentence_present=t["sub_sentence_present"],
        zero_padded_text=t["zero_padded_tokens"],
        dim_model=i["hidden_dim"],
        n_heads=i["nheads"],
        n_interaction_layers=i["num_layers"],
        dim_feedforward=i["dim_feedforward"],
        enhancer_inner_dim=i["enhancer_inner_dim"],
        text_dropout=i["text_dropout"],
        fusion_dropout=i["fusion_dropout"],
        fusion_droppath=i["fusion_droppath"],
        chunk_size=a["horizon"],
        n_action_steps=a["horizon"],
        num_state_tokens=a["num_state_tokens"],
        n_decoder_layers=a["num_layers"],
        action_mlp_hidden_dim=a["mlp_hidden_dim"],
        state_hidden_dim=a["state_hidden_dim"],
        action_dropout=a["dropout"],
    )
    kwargs.update(overrides)
    return TurboVLAConfig(**kwargs)


@pytest.fixture(scope="module")
def reference():
    return torch.load(REFERENCE, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def policy_with_upstream_weights(reference, tmp_path_factory):
    vision_dir, text_dir = make_backbone_dirs(tmp_path_factory.mktemp("backbones"))
    policy = TurboVLAPolicy(tiny_plugin_config(vision_dir, text_dir))
    report = load_upstream_weights(policy.model, reference["state_dict"])
    return policy.eval(), report


def batch_from(reference) -> dict:
    return {
        "observation.images.front": reference["images"][:, 0],
        "observation.images.wrist": reference["images"][:, 1],
        "observation.state": reference["state"],
        "task": list(reference["instructions"]),
    }


def test_reference_was_made_with_the_pinned_upstream_commit(reference):
    assert reference["upstream_commit"].startswith("a3365e8")
    assert reference["transformers"].startswith("4.56")


def test_every_upstream_tensor_loads_with_the_same_name_and_shape(policy_with_upstream_weights, reference):
    policy, report = policy_with_upstream_weights
    assert report["reinitialised"] == []
    assert report["loaded"] == len(reference["state_dict"]) == len(policy.model.state_dict())
    # transformers 5 nests DINOv3's blocks under `.model`: the only renaming the loader may do.
    renamed, n = align_backbone_names(reference["state_dict"], policy.model.state_dict())
    assert set(renamed) == set(policy.model.state_dict())
    assert n == report["renamed_backbone_keys"]
    assert all(".backbone.layer." in k for k in set(reference["state_dict"]) - set(policy.model.state_dict()))


def test_fused_condition_matches_upstream(policy_with_upstream_weights, reference):
    policy, _ = policy_with_upstream_weights
    batch = batch_from(reference)
    with torch.no_grad():
        pixels = policy._images(batch)
        condition = policy.model.encode_condition(batch["task"], {"dinov3": pixels})
    torch.testing.assert_close(condition, reference["condition"], atol=1e-5, rtol=1e-4)


def test_action_chunk_matches_upstream(policy_with_upstream_weights, reference):
    policy, _ = policy_with_upstream_weights
    actions = policy.predict_action_chunk(batch_from(reference))
    torch.testing.assert_close(actions, reference["actions"], atol=1e-5, rtol=1e-4)


def test_the_two_bowl_instructions_give_different_plans_in_the_reference(reference):
    """Sanity check on the fixture itself: a test that cannot fail on swapped instructions is useless."""
    a = reference["actions"]
    assert (a[0] - a[1]).abs().max() > 1e-4
