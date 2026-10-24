"""Build the parity reference with upstream's own code, in upstream's own environment.

    python tests/parity/make_upstream_reference.py --upstream <clone of H-EmbodVis/TurboVLA @ a3365e8> \\
        --out tests/parity/upstream_reference.pt

Run it in an environment with upstream's reference versions (transformers 4.56.2, timm). It builds a
tiny TurboVLA with upstream's `build_turbovla` (random init, seeded), runs it in eval mode on fixed
inputs, and saves the full state dict (backbones included, under transformers 4.56's parameter names,
as the released LIBERO checkpoint is), the inputs and the outputs. `tests/test_upstream_parity.py`
loads that state dict into the plugin and must reproduce the outputs.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tiny_backbones import IMAGE_SIZE, make_backbone_dirs  # noqa: E402

# Shared with tests/test_upstream_parity.py: the tiny architecture both sides build.
TINY = json.loads((HERE / "tiny_config.json").read_text())
INSTRUCTIONS = [
    "Put the blue bowl in the pink bowl.",
    "Put the pink bowl in the blue bowl.",
    "pick up the red cube. put it on top of the blue bowl?",
]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "upstream_reference.pt")
    args = ap.parse_args()
    sys.path.insert(0, str(args.upstream))
    import transformers
    from transformers import AutoConfig, AutoModel
    from turbovla.models.configuration import (
        ActionHeadConfig,
        InteractionConfig,
        TextEncoderConfig,
        TurboVLAConfig,
        VisionEncoderConfig,
    )
    from turbovla.models.turbovla import build_turbovla

    commit = subprocess.run(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        vision_dir, text_dir = make_backbone_dirs(tmp)
        torch.manual_seed(0)
        for d in (vision_dir, text_dir):  # upstream loads backbones with from_pretrained: give them weights
            AutoModel.from_config(AutoConfig.from_pretrained(d)).save_pretrained(d)
        t, a, i = TINY["text"], TINY["action"], TINY["interaction"]
        cfg = TurboVLAConfig(
            compatibility_profile="libero",
            text=TextEncoderConfig(model_name_or_path=text_dir, local_files_only=True, **t),
            vision=VisionEncoderConfig(
                model_name_or_path=vision_dir,
                local_files_only=True,
                image_size=IMAGE_SIZE,
                num_views=2,
                frozen=False,
                compute_precision="bf16_autocast",
                dropout=0.1,
            ),
            interaction=InteractionConfig(**i),
            action=ActionHeadConfig(**a),
        )
        torch.manual_seed(1)
        model = build_turbovla(cfg).eval()

    g = torch.Generator().manual_seed(2)
    images = torch.rand(len(INSTRUCTIONS), 2, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)  # [0, 1], as LeRobot
    state = torch.randn(len(INSTRUCTIONS), a["state_dim"], generator=g)  # already normalised
    pixels = (images - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        condition = model.encode_condition(INSTRUCTIONS, {"dinov3": pixels})
        actions = model(INSTRUCTIONS, {"dinov3": pixels}, state)
    torch.save(
        {
            "upstream_commit": commit,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "images": images,
            "state": state,
            "instructions": INSTRUCTIONS,
            "condition": condition,
            "actions": actions,
        },
        args.out,
    )
    print(
        f"wrote {args.out}: upstream {commit[:7]}, transformers {transformers.__version__}, "
        f"{len(model.state_dict())} tensors, actions {tuple(actions.shape)}"
    )


if __name__ == "__main__":
    main()
