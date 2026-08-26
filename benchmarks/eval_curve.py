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
"""Held-out action error across a training run's checkpoint series.

Why a curve and not a final number
----------------------------------
A single end-of-training number cannot distinguish the four things that matter, and they call for
opposite conclusions:

* one policy is ahead at every budget -- a claim robust to whatever budget was picked;
* the curves cross -- neither policy wins outright, and the useful result is *where* the crossover
  sits, since that tells a reader which one to choose for their budget;
* a policy is still descending steeply at the end -- the run was too short, and reporting its final
  number as a plateau would misrepresent it;
* a policy has turned back upward -- it is overfitting, and its best checkpoint is not its last.

Choosing the reporting budget *after* seeing the numbers is how a comparison gets discredited. Fix
the budget in advance, log the whole series, and report the shape that comes out.

Every policy is scored on the same held-out frames, at the same horizon, with the same normalizer,
so the only thing varying across a row is the checkpoint. Pass a single `--run` to get one policy's
own learning curve on real SO-101 data (this package's own use case), or several to compare against
baselines the way the upstream port's benchmarks do.

Usage
-----
    python benchmarks/eval_curve.py \
        --dataset abdul004/so101_multi_task_v1 \
        --run turbovla_so101=outputs/cmp_turbovla_so101 \
        --max-frames-per-task 2 --json benchmarks/results/curves.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_per_task import (  # noqa: E402
    evaluate,
    load_checkpoint,
    select_frames,
    split_episodes,
)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import (  # noqa: E402
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

import lerobot_policy_turbovla_so101  # noqa: F401,E402  (registers `turbovla_so101`)


def find_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    """(step, path) for every numbered checkpoint in a run, oldest first.

    `last` is skipped: it is a symlink to one of the numbered directories, and following it would
    plot the same checkpoint twice under two different x values.
    """
    out = []
    ckpt_root = Path(run_dir) / "checkpoints"
    for child in sorted(ckpt_root.glob("*")):
        if child.is_symlink() or not re.fullmatch(r"\d+", child.name):
            continue
        model = child / "pretrained_model"
        if model.is_dir():
            out.append((int(child.name), str(model)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--eval-split", type=float, default=0.2, help="must match training")
    parser.add_argument("--horizon", type=int, default=0, help="0 = smallest chunk across runs")
    parser.add_argument("--max-frames-per-task", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", default="", metavar="PATH", help="also dump the curves as JSON")
    args = parser.parse_args()

    runs = []
    for entry in args.run:
        if "=" not in entry:
            raise SystemExit(f"--run expects NAME=DIR, got {entry!r}")
        name, path = entry.split("=", 1)
        ckpts = find_checkpoints(path)
        if not ckpts:
            raise SystemExit(f"no numbered checkpoints under {path}/checkpoints")
        runs.append((name, ckpts))

    meta = LeRobotDatasetMetadata(args.dataset)
    _, eval_episodes = split_episodes(meta, args.eval_split)

    # The horizon has to be common across every checkpoint of every run, because mean absolute
    # error grows with prediction horizon -- a policy scored over 12 steps is not comparable to one
    # scored over 50 regardless of which is better. With a single --run this is just that run's own
    # native chunk size.
    native = {}
    for name, ckpts in runs:
        cfg = PreTrainedConfig.from_pretrained(ckpts[0][1])
        native[name] = len(cfg.action_delta_indices)
    horizon = args.horizon or min(native.values())
    if any(h < horizon for h in native.values()):
        raise SystemExit(f"--horizon {horizon} exceeds native chunks {native}")

    print(f"dataset  : {args.dataset}")
    print(f"held-out : {len(eval_episodes)} episodes (eval_split={args.eval_split})")
    print(f"horizon  : {horizon} steps  (native: {native})")

    # `delta_timestamps` is derived per-policy (it encodes `observation_delta_indices`, which is
    # `[0]` for a policy like SmolVLA that stacks a short observation history and `None` for one
    # that doesn't, e.g. ACT/TurboVLA). A dataset built once from the first run's config and then
    # reused would hand every other run observations shaped for a different policy -- so each run
    # gets its own `LeRobotDataset`, keyed by its own checkpoint's config. This is cheap since the
    # underlying frames/metadata are cached by lerobot regardless of how many times we instantiate.
    def _dataset_for(ckpt_path: str) -> LeRobotDataset:
        cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        cfg.chunk_size = horizon
        cfg.n_action_steps = min(cfg.n_action_steps, horizon)
        return LeRobotDataset(
            args.dataset,
            episodes=eval_episodes,
            delta_timestamps=resolve_delta_timestamps(cfg, meta),
            return_uint8=True,
        )

    anchor_dataset = _dataset_for(runs[0][1][0][1])
    frames_by_task = select_frames(anchor_dataset, args.max_frames_per_task)
    n_frames = sum(len(v) for v in frames_by_task.values())
    print(f"frames   : {n_frames} over {len(frames_by_task)} tasks\n")

    curves: dict[str, list[tuple[int, float]]] = {}
    per_task_scores: dict[tuple[str, int], dict[str, float]] = {}
    for name, ckpts in runs:
        curves[name] = []
        dataset = _dataset_for(ckpts[0][1])
        # `frames_by_task` holds indices picked against `anchor_dataset`; they're only valid here if
        # this run's dataset enumerates the same frame count in the same order, which holds as long
        # as episode selection (fixed above) is the only thing determining frame order.
        if dataset.num_frames != anchor_dataset.num_frames:
            raise SystemExit(
                f"{name}: dataset has {dataset.num_frames} frames, anchor run has "
                f"{anchor_dataset.num_frames}; frame indices would not line up between the two."
            )
        for step, path in ckpts:
            policy, pre, post, _ = load_checkpoint(path, args.device)
            per_task = evaluate(policy, pre, post, dataset, frames_by_task, args, horizon)
            mae = sum(per_task.values()) / len(per_task)
            curves[name].append((step, mae))
            # Keep the per-task breakdown, not just its mean. `eval_split` holds out
            # ceil(n_episodes * split) episodes PER TASK, and `ceil` is brutal on a dataset whose
            # tasks have one or two episodes: a 1-episode task sends its only episode to eval and
            # contributes nothing to training, so its "held-out" score is really a zero-shot score.
            # Recording each task separately means that distinction can be drawn afterwards from
            # this file, instead of costing another full pass over every checkpoint.
            per_task_scores[(name, step)] = per_task
            print(f"  {name:<12} step {step:>6}  MAE {mae:.4f}", flush=True)
            del policy
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    names = [n for n, _ in runs]
    steps = sorted({s for c in curves.values() for s, _ in c})
    print(f"\n{'step':>8}" + "".join(f"{n:>12}" for n in names))
    print("-" * (8 + 12 * len(names)))
    for s in steps:
        row = f"{s:>8}"
        for n in names:
            hit = dict(curves[n]).get(s)
            row += f"{hit:>12.4f}" if hit is not None else f"{'-':>12}"
        print(row)

    print("\nbest checkpoint per run (lowest held-out MAE):")
    for n in names:
        step, mae = min(curves[n], key=lambda t: t[1])
        final_step, final_mae = curves[n][-1]
        note = (
            "" if step == final_step else f"  <-- NOT the last checkpoint ({final_mae:.4f} at {final_step})"
        )
        print(f"  {n:<12} {mae:.4f} at step {step}{note}")

    print(
        "\nMAE is an offline proxy: it rewards imitating the demonstration, which is not the same\n"
        "as task success. Read it together with the latency table and a real rollout, not alone."
    )

    # Scoring a checkpoint series is expensive -- every point above is a full forward pass over the
    # held-out frames. Dumping the numbers means a plot can be redrawn, or a reviewer's question
    # answered, without paying for the evaluation a second time. The protocol is recorded next to
    # the results so a stray file cannot be mistaken for one produced under different settings.
    if args.json:
        payload = {
            "dataset": args.dataset,
            "eval_split": args.eval_split,
            "eval_episodes": len(eval_episodes),
            "horizon": horizon,
            "native_horizons": native,
            "frames": n_frames,
            "tasks": len(frames_by_task),
            "max_frames_per_task": args.max_frames_per_task,
            "curves": {n: [{"step": s, "mae": m} for s, m in curves[n]] for n in names},
            "per_task": {f"{n}@{st}": scores for (n, st), scores in per_task_scores.items()},
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
