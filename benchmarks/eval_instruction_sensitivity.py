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
"""Counterfactual instruction test: does the policy actually *use* the sentence?

Aggregate held-out error is a blunt instrument for this question. Two tasks that share a scene and
differ only in the instruction also share their approach and grasp phases, so most frames carry no
information about which instruction was given. Averaging over all of them dilutes the very effect
being measured, and a real language effect can vanish into the noise.

This measures the mechanism directly instead. For a frame recorded under instruction A, the policy
is run twice on the *same pixels* — once with A, once with its paired instruction B:

    err_factual        = |predict(obs, A) - recorded_actions|
    err_counterfactual = |predict(obs, B) - recorded_actions|
    divergence         = |predict(obs, A) - predict(obs, B)|

Three outcomes, all informative:

* `divergence ~ 0` — the policy ignores language entirely. For a policy with no language input at
  all (ACT) this is true by construction, and the run is a sanity check on the harness.
* `divergence > 0` but `err_counterfactual ~ err_factual` — the sentence perturbs the output without
  steering it anywhere useful.
* `err_counterfactual > err_factual` — swapping the instruction makes the prediction measurably
  worse against the recording. That is the signature of language actually being followed, and the
  gap size is how much it is worth.

A null result here is not automatically a fault in the policy: if the two destinations are
distinguishable in the frame (only one basket in shot, say), the task never required the sentence
and there is nothing for the policy to use. Read the outcome together with the dataset.

Usage:

    python benchmarks/eval_instruction_sensitivity.py \
        --dataset abdul004/so101_multi_task_v1 --eval-split 0.2 --resize 224 \
        --checkpoint turbovla_so101=outputs/cmp2_turbovla_so101/checkpoints/last/pretrained_model \
        --checkpoint act=outputs/cmp2_act/checkpoints/last/pretrained_model \
        --pair "Pick up the bangle and put it in the blue basket" \
               "Pick up the bangle and put it in the green basket"
"""

import argparse
import collections

import torch
import torch.nn.functional as F  # noqa: N812
from eval_per_task import group_frames_by_task, load_checkpoint, split_episodes
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import lerobot_policy_turbovla_so101  # noqa: F401  (registers `turbovla_so101`)


def subsample(indices: list[int], limit: int) -> list[int]:
    """Evenly spaced subset, so the sampled frames span the whole episode rather than its opening."""
    if limit > 0 and len(indices) > limit:
        step = len(indices) / limit
        return [indices[int(k * step)] for k in range(limit)]
    return indices


def build_observation(samples, resize: int) -> dict:
    obs = {}
    for key in samples[0]:
        if not key.startswith("observation."):
            continue
        stacked = torch.stack([s[key] for s in samples])
        # `_is_pad` companions share the image prefix but are 1-D boolean masks, not frames. They
        # appear whenever a policy sets `observation_delta_indices` (SmolVLA does, ACT does not).
        if key.startswith("observation.images.") and not key.endswith("_is_pad") and resize:
            stacked = stacked.float()
            if stacked.max() > 1.5:
                stacked = stacked / 255.0
            stacked = F.interpolate(stacked, size=(resize, resize), mode="bilinear", align_corners=False)
        obs[key] = stacked
    return obs


@torch.no_grad()
def run_pair(policy, pre, post, dataset, frames_by_task, task_a, task_b, args):
    """Errors under the true and the swapped instruction, for frames recorded under `task_a`."""
    indices = subsample(frames_by_task.get(task_a, []), args.max_frames)
    if not indices:
        return None

    sums = collections.Counter()
    count = 0

    for start in range(0, len(indices), args.batch_size):
        block = indices[start : start + args.batch_size]
        samples = [dataset[i] for i in block]
        obs = build_observation(samples, args.resize)

        target = torch.stack([s["action"] for s in samples]).cpu().float()
        if target.ndim == 2:
            target = target.unsqueeze(1)

        preds = {}
        for label, task in (("factual", task_a), ("counterfactual", task_b)):
            batch = pre({**obs, "task": [task] * len(block)})
            preds[label] = post(policy.predict_action_chunk(batch)).cpu().float()

        horizon = min(preds["factual"].shape[1], target.shape[1])
        target = target[:, :horizon]
        mask = torch.ones_like(target, dtype=torch.bool)
        if "action_is_pad" in samples[0]:
            pad = torch.stack([s["action_is_pad"] for s in samples]).cpu()[:, :horizon]
            mask = ~pad.unsqueeze(-1).expand_as(target)

        n = int(mask.sum().item())
        for label in preds:
            err = ((preds[label][:, :horizon] - target).abs() * mask).sum().item()
            sums[label] += err
        sums["divergence"] += (
            ((preds["factual"][:, :horizon] - preds["counterfactual"][:, :horizon]).abs() * mask).sum().item()
        )
        count += n

    return {k: v / max(count, 1) for k, v in sums.items()} | {"frames": len(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        required=True,
        metavar=("TASK_A", "TASK_B"),
        help="repeatable; two task strings that share a scene and differ only in the instruction",
    )
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoints = [entry.split("=", 1) for entry in args.checkpoint]

    meta = LeRobotDatasetMetadata(args.dataset)
    _, eval_episodes = split_episodes(meta, args.eval_split)
    first_config = PreTrainedConfig.from_pretrained(checkpoints[0][1])
    dataset = LeRobotDataset(
        args.dataset,
        episodes=eval_episodes,
        delta_timestamps=resolve_delta_timestamps(first_config, meta),
        return_uint8=True,
    )

    frames_by_task = group_frames_by_task(dataset)

    print(f"dataset  : {args.dataset}  ({len(eval_episodes)} held-out episodes)")
    print("Errors in the dataset's action units. 'swap cost' = counterfactual - factual;")
    print("positive means following the given instruction beats following the paired one.\n")

    for name, path in checkpoints:
        policy, pre, post, _ = load_checkpoint(path, args.device)
        print(f"### {name}")
        header = f"{'frames recorded under':<52}{'factual':>9}{'counterf':>10}{'swap cost':>11}{'diverg':>9}"
        print(header)
        print("-" * len(header))

        totals = collections.Counter()
        n_dirs = 0
        for task_a, task_b in [d for pair in args.pair for d in (pair, pair[::-1])]:
            res = run_pair(policy, pre, post, dataset, frames_by_task, task_a, task_b, args)
            if res is None:
                print(f"{task_a[:50]:<52}{'(no held-out frames)':>39}")
                continue
            swap = res["counterfactual"] - res["factual"]
            print(
                f"{task_a[:50]:<52}{res['factual']:>9.3f}{res['counterfactual']:>10.3f}"
                f"{swap:>11.3f}{res['divergence']:>9.3f}"
            )
            totals["factual"] += res["factual"]
            totals["counterfactual"] += res["counterfactual"]
            totals["divergence"] += res["divergence"]
            n_dirs += 1

        if n_dirs:
            f, c = totals["factual"] / n_dirs, totals["counterfactual"] / n_dirs
            print("-" * len(header))
            print(f"{'MEAN':<52}{f:>9.3f}{c:>10.3f}{c - f:>11.3f}{totals['divergence'] / n_dirs:>9.3f}")
        print()

        del policy
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
