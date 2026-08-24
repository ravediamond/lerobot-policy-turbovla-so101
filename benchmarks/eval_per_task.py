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
"""Per-task held-out action error, for comparing a language-conditioned policy against one that is not.

Why per-task and not a single number
------------------------------------
A multi-task dataset usually mixes two kinds of task:

* **Visually determined** — the instruction restates what the camera already shows ("pick up the
  ping pong ball" when a ping pong ball is the only object present). A policy that ignores language
  can infer the task from pixels and do fine.
* **Instruction-dependent** — two tasks share a scene and differ only in the sentence ("...put it in
  the *green* basket" vs "...the *blue* basket"). A policy that ignores language sees identical
  input for both and can do no better than predicting their average.

Aggregate error mixes these together, and the first kind dilutes the second: a real language effect
can hide behind a small overall gap. Splitting them makes the claim falsifiable — the prediction is
that the two policies are close on visually-determined tasks and diverge on instruction-dependent
ones. If they diverge everywhere, something other than language is driving the difference; if they
diverge nowhere, the language conditioning is not earning its keep.

What is measured
----------------
Mean absolute error between the predicted action chunk and the recorded one, in the dataset's own
action units (degrees for an SO-101 arm), over held-out episodes. Padded targets are excluded. This
is an offline proxy: it rewards imitating the demonstration, which is not the same as task success.
Treat it as evidence, not as a success rate.

The held-out split is recomputed with the same rule `lerobot-train` uses (the last
`ceil(n_episodes * eval_split)` episodes *per task*), so with a matching `--eval-split` these frames
are the ones no checkpoint was trained on. Pass the same value you trained with.

Usage
-----
    python benchmarks/eval_per_task.py \
        --dataset abdul004/so101_multi_task_v1 \
        --eval-split 0.2 --resize 224 \
        --checkpoint turbovla_so101=outputs/cmp_turbovla_so101/checkpoints/last/pretrained_model \
        --checkpoint act=outputs/cmp_act/checkpoints/last/pretrained_model
"""

import argparse
import collections
import itertools
import math

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

import lerobot_policy_turbovla_so101  # noqa: F401  (registers `turbovla_so101`)


def split_episodes(meta: LeRobotDatasetMetadata, eval_split: float) -> tuple[list[int], list[int]]:
    """Replicate `lerobot.datasets.factory.make_train_eval_datasets`: last N per task are held out."""
    episode_tasks = meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in range(meta.total_episodes):
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])
    return train_episodes, eval_episodes


def find_minimal_pairs(tasks: list[str]) -> list[tuple[str, str]]:
    """Task pairs whose sentences differ by exactly one token on each side.

    A heuristic for spotting instruction-dependent tasks. It flags any one-word difference, so it
    will also catch pairs that differ by a *visible* attribute (object colour, say) — those are
    still resolvable from pixels. The caller is expected to eyeball the reported list rather than
    trust it blindly.
    """
    pairs = []
    for a, b in itertools.combinations(sorted(tasks), 2):
        ta, tb = a.lower().split(), b.lower().split()
        only_a = collections.Counter(ta) - collections.Counter(tb)
        only_b = collections.Counter(tb) - collections.Counter(ta)
        if sum(only_a.values()) == 1 and sum(only_b.values()) == 1:
            pairs.append((a, b))
    return pairs


def recorded_signatures(dataset, frames_by_task) -> dict:
    """Mean recorded action chunk per task, used as a behavioural fingerprint.

    `find_minimal_pairs` works on strings, so it cannot tell the two kinds of one-word difference
    apart, and they carry opposite expectations:

    * **paraphrase** -- "collect the alphabet soup" vs "grab the alphabet soup". Same underlying
      manipulation, so a good policy should predict nearly the *same* actions. Divergence here is a
      fault (the policy is keying on surface wording).
    * **target swap** -- "put the bbq sauce in the basket" vs "put the tomato sauce in the basket".
      Different object, so a good policy must predict *different* actions. Divergence here is the
      whole point.

    Averaging the two together cancels the signal against its own opposite. The demonstrations
    already record which kind a pair is: two instructions that were carried out with the same
    motion are paraphrases, and two that were not are target swaps. This returns that fingerprint
    so pairs can be split on the data rather than on a hand-written word list.
    """
    sigs = {}
    for task, indices in frames_by_task.items():
        chunks = []
        for i in indices:
            a = dataset[i]["action"].float()
            chunks.append(a if a.ndim == 2 else a.unsqueeze(0))
        if chunks:
            sigs[task] = torch.stack(chunks).mean(0).flatten()
    return sigs


def rank_pairs_by_behaviour(pairs, sigs):
    """Minimal pairs sorted by how differently their demonstrations were actually performed.

    This deliberately does NOT return a paraphrase/target-swap classification. Distances measured
    this way tend to be smooth and unimodal rather than bimodal, so there is no natural boundary to
    cut on, and inventing one would launder a guess as a measurement.

    What the ranking is good for is choosing which pairs to spend evaluation on. Pairs at the top
    were carried out most differently, so they are the best candidates for testing whether the
    policy follows the instruction; pairs at the bottom were carried out most similarly, so they are
    the best candidates for testing that it is not merely reacting to surface wording. Confirm a
    candidate by reading it before trusting it either way.
    """
    scored = []
    for a, b in pairs:
        if a in sigs and b in sigs:
            n = min(sigs[a].numel(), sigs[b].numel())
            scored.append(((sigs[a][:n] - sigs[b][:n]).abs().mean().item(), a, b))
    scored.sort()
    return scored


def load_checkpoint(path: str, device: str):
    config = PreTrainedConfig.from_pretrained(path)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(path, config=config).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=path)
    return policy, preprocessor, postprocessor, config


def frame_episode_indices(dataset: LeRobotDataset) -> list[int]:
    """The `episode_index` of every frame, read as a whole column.

    Indexing `hf_dataset` row by row decodes each row (video frames included), which turns a simple
    lookup into minutes of work on a large dataset. Pulling the column once stays fast as datasets
    grow.
    """
    return [int(x) for x in dataset.hf_dataset["episode_index"]]


def group_frames_by_task(dataset: LeRobotDataset) -> dict[str, list[int]]:
    by_task: dict[str, list[int]] = collections.defaultdict(list)
    episode_tasks = dataset.meta.episodes["tasks"]
    for frame_index, episode_index in enumerate(frame_episode_indices(dataset)):
        tasks = episode_tasks[episode_index]
        by_task[tasks[0] if tasks else ""].append(frame_index)
    return by_task


def select_frames(dataset: LeRobotDataset, max_per_task: int) -> dict[str, list[int]]:
    """Evenly spaced frame indices per task, so no episode or task dominates the average."""
    by_task = group_frames_by_task(dataset)

    chosen = {}
    for task, indices in by_task.items():
        if max_per_task > 0 and len(indices) > max_per_task:
            step = len(indices) / max_per_task
            indices = [indices[int(k * step)] for k in range(max_per_task)]
        chosen[task] = indices
    return chosen


@torch.no_grad()
def evaluate(policy, preprocessor, postprocessor, dataset, frames_by_task, args, horizon=None):
    """Mean absolute action-chunk error per task, in the dataset's action units.

    `horizon` caps how many steps of the chunk are scored. Left as None, each batch is scored over
    as much of the chunk as both the prediction and the target provide, which is what a single
    policy wants. Comparing policies needs it pinned instead: mean absolute error grows with
    prediction horizon, so a policy scored over 12 steps and one scored over 50 are not on the same
    scale no matter which is actually better.
    """
    results = {}

    for task, indices in sorted(frames_by_task.items()):
        total_err, total_count = 0.0, 0

        for start in range(0, len(indices), args.batch_size):
            block = indices[start : start + args.batch_size]
            samples = [dataset[i] for i in block]

            obs = {}
            for key in samples[0]:
                if not key.startswith("observation."):
                    continue
                stacked = torch.stack([s[key] for s in samples])
                # `_is_pad` companions are boolean masks, not frames. They only appear for policies
                # whose `observation_delta_indices` is set (SmolVLA's is `[0]`, ACT's is None), and
                # they share the `observation.images.` prefix, so a prefix test alone routes a
                # 1-D mask into the image resize and fails there.
                if key.startswith("observation.images.") and not key.endswith("_is_pad") and args.resize:
                    stacked = stacked.float()
                    if stacked.max() > 1.5:  # uint8 frames arrive in [0, 255]
                        stacked = stacked / 255.0
                    stacked = F.interpolate(
                        stacked, size=(args.resize, args.resize), mode="bilinear", align_corners=False
                    )
                obs[key] = stacked
            obs["task"] = [task] * len(block)

            predicted = policy.predict_action_chunk(preprocessor(obs))
            predicted = postprocessor(predicted).cpu().float()

            target = torch.stack([s["action"] for s in samples]).cpu().float()
            if target.ndim == 2:  # no chunking configured on the dataset
                target = target.unsqueeze(1)
            h = min(predicted.shape[1], target.shape[1])
            if horizon:
                h = min(h, horizon)
            predicted, target = predicted[:, :h], target[:, :h]

            mask = torch.ones_like(target, dtype=torch.bool)
            if "action_is_pad" in samples[0]:
                pad = torch.stack([s["action_is_pad"] for s in samples]).cpu()[:, :h]
                mask = ~pad.unsqueeze(-1).expand_as(target)

            total_err += ((predicted - target).abs() * mask).sum().item()
            total_count += int(mask.sum().item())

        results[task] = total_err / max(total_count, 1)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeatable, e.g. --checkpoint turbovla_so101=outputs/x/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--eval-split", type=float, default=0.2, help="must match training")
    parser.add_argument(
        "--resize", type=int, default=0, help="resize frames to NxN before the policy (0 = leave alone)"
    )
    parser.add_argument("--max-frames-per-task", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoints = []
    for entry in args.checkpoint:
        if "=" not in entry:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got {entry!r}")
        name, path = entry.split("=", 1)
        checkpoints.append((name, path))

    meta = LeRobotDatasetMetadata(args.dataset)
    _, eval_episodes = split_episodes(meta, args.eval_split)
    print(f"dataset      : {args.dataset}")
    print(f"held-out     : {len(eval_episodes)} episodes (eval_split={args.eval_split})")
    print(f"resize       : {args.resize or 'none'}")

    # Chunk targets follow the first checkpoint's horizon; the comparison truncates to the shared
    # horizon anyway, and matched runs use the same chunk_size.
    first_config = PreTrainedConfig.from_pretrained(checkpoints[0][1])
    dataset = LeRobotDataset(
        args.dataset,
        episodes=eval_episodes,
        delta_timestamps=resolve_delta_timestamps(first_config, meta),
        return_uint8=True,
    )
    frames_by_task = select_frames(dataset, args.max_frames_per_task)
    print(f"frames       : {sum(len(v) for v in frames_by_task.values())} over {len(frames_by_task)} tasks")
    print()

    scores = {}
    for name, path in checkpoints:
        policy, pre, post, _ = load_checkpoint(path, args.device)
        scores[name] = evaluate(policy, pre, post, dataset, frames_by_task, args)
        del policy
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    names = [n for n, _ in checkpoints]
    tasks = sorted(frames_by_task)
    pairs = find_minimal_pairs(tasks)
    paired_tasks = {t for pair in pairs for t in pair}

    width = max(len(t) for t in tasks) + 2
    header = f"{'task':<{width}}" + "".join(f"{n:>12}" for n in names)
    if len(names) == 2:
        header += f"{'delta':>10}"
    print(header)
    print("-" * len(header))
    for task in tasks:
        marker = "*" if task in paired_tasks else " "
        row = f"{marker}{task:<{width - 1}}" + "".join(f"{scores[n][task]:>12.3f}" for n in names)
        if len(names) == 2:
            row += f"{scores[names[0]][task] - scores[names[1]][task]:>10.3f}"
        print(row)

    def group_mean(name, group):
        vals = [scores[name][t] for t in group]
        return sum(vals) / len(vals) if vals else float("nan")

    unpaired = [t for t in tasks if t not in paired_tasks]
    print()
    print(f"{'group':<{width}}" + "".join(f"{n:>12}" for n in names))
    print("-" * len(header))
    print(f"{'ALL tasks':<{width}}" + "".join(f"{group_mean(n, tasks):>12.3f}" for n in names))
    print(
        f"{'* one-word-different':<{width}}"
        + "".join(f"{group_mean(n, sorted(paired_tasks)):>12.3f}" for n in names)
    )
    print(f"{'  distinct tasks':<{width}}" + "".join(f"{group_mean(n, unpaired):>12.3f}" for n in names))

    print()
    print("Units are the dataset's action units (degrees for SO-101). Lower is better.")
    if pairs:
        sigs = recorded_signatures(dataset, frames_by_task)
        scored = rank_pairs_by_behaviour(pairs, sigs)
        dists = [d for d, _, _ in scored]

        print(f"\nDetected {len(pairs)} one-word-different task pairs (marked * above).")
        if dists:
            srt = sorted(dists)
            deciles = " ".join(f"{srt[int(q * (len(srt) - 1) / 10)]:.3f}" for q in range(11))
            print(f"Recorded-action distance across those pairs, deciles: {deciles}")
            print(
                "A one-word difference means two things that grade in OPPOSITE directions, and the\n"
                "sentence alone does not say which:\n"
                "  paraphrase  -- 'collect the soup' vs 'grab the soup'. Same intent, so predictions\n"
                "                 SHOULD match; divergence means the policy keys on surface wording.\n"
                "  target swap -- 'the bbq sauce' vs 'the tomato sauce'. Different object, so\n"
                "                 predictions SHOULD differ; divergence is language being used, and a\n"
                "                 policy with no language input scores zero here by construction.\n"
                "Averaging the two together cancels the signal against its own opposite, so they have\n"
                "to be evaluated separately.\n"
                "\nThe distances above rank the pairs by how differently the demonstrations were\n"
                "actually performed, which is the best automatic guide available -- but they are\n"
                "NOT a classification. If they turn out smooth and unimodal, with no boundary to cut\n"
                "on, the two ends are candidates to confirm by reading, not labels.\n"
                "Feed the confirmed pairs to eval_instruction_sensitivity.py, which measures the\n"
                "divergence itself; this script only finds candidates."
            )
            print("\n  most differently performed -- candidate TARGET SWAPS (expect divergence):")
            for d, a, b in scored[:-9:-1]:
                print(f"    {d:.3f}  {a!r}\n           {b!r}")
            print("\n  most similarly performed -- candidate PARAPHRASES (expect agreement):")
            for d, a, b in scored[:8]:
                print(f"    {d:.3f}  {a!r}\n           {b!r}")


if __name__ == "__main__":
    main()
