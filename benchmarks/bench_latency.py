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
"""Inference-latency benchmark: TurboVLA against other LeRobot policies.

Latency depends on the architecture and the weights, not on how well a policy was trained, so this
runs without training anything. It measures speed only — it says nothing about task success.

The protocol, and why each part of it is there:

- **Identical inputs.** Every policy sees the same camera count, resolution, state dim, action dim
  and chunk size. A policy timed at a different resolution is not being compared.
- **Batch size 1**, because that is the rollout case. Throughput at batch 64 is a different question.
- **Warmup before timing.** The first calls pay for kernel autotuning and lazy allocation.
- **`torch.cuda.synchronize()` around every call.** GPU work is asynchronous; timing without a sync
  measures how fast Python queues kernels, which is not a number anyone wants.
- **Median, not mean**, so one scheduler hiccup does not move the headline. p10/p90 show the spread.
- **Two latencies, both reported.** `chunk_ms` is one forward pass. `step_ms` is what the robot
  experiences, since action chunking amortizes that pass over `n_action_steps` — which is the whole
  point of predicting a chunk. Quoting only one of the two tells half the story, and quoting only
  `step_ms` is the flattering half.

A note on resolution when comparing against ACT: TurboVLA resizes to a square `image_size` because a
ViT needs a fixed patch grid, while ACT's ResNet consumes whatever it is given. Benchmarking both at
224px is therefore ACT's best case and the conservative choice for TurboVLA. Use `--resolution` to
check the native-resolution case too, and report which one you quoted.

Usage:

    python benchmarks/bench_latency.py                       # TurboVLA vs ACT @ 224px
    python benchmarks/bench_latency.py --resolution 480      # native-ish resolution
    python benchmarks/bench_latency.py --policies turbovla_so101 --device cpu --iters 20
"""

import argparse
import statistics
import time
import traceback

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors

import lerobot_policy_turbovla_so101  # noqa: F401  (registers `turbovla_so101`)


def synthetic_dataset_stats(features: dict) -> dict:
    """Neutral normalization statistics, so no policy is advantaged by its normalizer.

    Latency does not depend on the values, only on the shapes, and every policy declares a
    different `normalization_mapping` (mean/std, min/max, quantiles). Providing every statistic a
    normalizer might ask for keeps this from turning into a per-policy special case.
    """
    stats = {}
    for key, feature in features.items():
        shape = (feature.shape[0], 1, 1) if feature.type is FeatureType.VISUAL else feature.shape
        zeros, ones = torch.zeros(shape), torch.ones(shape)
        stats[key] = {
            "mean": zeros,
            "std": ones,
            "min": -ones,
            "max": ones,
            "q01": -ones,
            "q99": ones,
            "q10": -ones,
            "q90": ones,
        }
    return stats


def build_features(n_cameras: int, height: int, width: int, state_dim: int, action_dim: int):
    input_features = {
        f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        for i in range(n_cameras)
    }
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return input_features, output_features


def build_config(name: str, args):
    """Build a policy config with I/O matched to every other policy in the run."""
    input_features, output_features = build_features(
        args.cameras, args.resolution, args.resolution, args.state_dim, args.action_dim
    )
    common = {
        "input_features": input_features,
        "output_features": output_features,
        "device": args.device,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
    }

    if name == "turbovla_so101":
        from lerobot_policy_turbovla_so101.configuration_turbovla_so101 import TurboVLAConfig

        return TurboVLAConfig(
            vision_backbone=args.vision_backbone,
            language_backbone=args.language_backbone,
            image_size=args.image_size or args.resolution,
            **common,
        )

    if name == "act":
        from lerobot.policies.act.configuration_act import ACTConfig

        return ACTConfig(**common)

    if name == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        # Architecture determines latency, so the VLM weights are not downloaded.
        return SmolVLAConfig(load_vlm_weights=False, **common)

    if name == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        return PI05Config(**common)

    raise SystemExit(
        f"Unknown policy '{name}'. Known: turbovla_so101, act, smolvla, pi05. Add a branch to compare "
        "another one."
    )


def build_policy(config):
    """Instantiate the policy class registered for this config, by LeRobot's own convention."""
    from lerobot.policies.factory import get_policy_class

    return get_policy_class(config.type)(config)


def build_observation(args, device):
    obs = {
        f"observation.images.cam{i}": torch.rand(
            1, 3, args.resolution, args.resolution, device=device
        )
        for i in range(args.cameras)
    }
    obs["observation.state"] = torch.randn(1, args.state_dim, device=device)
    obs["task"] = ["pick up the pink whistle and put it in the green basket"]
    return obs


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_calls(fn, iters: int, warmup: int, device: str) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)

    samples = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def benchmark(name: str, args) -> dict:
    device = args.device
    config = build_config(name, args)
    policy = build_policy(config).to(device)
    policy.eval()

    # Each policy's own preprocessor turns a raw observation into its model-ready batch. For the
    # VLAs that is also where the instruction gets tokenized, so it cannot be skipped. Running it
    # once outside the timed loop means we time the forward pass and not the tokenizer.
    features = {**config.input_features, **config.output_features}
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=synthetic_dataset_stats(features))
    batch = preprocessor(build_observation(args, device))

    n_total = sum(p.numel() for p in policy.parameters())

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        samples = time_calls(lambda: policy.predict_action_chunk(batch), args.iters, args.warmup, device)

    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.startswith("cuda") else float("nan")
    )

    samples.sort()
    median = statistics.median(samples)
    result = {
        "policy": name,
        "chunk_ms": median,
        "p10": samples[int(0.10 * len(samples))],
        "p90": samples[int(0.90 * len(samples))],
        "step_ms": median / args.n_action_steps,
        "hz": 1000.0 / (median / args.n_action_steps),
        "peak_mb": peak_mb,
        "params_m": n_total / 1e6,
    }

    del policy
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", default=["turbovla_so101", "act", "smolvla", "pi05"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=224, help="input frame size fed to every policy")
    parser.add_argument(
        "--image-size", type=int, default=None, help="TurboVLA internal resize (defaults to --resolution)"
    )
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--vision-backbone", default="facebook/dinov2-base")
    parser.add_argument("--language-backbone", default="google-bert/bert-base-uncased")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False.")

    device_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
    print(f"device   : {device_name}")
    print(f"torch    : {torch.__version__}")
    print(
        f"protocol : batch 1, {args.cameras} cams @ {args.resolution}px, chunk {args.chunk_size}, "
        f"{args.iters} timed iters after {args.warmup} warmup"
    )
    print()

    results = []
    for name in args.policies:
        try:
            results.append(benchmark(name, args))
        except Exception:
            # One unavailable baseline should not throw away the rest of the run.
            print(f"SKIPPED {name}:")
            traceback.print_exc()
            print()

    if not results:
        raise SystemExit("No policy could be benchmarked.")

    header = (
        f"{'policy':<10} {'chunk ms':>9} {'p10-p90':>16} {'step ms':>8} {'Hz':>7} "
        f"{'VRAM MB':>9} {'params M':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['policy']:<10} {r['chunk_ms']:>9.2f} {r['p10']:>7.2f}-{r['p90']:<8.2f} "
            f"{r['step_ms']:>8.2f} {r['hz']:>7.1f} {r['peak_mb']:>9.1f} {r['params_m']:>9.1f}"
        )

    print()
    print("chunk ms = one forward pass; step ms = amortized over n_action_steps (what the robot sees).")
    if len(results) > 1:
        base = results[0]
        for other in results[1:]:
            ratio = other["chunk_ms"] / base["chunk_ms"]
            verb = "faster than" if ratio > 1 else "slower than"
            print(f"{base['policy']} is {max(ratio, 1 / ratio):.2f}x {verb} {other['policy']} per chunk.")


if __name__ == "__main__":
    main()
