# Inference latency: TurboVLA vs ACT vs SmolVLA on Jetson Orin NX

Measured, not estimated. Reproduce with `benchmarks/bench_latency.py`.

## Environment

| | |
|---|---|
| device | Jetson Orin NX, 16GB unified memory |
| torch | 2.11.0, CUDA 12.6, built from source for `sm_87` |
| date | 2026-08-26 |

## Protocol

Batch 1 (the rollout case), 2 cameras @ 224px, chunk size 12. 20 warmup iterations discarded, then
100 timed. Median reported, p10-p90 shows spread. Same protocol as `bench_latency.py`'s own
docstring — see there for why each choice (warmup, `torch.cuda.synchronize()`, median-not-mean)
matters.

```
python benchmarks/bench_latency.py --policies turbovla_so101 act smolvla --device cuda
```

## Results

| policy | chunk ms | p10-p90 | step ms | Hz | VRAM MB | params M |
|---|---|---|---|---|---|---|
| turbovla_so101 | 152.34 | 152.09-152.52 | 12.70 | 78.8 | 827.7 | 210.2 |
| act | 27.67 | 27.51-27.82 | 2.31 | 433.6 | 219.2 | 51.6 |
| smolvla | 1062.83 | 1061.77-1063.92 | 88.57 | 11.3 | 1302.3 | 450.0 |

TurboVLA is 5.5x slower than ACT and 7.0x faster than SmolVLA, per chunk. That ranking matches
discrete-GPU numbers reported elsewhere for the same three architectures — the gap holds on
unified-memory ARM hardware, not just a desktop card.

## Caveats

- **Latency only.** Says nothing about task success — see the top-level `Results` section for that
  once the SO-101 training run lands.
- **Unauthenticated HF Hub pulls.** No `HF_TOKEN` set for this run; fine for a one-off but worth
  setting for repeated runs to avoid rate limiting.
- **`torch.nn.functional.scaled_dot_product_attention` without a memory-efficient backend** — this
  torch build wasn't compiled with it, so every policy here pays for a slower attention kernel than
  its best case on this hardware. All three are affected equally, so the relative ranking above
  should hold, but the absolute numbers have headroom left on the table.
