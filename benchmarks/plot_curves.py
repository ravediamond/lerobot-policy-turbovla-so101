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
"""Plot the two curves a training run needs, from artifacts already on disk.

Held-out MAE vs step is the evidence: it says how well the policy predicts actions on frames it
never trained on, and at which budget. Training loss vs step is the health check that has to be
read first -- it shows whether the run actually converged, and (when comparing runs) whether they
converged *comparably*. A MAE gap between a run that plateaued and one still descending is not an
architecture result, it is a statement that one run was too short.

The two come from different places and neither needs a GPU:

* training loss is parsed out of the `lerobot-train` stdout logs (`outputs/*.log`), which is
  where it lives when `--wandb.enable=false`;
* held-out MAE is read from the JSON that `eval_curve.py --json` writes, so re-plotting never
  re-runs the evaluation.

When plotting more than one run, losses are NOT comparable across policies in absolute terms -- ACT's
total includes a KL term TurboVLA has no analogue for, and each policy normalizes its action space
its own way. The loss panel is for reading the *shape* of each curve. Only the MAE panel, where
every policy is scored on identical held-out frames at one common horizon with the same normalizer,
compares across policies.

Usage
-----
    python benchmarks/plot_curves.py --loss-log outputs/cmp_turbovla_so101.log \
        --mae-json benchmarks/results/curves.json \
        --out benchmarks/results/curves.png
"""

import argparse
import json
import re
from pathlib import Path

# The step count in the log text is NOT usable: lerobot-train prints it through a human-readable
# formatter, so step 5600 appears as `step:6K` and everything from 1000 up collapses into a handful
# of rounded labels. The exact count is in the tqdm bar that precedes each log line -- the bar and
# the INFO line share a line because tqdm writes carriage returns to the same stream:
#
#     Training: 93%|#########3| 5600/6000 [7:19:13<31:18, 4.70s/step]INFO ... step:6K ... loss:0.349
#
# so anchor on the bar's `<done>/<total>` and take the loss from the INFO text that follows it.
STEP_LOSS = re.compile(r"(\d+)/\d+ \[[^\]]*\]INFO[^\n]*?\bloss:([0-9.]+)")


def parse_loss_log(path: Path) -> tuple[str, list[tuple[int, float]]]:
    """Returns `(run_name, [(step, loss), ...])` for one lerobot-train log."""
    text = path.read_text(errors="replace").replace("\r", "\n")
    points = [(int(s), float(v)) for s, v in STEP_LOSS.findall(text)]
    # A run restarted into the same log would repeat step numbers; keep the last value seen for
    # each step so the curve stays monotonic in x rather than doubling back.
    dedup = dict(points)
    name = path.stem
    return name, sorted(dedup.items())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loss-log", action="append", default=[], metavar="PATH")
    p.add_argument("--mae-json", default="", metavar="PATH")
    p.add_argument("--out", default="benchmarks/results/curves.png")
    p.add_argument("--title", default="TurboVLA-SO101 training curve")
    args = p.parse_args()

    if not args.loss_log and not args.mae_json:
        raise SystemExit("nothing to plot: pass --loss-log and/or --mae-json")

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is not installed. `pip install matplotlib`.") from e

    losses = [parse_loss_log(Path(f)) for f in args.loss_log]
    losses = [(n, pts) for n, pts in losses if pts]

    mae = {}
    if args.mae_json:
        payload = json.loads(Path(args.mae_json).read_text())
        mae = {n: [(d["step"], d["mae"]) for d in c] for n, c in payload["curves"].items()}

    panels = int(bool(losses)) + int(bool(mae))
    if panels == 0:
        raise SystemExit("no usable data found in the inputs")

    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 5), squeeze=False)
    axes = axes[0]
    i = 0

    if losses:
        ax = axes[i]
        i += 1
        for name, pts in losses:
            ax.plot([s for s, _ in pts], [v for _, v in pts], label=name, linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel("training loss")
        ax.set_yscale("log")
        shape_note = " (shape only -- not comparable across policies)" if len(losses) > 1 else ""
        ax.set_title("Training loss" + shape_note)
        ax.grid(alpha=0.3)
        ax.legend()

    if mae:
        ax = axes[i]
        for name, pts in mae.items():
            ax.plot([s for s, _ in pts], [v for _, v in pts], marker="o", label=name, linewidth=1.5)
            # The lowest point is the honest one to quote; if it is not the final checkpoint the
            # run overfit, and marking it keeps that visible in the figure rather than only in the
            # table.
            best_step, best_mae = min(pts, key=lambda t: t[1])
            if best_step != pts[-1][0]:
                ax.annotate(
                    f"best {best_mae:.3f}",
                    (best_step, best_mae),
                    textcoords="offset points",
                    xytext=(0, -14),
                    fontsize=8,
                    ha="center",
                )
        ax.set_xlabel("step")
        ax.set_ylabel("held-out MAE")
        ax.set_title("Held-out action error (same frames, horizon, normalizer)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    for name, pts in losses:
        print(
            f"  loss  {name:<12} {len(pts):>4} points, step {pts[0][0]}-{pts[-1][0]}, final {pts[-1][1]:.4f}"
        )
    for name, pts in mae.items():
        best_step, best_mae = min(pts, key=lambda t: t[1])
        print(f"  mae   {name:<12} {len(pts):>4} points, best {best_mae:.4f} at step {best_step}")


if __name__ == "__main__":
    main()
