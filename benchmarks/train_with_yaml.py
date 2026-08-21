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
"""Run `lerobot-train` with a YAML overlay that the CLI cannot express.

Why this exists: draccus builds its argument parser from dataclass *fields*, so the entries of a
`dict[str, ...]` field have no CLI flags. `dataset.image_transforms.tfs` is such a dict, which means
there is no way to say "resize every frame to 224x224" on the command line — and
`lerobot-train --config_path=...` cannot be borrowed for it either, because `TrainPipelineConfig`
defines `from_pretrained`, so the CLI routes `--config_path` to a checkpoint loader instead of to
draccus's YAML overlay.

This launcher takes the one path that is left: parse the YAML overlay plus the usual CLI flags into
a config object, then hand that object straight to `train()`. LeRobot's `@parser.wrap()` returns
early when it is passed an already-built config, so nothing is re-parsed and behaviour is otherwise
identical to `lerobot-train`.

Usage:

    python benchmarks/train_with_yaml.py <overlay.yaml> [any lerobot-train flags...]

Example, forcing both policies in a comparison to see identical 224px input:

    python benchmarks/train_with_yaml.py resize224.yaml \
        --dataset.repo_id=user/dataset --policy.type=act --steps=8000
"""

import sys

import draccus
import lerobot.policies  # noqa: F401  (populates the built-in policy registry)
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        raise SystemExit(f"usage: {sys.argv[0]} <overlay.yaml> [lerobot-train flags...]")

    overlay, cli_args = sys.argv[1], sys.argv[2:]

    # Same first step as `lerobot_train.main()`: make third-party policies (turbovla_so101) resolvable.
    register_third_party_plugins()

    cfg = draccus.parse(config_class=TrainPipelineConfig, config_path=overlay, args=cli_args)
    train(cfg)


if __name__ == "__main__":
    main()
