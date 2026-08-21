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
"""TurboVLA as a standalone LeRobot policy plugin.

LeRobot discovers this package because its *distribution* name starts with `lerobot_policy_`;
importing it runs the `@PreTrainedConfig.register_subclass("turbovla_so101")` decorator, after which
`--policy.type=turbovla_so101` resolves anywhere `--policy.type=act` would.

Importing the config is what performs the registration. `transformers` is imported only when a
model is actually built, so plugin discovery stays cheap and `import lerobot` keeps working even if
the backbones cannot be downloaded.
"""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.") from e

from .configuration_turbovla_so101 import TurboVLAConfig
from .modeling_turbovla_so101 import TurboVLA, TurboVLAPolicy
from .processor_turbovla_so101 import make_turbovla_so101_pre_post_processors

__all__ = [
    "TurboVLA",
    "TurboVLAConfig",
    "TurboVLAPolicy",
    "make_turbovla_so101_pre_post_processors",
]
