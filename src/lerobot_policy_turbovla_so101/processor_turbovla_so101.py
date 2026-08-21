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
from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_turbovla_so101 import TurboVLAConfig


def make_turbovla_so101_pre_post_processors(
    config: TurboVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the TurboVLA policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model
    inputs. The post-processing pipeline handles unnormalization and moves the model outputs back to
    the CPU.

    The default pipeline is all TurboVLA needs. Two things worth knowing about why:

    - The task string rides along as complementary data, so `batch["task"]` reaches the policy
      untouched. Tokenization happens inside the model, against whichever `language_backbone` the
      config names.
    - The visual stream is mapped to `NormalizationMode.IDENTITY` in the config, so the normalizer
      leaves images alone and the model applies the vision backbone's own ImageNet normalization.

    Args:
        config (TurboVLAConfig): The TurboVLA policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
