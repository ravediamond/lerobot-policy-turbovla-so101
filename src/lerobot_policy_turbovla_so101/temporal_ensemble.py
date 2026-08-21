#!/usr/bin/env python

# Copyright 2026 ravediamond. All rights reserved.
#
# Adapted from lerobot.policies.act.modeling_act.ACTTemporalEnsembler
# (Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team), licensed Apache-2.0.
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
"""Temporal ensembling for single-shot, parallel-decode action-chunk policies.

TurboVLA's decoder (like ACT's) predicts a whole chunk in one forward pass with no iterative
denoising step to correct — RTC (real-time chunking) does not apply here, since there is no
trajectory to re-splice. The applicable smoothing technique for this architecture is temporal
ensembling: overlapping chunk predictions from consecutive timesteps are exponentially averaged,
which removes the visible per-chunk-boundary jitter a plain action queue has.

The algorithm itself (Algorithm 2 in https://huggingface.co/papers/2304.13705) has nothing
ACT-specific about it — it operates purely on the (batch, chunk_size, action_dim) output tensor —
so it is reproduced here rather than imported across policy packages.
"""

import torch
from torch import Tensor


class TemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """
        The weights are calculated as wi = exp(-temporal_ensemble_coeff * i) where w0 is the oldest
        action. They are then normalized to sum to 1 by dividing by the sum of weights. Setting the
        coefficient to 0 uniformly weighs all actions. Setting it positive gives more weight to older
        actions; negative gives more weight to newer actions.

        Uses an online method for computing the average rather than caching a history of actions.
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, updates the temporal ensemble
        for all time steps, and pops/returns the next single action in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action
