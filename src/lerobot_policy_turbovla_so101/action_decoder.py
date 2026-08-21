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
"""ACT-style parallel action decoder.

`H = config.chunk_size` learnable action queries attend to each other and cross-attend to the fused
vision-language memory, then a linear head reads out the whole action chunk in a single forward
pass. Same idea as ACT's decode — no action tokenization, no autoregressive step, no VAE (TurboVLA
trains on a plain L1 objective, so ACT's `use_vae`/`latent_dim`/`kl_weight` have no analogue here).
"""

from torch import Tensor, nn

from .vl_interaction import TurboVLAAttention, TurboVLAFeedForward


class TurboVLADecoderLayer(nn.Module):
    """Pre-norm decoder layer: query self-attention, cross-attention to memory, feed-forward."""

    def __init__(self, config):
        super().__init__()
        self.norm_self_attn = nn.LayerNorm(config.dim_model)
        self.self_attn = TurboVLAAttention(config.dim_model, config.n_heads, config.dropout)

        self.norm_cross_attn = nn.LayerNorm(config.dim_model)
        self.cross_attn = TurboVLAAttention(config.dim_model, config.n_heads, config.dropout)

        self.norm_ffn = nn.LayerNorm(config.dim_model)
        self.ffn = TurboVLAFeedForward(
            config.dim_model, config.dim_feedforward, config.dropout, config.feedforward_activation
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normed = self.norm_self_attn(queries)
        # No causal mask: the chunk is decoded in parallel, so every query sees every other one.
        queries = queries + self.dropout(self.self_attn(normed, normed))

        queries = queries + self.dropout(
            self.cross_attn(
                self.norm_cross_attn(queries), memory, key_padding_mask=memory_padding_mask
            )
        )

        queries = queries + self.dropout(self.ffn(self.norm_ffn(queries)))
        return queries


class TurboVLAActionDecoder(nn.Module):
    """`H` learnable action queries -> a continuous `(H, action_dim)` chunk."""

    def __init__(self, config, action_dim: int):
        super().__init__()
        self.config = config
        self.query_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.layers = nn.ModuleList(
            [TurboVLADecoderLayer(config) for _ in range(config.n_decoder_layers)]
        )
        self.norm = nn.LayerNorm(config.dim_model)
        self.action_head = nn.Linear(config.dim_model, action_dim)

    def forward(
        self,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            memory: (B, n_memory_tokens, dim_model) fused vision-language tokens.
            memory_padding_mask: (B, n_memory_tokens) bool, True where the token is real.

        Returns:
            (B, chunk_size, action_dim) continuous action chunk.
        """
        batch_size = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # `expand` returns a view of the embedding weight; the residual adds in the layers below
        # allocate new tensors, so nothing writes back into the parameter.
        queries = queries.to(memory.dtype)

        for layer in self.layers:
            queries = layer(queries, memory, memory_padding_mask=memory_padding_mask)

        return self.action_head(self.norm(queries))
