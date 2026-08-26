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
"""Bidirectional vision-language interaction, the core of TurboVLA.

Each of the `N` layers runs vision->text and text->vision cross-attention off the same pre-update
snapshot, so both directions update simultaneously rather than chaining. The paper's ablation shows
this beats single-direction fusion (97.7% vs 96.1-96.5%).

Attention runs through `scaled_dot_product_attention` for backend portability (ROCm/gfx1201, HF
Jobs CUDA images) without a `flash-attn` dependency.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def get_activation_fn(activation: str) -> nn.Module:
    """Return an activation module by name, matching ACT's supported set."""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"`feedforward_activation` must be 'relu' or 'gelu'. Got {activation}.")


class TurboVLAAttention(nn.Module):
    """Multi-head attention on top of `scaled_dot_product_attention`.

    Used for both self-attention (pass the same tensor as `query` and `key_value`) and
    cross-attention. `key_padding_mask` is a boolean `(batch, kv_len)` tensor where True marks a
    *valid* key, matching the convention of `attention_mask` as returned by HF tokenizers.
    """

    def __init__(self, dim_model: int, n_heads: int, dropout: float):
        super().__init__()
        if dim_model % n_heads != 0:
            raise ValueError(f"`dim_model` {dim_model} must be divisible by `n_heads` {n_heads}.")
        self.n_heads = n_heads
        self.head_dim = dim_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, L, D) -> (B, n_heads, L, head_dim)."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, q_len, dim_model = query.shape

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))

        attn_mask = None
        if key_padding_mask is not None:
            # (B, L_kv) -> (B, 1, 1, L_kv); SDPA reads a bool mask as "True = take part".
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(batch_size, q_len, dim_model)
        return self.out_proj(out)


class TurboVLAFeedForward(nn.Module):
    """The standard two-layer position-wise feed-forward block."""

    def __init__(self, dim_model: int, dim_feedforward: int, dropout: float, activation: str):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TurboVLAFusionLayer(nn.Module):
    """One bidirectional vision-language cross-attention layer (pre-norm, residual).

    Both streams are updated from the *same* normalized snapshot of the other stream, so neither
    direction sees the other's update within a layer.
    """

    def __init__(self, config):
        super().__init__()
        self.norm_vision_attn = nn.LayerNorm(config.dim_model)
        self.norm_text_attn = nn.LayerNorm(config.dim_model)

        # text -> vision: patch tokens query the instruction tokens.
        self.text_to_vision_attn = TurboVLAAttention(config.dim_model, config.n_heads, config.dropout)
        # vision -> text: instruction tokens query the patch tokens.
        self.vision_to_text_attn = TurboVLAAttention(config.dim_model, config.n_heads, config.dropout)

        self.norm_vision_ffn = nn.LayerNorm(config.dim_model)
        self.norm_text_ffn = nn.LayerNorm(config.dim_model)
        self.vision_ffn = TurboVLAFeedForward(
            config.dim_model, config.dim_feedforward, config.dropout, config.feedforward_activation
        )
        self.text_ffn = TurboVLAFeedForward(
            config.dim_model, config.dim_feedforward, config.dropout, config.feedforward_activation
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        vision: Tensor,
        text: Tensor,
        text_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            vision: (B, n_vision_tokens, dim_model)
            text: (B, n_text_tokens, dim_model)
            text_padding_mask: (B, n_text_tokens) bool, True where the token is real.

        Returns:
            The updated `(vision, text)` pair, same shapes as the inputs.
        """
        vision_normed = self.norm_vision_attn(vision)
        text_normed = self.norm_text_attn(text)

        # Both branches read the pre-update snapshot, so neither direction sees the other's result.
        vision = vision + self.dropout(
            self.text_to_vision_attn(vision_normed, text_normed, key_padding_mask=text_padding_mask)
        )
        text = text + self.dropout(self.vision_to_text_attn(text_normed, vision_normed))

        vision = vision + self.dropout(self.vision_ffn(self.norm_vision_ffn(vision)))
        text = text + self.dropout(self.text_ffn(self.norm_text_ffn(text)))
        return vision, text


class TurboVLAVLInteraction(nn.Module):
    """The stack of `config.n_fusion_layers` bidirectional fusion layers."""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TurboVLAFusionLayer(config) for _ in range(config.n_fusion_layers)])
        self.norm_vision = nn.LayerNorm(config.dim_model)
        self.norm_text = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        vision: Tensor,
        text: Tensor,
        text_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            vision, text = layer(vision, text, text_padding_mask=text_padding_mask)
        return self.norm_vision(vision), self.norm_text(text)
