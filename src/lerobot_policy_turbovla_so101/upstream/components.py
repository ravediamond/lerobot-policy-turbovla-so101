# ------------------------------------------------------------------------
# Vendored from H-EmbodVis/TurboVLA @ a3365e8f1787d30bafd99d2806edbf4dc82b5370
#   turbovla/models/components/utils.py       (_get_clones, MLP, _get_activation_fn)
#   turbovla/models/components/transformer.py (TransformerEncoderLayer)
# plus timm.layers.drop (drop_path, DropPath), copied so the plugin does not depend on timm.
#
# Grounding DINO (https://github.com/IDEA-Research/GroundingDINO), Copyright (c) 2023 IDEA.
# DETR, Copyright (c) Facebook, Inc. and its affiliates. timm, Copyright (c) Ross Wightman.
# Licensed under the Apache License, Version 2.0 (see LICENSE.upstream).
#
# Only the pieces TurboVLA uses are kept; their bodies are unchanged.
# ------------------------------------------------------------------------
import copy

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _get_clones(module, N, layer_share=False):  # noqa: N803
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        use_optimized_attention=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead
        self.use_optimized_attention = bool(use_optimized_attention)

    def with_pos_embed(self, tensor, pos: Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ):
        # repeat attn mask
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.repeat(self.nhead, 1, 1)

        q = k = self.with_pos_embed(src, pos)

        if self.use_optimized_attention:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)[0]
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        # Upstream note: src_key_padding_mask is accepted but deliberately not passed to the
        # attention (the line doing so is commented out upstream). Kept as is.
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (timm.layers.drop.drop_path)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (timm.layers.drop.DropPath)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
