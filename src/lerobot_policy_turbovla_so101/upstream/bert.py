# ------------------------------------------------------------------------
# Vendored from H-EmbodVis/TurboVLA @ a3365e8f1787d30bafd99d2806edbf4dc82b5370
#   turbovla/text/bert.py (Grounding DINO, https://github.com/IDEA-Research/GroundingDINO,
#   Copyright (c) 2023 IDEA. Modified for TurboVLA.)
# Licensed under the Apache License, Version 2.0 (see LICENSE.upstream).
#
# Changes for the LeRobot plugin: `BertModelWarper.forward` keeps only the encoder path TurboVLA uses
# (no decoder cross-attention, no KV cache, no head mask) so that it runs on transformers 4.x and 5.x:
# transformers 5 removed `get_head_mask` and the encoder's `head_mask` argument. Same embeddings, same
# extended mask (`get_extended_attention_mask` on the 3-D sub-sentence mask), same encoder call;
# `tests/test_upstream_parity.py` checks the output against upstream on transformers 4.56.
# `generate_masks_with_special_tokens` is unchanged.
# ------------------------------------------------------------------------

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertModelWarper(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.config = bert_model.config
        self.embeddings = bert_model.embeddings
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler

        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.invert_attention_mask = bert_model.invert_attention_mask
        # transformers < 5 still requires a (None) head mask per layer in the encoder call.
        self._legacy_head_mask = hasattr(bert_model, "get_head_mask")
        if self._legacy_head_mask:
            self.get_head_mask = bert_model.get_head_mask

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **unused):
        if self.config.is_decoder:
            raise ValueError("TurboVLA uses BERT as an encoder only")
        if input_ids is None:
            raise ValueError("input_ids is required")
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        encoder_kwargs = {"attention_mask": extended_attention_mask}
        if self._legacy_head_mask:
            encoder_kwargs["head_mask"] = self.get_head_mask(None, self.config.num_hidden_layers)
            encoder_kwargs["return_dict"] = True
        encoder_outputs = self.encoder(embedding_output, **encoder_kwargs)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


def generate_masks_with_special_tokens(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )

        previous_col = col

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.to(torch.long)
