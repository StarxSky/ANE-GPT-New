#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import logging

logger = logging.getLogger(__name__)

import copy

import torch
import torch.nn as nn

from Core.layer_norm import LayerNormANE


class TransformerDecoder(nn.Module):

    def __init__(self, layer, num_layers):
        super(TransformerDecoder, self).__init__()


        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = LayerNormANE(layer.embed_dim)
        

    def forward(self, decoder_embed, decoder_pos_embed, decoder_k_mask=None, decoder_qk_mask=None,  return_intermediate=False,early_exit_from_layer_idx=None) :

        if early_exit_from_layer_idx:
            assert early_exit_from_layer_idx > 0 and \
                early_exit_from_layer_idx < self.num_layers

        x = decoder_embed
        intermediates = []
        for idx, l in enumerate(self.layers):
            if idx == early_exit_from_layer_idx:
                logger.warning("Early exit taken from TransformerDecoder "
                               f"after {idx}/{self.num_layers} layers")
                break

            x = l(x,  decoder_k_mask, decoder_qk_mask, decoder_pos_embed)

            if return_intermediate:
                intermediates.append(self.norm(x))

        if return_intermediate:
            return torch.stack(intermediates, 0)
        return self.norm(x).sum(-2) # [Batch_size, embed_dim, 1, vocab_size] ===> [batch_size, embed_dim, vocab_size]



class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, ffn_dim, self_attn_cls, ffn_cls, n_head=8, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.self_attn = self_attn_cls(
            embed_dim,
            n_head=n_head,
            dropout=dropout,
        )
        
        self.ffn = ffn_cls(embed_dim, ffn_dim=ffn_dim, dropout=dropout)

    def forward(self, decoder_embed, decoder_k_mask, decoder_qk_mask, decoder_pos_embed):

        x = decoder_embed
        x, _ = self.self_attn(
            qkv=x,
            qpos=decoder_pos_embed,
            kpos=decoder_pos_embed,
            qk_mask=decoder_qk_mask,
            k_mask=decoder_k_mask,
        )

        x = self.ffn(x)
        return x
