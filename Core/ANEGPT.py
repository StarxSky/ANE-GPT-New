import os 
import torch 
import logging

logger = logging.getLogger(__name__)

from torch import nn 
from Core import decoder, ffn, multihead_attention
from Core.testing_utils import assert_rank, assert_shape



class TempCFG :
    block_size = 4
    vocab = 234
    embed_dim=512
    ffn_dim=2048
    dec_self_attn_type=multihead_attention.ResidualSelfAttention
    dec_ffn_type=ffn.ResidualFFN
    nb_dec_layers=6
    nb_attention_heads=8
    dropout=0.1
    return_intermediate_decoder_layers=False



class AppleNeuralEngineGPT(nn.Module) :
    def __init__(self, cfg=TempCFG):
        super(AppleNeuralEngineGPT, self).__init__()

        self.drop = nn.Dropout(0.04)
        self.embed = nn.Embedding(cfg.vocab, cfg.embed_dim)
        self.pos_emb = nn.Parameter(torch.ones(1, cfg.embed_dim, 1, cfg.block_size))

        self.decoder = decoder.TransformerDecoder(
            layer=decoder.TransformerDecoderLayer(
                embed_dim=cfg.embed_dim,
                ffn_dim=cfg.ffn_dim,
                self_attn_cls=cfg.dec_self_attn_type, 
                ffn_cls=cfg.dec_ffn_type,
                n_head=cfg.nb_attention_heads,
                dropout=cfg.dropout,
                ),
                num_layers=cfg.nb_dec_layers
        )
        
        self.ln = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab, bias=False)
        self.return_intermediate_decoder = cfg.return_intermediate_decoder_layers
        self.embed_dim = cfg.embed_dim
        
    def forward(
        self,
        inputs, 
        decoder_k_mask=None, 
        decoder_qk_mask=None,
        ) :


        b, lens = inputs.size()
        x = inputs.unsqueeze(-1)
        decoder_embed = self.embed(x).permute(0,3,2,1)
        decoder_pos_embed = self.pos_emb[:, :, :, :lens]
        decoder_input = self.drop(decoder_embed + decoder_pos_embed)

        # Verify ranks
        assert_rank(decoder_pos_embed, "decoder_pos_embed", 4)
        assert_rank(decoder_k_mask, "decoder_k_mask", 4)
        assert_rank(decoder_k_mask, "decoder_qk_mask", 4)


        # Verify and prepare decoder inputs
        batch_size, _, _, tgt_seq_len = decoder_input.shape
        assert_shape(decoder_input, "decoder_input",
                     [batch_size, self.embed_dim, 1, tgt_seq_len])

        
        if decoder_k_mask is not None:
            assert_shape(decoder_k_mask, "decoder_k_mask",
                         [batch_size, tgt_seq_len, 1, 1])
        if decoder_qk_mask is not None:
            assert_shape(decoder_qk_mask, "decoder_qk_mask",
                         [batch_size, tgt_seq_len, 1, tgt_seq_len])
        

        # TransformerDecoder forward pass
        out = self.decoder(
            decoder_input,
            decoder_k_mask=decoder_k_mask,
            decoder_qk_mask=decoder_qk_mask,
            decoder_pos_embed=decoder_pos_embed,
            return_intermediate=self.return_intermediate_decoder,
        )

        out = self.ln(out.transpose(2,1))
        out = self.head(out)
        return out 
