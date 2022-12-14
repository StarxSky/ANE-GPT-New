import torch 
import logging 

logger = logging.getLogger(__name__)

from torch import nn 
from Core import decoder, ffn, multihead_attention
from Core.testing_utils import assert_rank, assert_shape


# AppleNeuralEngineGPT
class AppleNeuralEngineGPT(nn.Module) :
    def __init__(
        self,
        *, 
        embed_dim=512, 
        ffn_dim=2048,
        dec_self_attn_type=multihead_attention.ResidualSelfAttention,
        dec_ffn_type=ffn.ResidualFFN,
        nb_dec_layers=6,
        nb_attention_heads=8,
        dropout=0.1,
        return_intermediate_decoder_layers=False,
        cfg,
        **kwargs
        ) -> None:
        """
        Args:
            embed_dim:                          Dimensionality of the embedding space attention is computed in
            ffn_dim:                            Number of channels to use in the feed-forward network's hidden layer
            dec_self_attn_type:                 The self-attention module to use in each layer of Transformer decoder
            dec_ffn_type:                       The feed-forward network module to use in each layer of the Transformer decoder
            nb_dec_layers:                      Number of identically configured Transformer decoder layers to stack
            nb_attention_heads:                 Number of attention heads generate in each and every attention block in the Transformer model
            dropout:                            The dropout probability (`1- keep_probability`) to use on both the attention weights and the
                                                output of the attention block in each and every attention block in the Transformer model
            return_intermediate_decoder_layers: If True, returns the output of all Transformer decoder layers stacked in the 0-th axis.
                                                Example use case: When supervising the Transformer on all intermediate outputs for training stability
        Note: The positional embeddings are passed by the caller of forward() and are not part of this module

        Note: The default configuration reflects the "base" configuration in the original Transformer paper [1] (page 9, table 3, row 1)

        [1] https://arxiv.org/pdf/1706.03762
        """
        super(AppleNeuralEngineGPT, self).__init__()


        self.drop = nn.Dropout2d(0.04)
        self.embed = nn.Embedding(cfg.vocab, cfg.embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.embed_dim, 1, cfg.block_size))

        self.decoder = decoder.TransformerDecoder(
            layer=decoder.TransformerDecoderLayer(
                embed_dim,
                ffn_dim,
                dec_self_attn_type,
                dec_ffn_type,
                nb_attention_heads,
                dropout,
            ),
            num_layers=nb_dec_layers)

        self.pool = nn.AdaptiveAvgPool2d(cfg.block_size)
        self.head = nn.Linear(cfg.block_size, cfg.vocab)
        self.return_intermediate_decoder = return_intermediate_decoder_layers
        self.embed_dim = embed_dim
        
    def forward(
        self,
        inputs, 
        decoder_k_mask=None, 
        decoder_qk_mask=None,
        ) :

        """
        Notation:
            src_seq_len:        The sequence length of the source sequence which is the input to the TransformerEncoder
            tgt_seq_len:        The sequence length of the target sequence which is the input to the TransformerDecoder

        Args:
           
            decoder_input:      Float tensor input to the TransformerDecoder
            decoder_pos_embed:  Same shape and dtype as `decoder_input`. Serves as additive positional encodings to `decoder_input`
            decoder_k_mask:     Float tensor similar to the `tgt_key_padding_mask` in `torch.nn.Transformer.forward`. Example use: masking zero-padded tokens in the target sequence
            decoder_qk_mask:    Float tensor similar to `tgt_mask` in `torch.nn.Transformer.forward`. Example use: masking future tokens in the decoder self-attention

        Shapes:
            
            decoder_input:      (batch_size, embed_dim, 1, tgt_seq_len)
            decoder_pos_embed:  (batch_size, embed_dim, 1, tgt_seq_len)
            decoder_k_mask:     (batch_size, tgt_seq_len, 1, 1)
            decoder_qk_mask:    (batch_size, tgt_seq_len, 1, tgt_seq_len)

        Returns:
            decoder_output:     Output of the TransformerDecoder
            


        Note: All arguments ending in "_mask", are applied additively on the intermediate tensor right before softmax in the attention function.
        The recommended float value for preventing attention is -1e4. This allows for composition of multiple masks while staying in the float16-friendly range.
        Use a value of 0 to keep attention unchanged.
        """

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

        out = out.sum(-2)
        out = self.pool(out)
        out = self.head(out)

        return out
