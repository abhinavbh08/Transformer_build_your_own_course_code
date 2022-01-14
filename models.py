import torch
import torch.nn as nn
from attention_functions import MultiHeadAttention
from pos_enc import PositionalEncoding
import math


class LNorm(nn.Module):
    """Class to do layer normalisation and addition of residual connection."""
    def __init__(self, lnorm_size, dropout):
        super(LNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lnorm_size)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))


class TransformerEncoderDecoder(nn.Module):
    """Base class for combining functionality of transformer encoder and decoder in a single forward call."""
    def __init__(self, encoder, decoder, **kwargs):
        super(TransformerEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x, *args)
        enc_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, enc_state)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, query, key, value, hidden_size, num_head, dropout, lnorm_size, ffn_input, ffn_hidden):
        # In a transformer the query, key, value, hidden_size, ffn_input, all will be of the same size/dimension.
        super(TransformerEncoderBlock, self).__init__()

        # Encoder self attention.
        self.self_attention = MultiHeadAttention(query, key, value, hidden_size, num_head, dropout)
        self.l_norm_attention = LNorm(lnorm_size, dropout)

        # Encoder feed forward layer.
        self.first_ffl = nn.Linear(ffn_input, ffn_hidden)
        self.relu = nn.ReLU()
        self.second_ffl = nn.Linear(ffn_hidden, hidden_size)
        self.l_norm_ffn = LNorm(lnorm_size, dropout)

    def forward(self, x, valid_lens):
        # passing input through the self - attention layer and layer normalisation
        attn_op = self.self_attention(x, x, x, valid_lens)
        y = self.l_norm_attention(x, attn_op)

        # passing output of layer normalisation to the feed forward layer.
        ffn_op = self.second_ffl(self.relu(self.first_ffl(y)))
        return self.l_norm_ffn(y, ffn_op)


class TransformerEncoder(nn.Module):
    """Transformer encoder composed of multiple transformer encoder blocks."""
    def __init__(self, query, key, value, hidden_size, num_head, dropout, lnorm_size, ffn_input, ffn_hidden, vocab_size, num_layers):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        # Initialise word embedding matrix
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Initialise the position embedding matrix
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(str(i), TransformerEncoderBlock(query, key, value, hidden_size, num_head, dropout, lnorm_size, ffn_input, ffn_hidden))

    def forward(self, x, valid_lens, *args):
        # Pass word embedding to the positional embedding class object so that they get added together.
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.hidden_size))
        # Loop over all the blocks passing one input to the next.
        for i, block in enumerate(self.blocks):
            x = block(x, valid_lens)
        return x    # (B, max_len, hidden_dim)



class Transformerdecoderblock(nn.Module):
    """Single Transformer decoder block."""
    def __init__(self, query, key, value, hidden_size, num_head, dropout, lnorm_size, ffn_input, ffn_hidden, i):
        super(Transformerdecoderblock, self).__init__()
        self.i = i
        # Decoder self attention layer
        self.dec_self_att = MultiHeadAttention(query, key, value, hidden_size, num_head, dropout)
        self.l_norm_dec_self_attention = LNorm(lnorm_size, dropout)

        # ENcoder decoder attention layer
        self.enc_dec_attention = MultiHeadAttention(query, key, value, hidden_size, num_head, dropout)
        self.l_norm_enc_dec_att = LNorm(lnorm_size, dropout)

        # feed forward layer
        self.first_ffl = nn.Linear(ffn_input, ffn_hidden)
        self.relu = nn.ReLU()
        self.second_ffl = nn.Linear(ffn_hidden, hidden_size)
        self.l_norm_fnn = LNorm(lnorm_size, dropout)

    def forward(self, x, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            keys = x
        else:
            keys = torch.cat((state[2][self.i], x), axis=1) # This is required during prediction because each prediction is being done word by word.
        state[2][self.i] = keys
        if self.training:
            bs, max_len, dim = x.shape
            dec_valid_lens = torch.arange(1, max_len + 1, device=x.device).repeat(bs, 1)    # (B, max_len)
        else:
            dec_valid_lens = None # No need for maskind during predicition since we do not have future tokens.

        # Passing the input of decoder to the decoder self attention block
        op_att1 = self.dec_self_att(x, keys, keys, dec_valid_lens)
        y = self.l_norm_dec_self_attention(x, op_att1)
        # Passing the output of layer_ normalisation to the encoder  decoder attention block.
        op_att2 = self.enc_dec_attention(y, enc_outputs, enc_outputs, enc_valid_lens)
        z = self.l_norm_enc_dec_att(y, op_att2)        
        # Passing the output of previous layer normalisation to the feedforward layer.
        op_ffn = self.second_ffl(self.relu(self.first_ffl(z)))
        return self.l_norm_fnn(z, op_ffn), state


class TransformerDecoder(nn.Module):
    """Transformer decoder consisting of many transformer decoder blocks."""
    def __init__(self, query, key, value, hidden_size, num_head, dropout, lnorm_size, ffn_input, ffn_hidden, vocab_size, num_layers):
        super(TransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(str(i), Transformerdecoderblock(query, key, value, hidden_size, num_head, dropout, lnorm_size, ffn_input, ffn_hidden, i))
        # Final linear layer for converting hidden dimension to vocab size.
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, x, state):
        # Pass word embedding to the positional embedding class object so that they get added together.
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.hidden_size))
        for block in self.blocks:
            x, state = block(x, state)
        return self.linear(x), state