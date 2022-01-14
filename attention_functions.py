import torch
from torch.autograd.grad_mode import F
import torch.nn as nn
import math
from loss import sequence_mask

def masked_softmax(X, valid_lens):
    """Performs a masked softmax operation so that padding tokens are not included in attention score computation."""
    
    # Normal softmax if valid lengths is none.
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # Repeating the valid lengths of inputs so that masking can be done be converting the input to 2 dimensions.
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        # Mask the values with with a large negative value so that its exponential becomes zero.
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# print(masked_softmax(torch.rand(2, 4, 5), torch.tensor([2, 3])))

class DotProductAttention(nn.Module):
    """Basic scaled dot product attention."""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """q = (B, Number of queries, embedding_dim), v = (B, no. of keys, embedding_dim), k = (B, no. of keys, embedding_dim)"""
        dim = queries.shape[-1]
        qk = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(dim)  # (B, n_queries, n_keys)
        self.attention_scores = masked_softmax(qk, valid_lens)  
        return torch.bmm(self.dropout(self.attention_scores), values)   # (B, n_queries, embedding_dim))

# queries, keys, values = torch.normal(0, 1, (2, 6, 10)), torch.ones((2, 6, 10)), torch.ones((2, 6, 10))
# valid_lens = torch.tensor([2, 3])
# dpa = DotProductAttention(0.1)
# print(dpa(queries, keys, values, valid_lens).shape)

class MultiHeadAttention(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, dimensionality, num_heads, dropout):
        # query_dim, key_dim, valid_dim are the dimensionality of the query, key and value.
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        # Weight matrices required to compute multi head attention.
        self.wq = nn.Linear(query_dim, dimensionality, bias=False)
        self.wk = nn.Linear(key_dim, dimensionality, bias=False)
        self.wv = nn.Linear(value_dim, dimensionality, bias=False)
        self.wo = nn.Linear(dimensionality, dimensionality, bias=False)
        # Dot-product attention
        self.attention = DotProductAttention(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.wq(queries)
        keys = self.wk(keys)
        values = self.wv(values)

        # This basically converts the dimensionality of queries to dimensionality/num_heads to perform multi-head attention
        # The shape of these matrices will be (B * num_heads, num_queries, keys or values, dimensionality / num_heads)
        queries = self.transpose_input(queries)
        keys = self.transpose_input(keys)   
        values = self.transpose_input(values)   

        # Repeat the valid lengths by the number of heads, to be useful in masked softamx
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # Do dot product attention
        output = self.attention(queries, keys, values, valid_lens)

        # This will convert the output back to the original input query matrix shape.
        output = self.transpose_output(output)  # (B, num_query, dimensionality)
        output = self.wo(output)    # (B, num_query, dimensionality)
        return output


    def transpose_input(self, x):
        """Converts the input into form of dim / num_heads so that multi head attention can be performed."""
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x

    def transpose_output(self, x):
        """Converst the transformed data back to the input form in which it was received by multi head attention function"""
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x        

# num_hiddens, num_heads = 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.1)
# attention.eval()

# batch_size = 2
# valid_lens = torch.tensor([3, 4])
# q = torch.ones((batch_size, 10, 100))
# attention(q, q, q, valid_lens)