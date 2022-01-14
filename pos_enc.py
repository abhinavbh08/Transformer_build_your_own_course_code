import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):

    def __init__(self, dimension, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos = torch.zeros((1, max_len, dimension))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(
        0, dimension, 2, dtype=torch.float32) / dimension)
        self.pos[:, :, 0::2] = torch.sin(x)
        self.pos[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        # Add positional embeddings to the word embeddings obtained as input.
        x = x + self.pos[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)