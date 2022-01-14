import torch
import torch.nn as nn

def sequence_mask(x, valid_len, value=0):
    """Masks a 2d tensor x if with a given value on places where padding is done."""

    maxlen = x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=x.device)[None, :] < valid_len[:, None]  # (x.shape)
    x[~mask] = value
    return x


class MaskedCELoss(nn.CrossEntropyLoss):
    """Cross entropy loss with masking so that we do not consider the loss on tokens where poadding is done."""
    def forward(self, pred, labels, valid_len):
        # Pred: (B, max_len, vocab_size)
        # labels: (B, max_len)
        # valid_lens: (B)

        weights = torch.ones_like(labels)

        # Create a mask with 1s at the valid entries and 0 otherwise.
        weights = sequence_mask(weights, valid_len) # (B, max_len)
        self.reduction = "none"
        raw_loss = super(MaskedCELoss, self).forward(pred.permute(0, 2, 1), labels) # (B, max_len)

        # Make the loss at the padding entries zero.
        weighted_loss = (raw_loss * weights).mean(dim=1)    # (B)
        return weighted_loss