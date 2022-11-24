import imp
from torch import nn
from models.transformer.embedding import PositionalEncoding

class MaskedPositionalEncoding(PositionalEncoding):
    """Positional encoding module
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super(MaskedPositionalEncoding, self).__init__(d_model, dropout_rate, max_len)
    def forward(self, x, mask):
        x = x * self.xscale + self.pe[:, mask]
        return self.dropout(x)