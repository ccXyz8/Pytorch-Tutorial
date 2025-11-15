import torch
from torch import nn
import math
import torch.nn.functional as F

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__(vocab_size, embedding_dim,padding_idx=1)