import numpy as np
import torch
from torch import nn

class My_module(nn.Module):
    def __init__(self):
        super(My_module, self).__init__()

    def forward(self, x):
        output = x+1
        return output

my_module = My_module()

x = np.random.randn(10)
x = torch.tensor(x)
print(x)
print(x.shape)
output = my_module(x)
print(output)