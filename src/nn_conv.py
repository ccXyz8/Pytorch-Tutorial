import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype

input = torch.tensor(np.random.randint(1,10,(5,5)))
kernel = torch.tensor(np.random.randint(1,2,(3,3)))

input = torch.reshape(input,(1,1,5,5)).to(dtype=torch.float)
kernel = torch.reshape(kernel,(1,1,3,3)).to(dtype=torch.float)
print(input)
print(input.shape)

conv = nn.Conv2d(1,1,(3,3),1)
act = nn.ReLU()
output = conv(input)
print(output)

output2 = F.conv2d(input,kernel,stride=1)
print(output2)