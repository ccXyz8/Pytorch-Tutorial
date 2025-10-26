import torch
import torchvision
from torch import nn
from torch.utils.data import dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

net = Net()
writer = SummaryWriter('logs')

for index,data in enumerate(dataloader):
    imgs, labels = data
    output = net(imgs)
    writer.add_images('input',imgs, index)
    writer.add_images('output',output, index)

writer.close()