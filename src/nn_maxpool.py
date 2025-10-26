import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        return self.maxpool(x)

net = Net()

writer = SummaryWriter('logs')

for index,data in enumerate(dataloader):
    images, labels = data
    output = net(images)
    print(images.shape)
    print(output.shape)

    writer.add_images('input', images, index)
    writer.add_images('output', output, index)

writer.close()