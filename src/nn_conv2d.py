import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(3, 6, 3)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.c1(x))

net = Net()

writer = SummaryWriter('logs')
step=0
for data in dataloader:
    images, labels = data
    output = net(images)
    print(images.shape)
    print(output.shape)

    writer.add_images('input', images, step)

    output = torch.reshape(output, (-1,3, 30, 30))
    writer.add_images('output', output, step)

    step+=1

writer.close()