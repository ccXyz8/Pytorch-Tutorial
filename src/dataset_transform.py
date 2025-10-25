import torchvision
from torch.utils.tensorboard import SummaryWriter
from src.tensorboard_usage import writer

writer = SummaryWriter('p10')

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root='../dataset', train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root='../dataset', train=False,transform=dataset_transform,download=True)

for i in range(10):
    img, label = train_set[i]
    writer.add_image('test_set',img,i)

writer.close()

