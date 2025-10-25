import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='../dataset', train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

img,label = test_set[0]
print(img.shape)
print(label)

writer = SummaryWriter('logs')
step=0
for data in test_loader:
    imgs,labels = data
    writer.add_images('test_data',imgs,step)
    step+=1

writer.close()