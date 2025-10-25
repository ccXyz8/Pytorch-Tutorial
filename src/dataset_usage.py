from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_paths = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = Image.open(os.path.join(self.root_dir,self.label_dir,img_name))
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_paths)

root_dir = os.path.join('..','dataset','my_dataset','train')
label_dir = "ants"
ants_dataset = MyDataset(root_dir,label_dir)
img,label = ants_dataset[0]
img.show()