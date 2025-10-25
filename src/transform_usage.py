from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "../dataset/my_dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer=SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("ants_image", tensor_img, 0)

writer.close()