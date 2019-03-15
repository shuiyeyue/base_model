from PIL import Image
import torch
import os

class ImageNet(torch.utils.data.Dataset):

  def __init__(self, root, is_train=True, transform=None):
    self.root = root
    self.is_train = is_train
    
    imgs = []
    if self.is_train:
      f = open(os.path.join(root, "meta", "train.txt"))
    else:
      f = open(os.path.join(root, "meta", "val.txt"))
    
    for line in f.readlines():
      line = line.strip().split()
      imgs.append((line[0],int(line[1])))

    f.close()

    self.imgs = imgs
    self.transform = transform
    self.length = len(imgs)

  def __getitem__(self, index):
    
    if self.is_train:
      img_dir = os.path.join(self.root, "train")
    else:
      img_dir = os.path.join(self.root, "val")

    image = Image.open(os.path.join(img_dir, self.imgs[index][0])).convert('RGB')

    return image, self.imgs[index][1]

  def __len__(self):
    return self.length;