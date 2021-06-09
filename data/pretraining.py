import os
import torch
from PIL import Image
from torch.utils.data._utils.collate import default_collate


def custom_collate(batch):
    img, label = default_collate(batch)
    if isinstance(img, list):
        img = torch.cat(img, dim=0)
        label = torch.cat(label, dim=0)
    return img, label


class DataReaderPlainImg:
    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = root
        print (f'root is  {root}')
        self.image_files = [f for f in os.listdir(root) if f.endswith(".jpg")]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.image_files[idx])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_files)
