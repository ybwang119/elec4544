import os
import pandas as pd
import pathlib
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

current_dir = pathlib.Path(__file__).parent.resolve()

class FlowerDataset(Dataset):
    def __init__(self, annotations_file:str, img_dir:str, transform=None,
        target_transform=None):
        
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(
            os.path.join(*[current_dir,self.img_dir, annotations_file])
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label