"""
File to handle dataset class and data processing. Resize images to match 224x336
before feeding in to network.
"""
import torch.utils.data as data
import json
import requests
import os
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import numpy as np


class CardDataset(data.Dataset):

    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.index_map = dict(
            [(i, image_path + "/" + file_name) for i, file_name in enumerate(os.listdir(image_path))])
        self.len = len(os.listdir(self.image_path))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        filename = self.index_map[index]
        image = Image.open(filename)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image
