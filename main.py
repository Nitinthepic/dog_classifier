import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import heapq
from os.path import dirname, join as pjoin
import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.io import read_image
# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

class CustomImageDataset(Dataset):

    def __init__(self, dog_list, transform=None, target_transform=None):
        self.dog_list = dog_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dog_list)

    def __getitem__(self, idx):
        img_path = os.path.join(image_path,self.dog_list[idx][0])
        image = read_image(img_path)
        label = self.dog_list[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

annot_path = 'data/Annotation'
image_path = 'data/Images'

def main():
    valid = []
    imageList = list()
    for element in os.listdir(image_path):
        if(element.startswith("n0")):
            for img_id in os.listdir(os.path.join(image_path, element)):
                path = os.path.join(element, img_id)
                label = element.split('-')[1]
                valid.append((path, label))


    
    dog = CustomImageDataset(valid)
    train_dataloader = DataLoader(dog, shuffle=True)
    
    train_features, train_labels = next(iter(train_dataloader))
    print(train_labels)
    #print(f"Feature batch shape: {train_features.size()}")
    #print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze().permute(1,2,0)
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")




main()

