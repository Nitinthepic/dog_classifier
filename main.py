import exiftool
import torch.nn.functional as F
from os.path import dirname, join as pjoin
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
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
max_Width = 1000
max_Height = 1000

def path_label_creator():
    path_label_list = list()
    et = exiftool.ExifToolHelper()
    for element in os.listdir(image_path):
        if(element.startswith("n0")):
            for img_id in os.listdir(os.path.join(image_path, element)):
                path = os.path.join(element, img_id)
                metadata = et.get_tags(os.path.join(image_path,path),tags=["ImageWidth", "ImageHeight"])
                try:
                    if metadata[0]['File:ImageWidth'] <= max_Width and metadata[0]['File:ImageHeight'] <= max_Height:
                        label = element.split('-')[1]
                        path_label_list.append((path, label))
                except KeyError:
                    print(f"{path} is missing certain label")
    return path_label_list

def main():
    valid = path_label_creator()
    dog = CustomImageDataset(valid)
    train_dataloader = DataLoader(dog, shuffle=True)
    
    dataloader_iterm = (iter(train_dataloader))
    train_features, train_labels = next(dataloader_iterm)
    
    img = train_features[0].squeeze().permute(1,2,0)
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    ###




main()

