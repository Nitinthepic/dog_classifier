import exiftool
import torch.nn.functional as F
from os.path import dirname, join as pjoin
import os
import torch.nn as nn
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import intro_pytorch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import student_code
# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

class CustomImageDataset(Dataset):
    def __init__(self, dog_list, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.dog_list = dog_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dog_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.dog_list[idx][0])
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
breed_dict = {'silky_terrier': 0,
 'Scottish_deerhound': 1,
 'Chesapeake_Bay_retriever': 2,
 'Ibizan_hound': 3,
 'wire': 4,
 'Saluki': 5,
 'cocker_spaniel': 6,
 'schipperke': 7,
 'borzoi': 8,
 'Pembroke': 9,
 'komondor': 10,
 'Staffordshire_bullterrier': 11,
 'standard_poodle': 12,
 'Eskimo_dog': 13,
 'English_foxhound': 14,
 'golden_retriever': 15,
 'Sealyham_terrier': 16,
 'Japanese_spaniel': 17,
 'miniature_schnauzer': 18,
 'malamute': 19,
 'malinois': 20,
 'Pekinese': 21,
 'giant_schnauzer': 22,
 'Mexican_hairless': 23,
 'Doberman': 24,
 'standard_schnauzer': 25,
 'dhole': 26,
 'German_shepherd': 27,
 'Bouvier_des_Flandres': 28,
 'Siberian_husky': 29,
 'Norwich_terrier': 30,
 'Irish_terrier': 31,
 'Norfolk_terrier': 32,
 'Saint_Bernard': 33,
 'Border_terrier': 34,
 'briard': 35,
 'Tibetan_mastiff': 36,
 'bull_mastiff': 37,
 'Maltese_dog': 38,
 'Kerry_blue_terrier': 39,
 'kuvasz': 40,
 'Greater_Swiss_Mountain_dog': 41,
 'Lakeland_terrier': 42,
 'Blenheim_spaniel': 43,
 'basset': 44,
 'West_Highland_white_terrier': 45,
 'Chihuahua': 46,
 'Border_collie': 47,
 'redbone': 48,
 'Irish_wolfhound': 49,
 'bluetick': 50,
 'miniature_poodle': 51,
 'Cardigan': 52,
 'EntleBucher': 53,
 'Norwegian_elkhound': 54,
 'German_short': 55,
 'Bernese_mountain_dog': 56,
 'papillon': 57,
 'Tibetan_terrier': 58,
 'Gordon_setter': 59,
 'American_Staffordshire_terrier': 60,
 'vizsla': 61,
 'kelpie': 62,
 'Weimaraner': 63,
 'miniature_pinscher': 64,
 'boxer': 65,
 'chow': 66,
 'Old_English_sheepdog': 67,
 'pug': 68,
 'Rhodesian_ridgeback': 69,
 'Scotch_terrier': 70,
 'Shih': 71,
 'affenpinscher': 72,
 'whippet': 73,
 'Sussex_spaniel': 74,
 'otterhound': 75,
 'flat': 76,
 'English_setter': 77,
 'Italian_greyhound': 78,
 'Labrador_retriever': 79,
 'collie': 80,
 'cairn': 81,
 'Rottweiler': 82,
 'Australian_terrier': 83,
 'toy_terrier': 84,
 'Shetland_sheepdog': 85,
 'African_hunting_dog': 86,
 'Newfoundland': 87,
 'Walker_hound': 88,
 'Lhasa': 89,
 'beagle': 90,
 'Samoyed': 91,
 'Great_Dane': 92,
 'Airedale': 93,
 'bloodhound': 94,
 'Irish_setter': 95,
 'keeshond': 96,
 'Dandie_Dinmont': 97,
 'basenji': 98,
 'Bedlington_terrier': 99,
 'Appenzeller': 100,
 'clumber': 101,
 'toy_poodle': 102,
 'Great_Pyrenees': 103,
 'English_springer': 104,
 'Afghan_hound': 105,
 'Brittany_spaniel': 106,
 'Welsh_springer_spaniel': 107,
 'Boston_bull': 108,
 'dingo': 109,
 'soft': 110,
 'curly': 111,
 'French_bulldog': 112,
 'Irish_water_spaniel': 113,
 'Pomeranian': 114,
 'Brabancon_griffon': 115,
 'Yorkshire_terrier': 116,
 'groenendael': 117,
 'Leonberg': 118,
 'black': 119}

def path_label_creator(img_dir):
    path_label_list = list()
    et = exiftool.ExifToolHelper()
    for element in os.listdir(img_dir):
        if(element.startswith("n0")):
            for img_id in os.listdir(os.path.join(img_dir, element)):
                path = os.path.join(element, img_id)
                metadata = et.get_tags(os.path.join(img_dir,path),tags=["ImageWidth", "ImageHeight"])
                try:
                    if metadata[0]['File:ImageWidth'] <= max_Width and metadata[0]['File:ImageHeight'] <= max_Height:
                        label = element.split('-')[1]
                        label = breed_dict[label]
                        path_label_list.append((path, label))
                except KeyError:
                    print(f"{path} is missing certain label")
    return path_label_list



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, train_loader, optimizer, criterion, epochs):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    loss = 0
    epoch_cout = 0
    for epoch in range(epochs):
        for img, label in tqdm(train_loader, total=len(train_loader)):
            # print(img)
            # print(label)
            img = img.float()
            optimizer.zero_grad()
            output = model(img)
            batch_loss = criterion(output, label)
            batch_loss.backward()
            optimizer.step()
            loss +=batch_loss.item()
        train_loss = loss/32
        print('Training loss for epoch {} is {:.4f}'.format(epoch, train_loss))
        print('Validation for epoch {}'.format(epoch))
        torch.save(model.state_dict(), f"epoch_{epoch_cout}_"+'alexnet_finetuning.pth')
        print("Finished Saving!!!")
        epoch_cout+=1

    return train_loss

def main():
    valid = path_label_creator(image_path)
    transformer = transforms.Resize((max_Width,max_Height))
    dog = CustomImageDataset(valid,img_dir = image_path, transform=transformer)
    train_loader = DataLoader(dog, batch_size = 32, shuffle=True)
    test_loader = DataLoader(dog, batch_size = 32, shuffle=True)

    model = models.alexnet(pretrained=False)
    
    model.classifier[6] = nn.Linear(4096, 120)
    optimizer = SGD(model.classifier.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0005) 
    scheduler = ReduceLROnPlateau(optimizer, patience = 4, factor = 0.1, mode = 'min')
    criterion = nn.CrossEntropyLoss()   

    # print("dog")
    train_model(model,train_loader,optimizer,criterion,5)
    # dataloader_iterm = (iter(train_dataloader))
    # train_features, train_labels = next(dataloader_iterm)
    
    # img = train_features[0].squeeze().permute(1,2,0)
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
    # 




main()

