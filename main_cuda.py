
from os.path import dirname, join as pjoin
import os
import torch.nn as nn
from tqdm import tqdm
import torch
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.decomposition import PCA
import torchvision.models as models
import random
# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

class CustomImageDataset(Dataset):
    def __init__(self, dog_list, image_path, transform=None, target_transform=None, pca_enabled=True):
        self.dog_list = dog_list
        self.transform = transform
        self.source_data_path = image_path
        self.image_path = image_path if not pca_enabled else f"{image_path}-pca"
        self.target_transform = target_transform
        self.max_image_size = 1250 #Max image size is 1250x1250
        self.pca_enabled = pca_enabled
        
        #THIS TAKES A LONG TIME.
        #Will generate images that have been PCA'd, make sure you do not have the Images-pca folder in your data folder.
        #Otherwise it will not generate new images to save resources.
        if self.pca_enabled and not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
            for image in glob.glob(f"{self.source_data_path}/**/*.jpg", recursive=True):
                folder = image.split("/")[2]
                file_name = image.split("/")[3]
                with Image.open(image) as im:
                    pixels = np.array(im)
                    r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
                    pca_r = PCA(n_components=0.99)
                    pca_g = PCA(n_components=0.99)
                    pca_b = PCA(n_components=0.99)
                    pca_r_trans = pca_r.fit_transform(r)
                    pca_g_trans = pca_g.fit_transform(g)
                    pca_b_trans = pca_b.fit_transform(b)

                    pca_r_org = pca_r.inverse_transform(pca_r_trans)
                    pca_g_org = pca_g.inverse_transform(pca_g_trans)
                    pca_b_org = pca_b.inverse_transform(pca_b_trans)
                    
                    temp = np.dstack((pca_r_org,pca_g_org,pca_b_org))
                    temp = temp.astype(np.uint8)
                    new_image = Image.fromarray(temp)
                    new_image.convert("RGB")

                    if not os.path.exists(f"{self.image_path}/{folder}"):
                        os.mkdir(f"{self.image_path}/{folder}")

                    new_image.save(f"{self.image_path}/{folder}/{file_name}")
                    


    def __len__(self):
        return len(self.dog_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path,self.dog_list[idx][0])
        image = read_image(img_path,mode=ImageReadMode.RGB)
        label = self.dog_list[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

annot_path = 'data/Annotation'
IMAGE_PATH = 'data/Images'
max_Width = 1250
max_Height = 1250
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

def path_label_creator(img_dir, truncate):
    path_label_list = list()
    if truncate:
        for element in os.listdir(img_dir):
            if(element.startswith("n0")):
                for img_id in os.listdir(os.path.join(img_dir, element)):
                    path = os.path.join(element, img_id)
                    #metadata = et.get_tags(os.path.join(img_dir,path),tags=["ImageWidth", "ImageHeight"])
                    try:
                        #if metadata[0]['File:ImageWidth'] <= max_Width and metadata[0]['File:ImageHeight'] <= max_Height:
                            label = element.split('-')[1]
                            label = breed_dict[label]
                            path_label_list.append((path, label))
                    except KeyError:
                        print(f"{path} is missing certain label")
    else:
        for element in os.listdir(img_dir):
            if(element.startswith("n0")):
                for img_id in os.listdir(os.path.join(img_dir, element)):
                    path = os.path.join(element, img_id)
                    try:
                        label = element.split('-')[1]
                        label = breed_dict[label]
                        path_label_list.append((path, label))
                    except KeyError:
                        print(f"{path} is missing certain label")
    return path_label_list



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, train_loader, optimizer, criterion, epoch, batch_count, device):
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
    for img, label in tqdm(train_loader, total=len(train_loader)):
        img = img.float()
        img, label = img.to(device, dtype = torch.float), label.to(device, dtype = torch.long)
        optimizer.zero_grad()
        output = model(img)
        batch_loss = criterion(output, label)
        batch_loss.backward()
        optimizer.step()
        loss +=batch_loss.item()
    train_loss = loss/batch_count
    print('Training loss for epoch {} is {:.4f}'.format(epoch, train_loss))
    print('Validation for epoch {}'.format(epoch))
    torch.save(model.state_dict(), f"epoch_{epoch}_"+'alexnet_finetuning.pth')
    print("Finished Saving!!!")

    return train_loss

def eval_model(model, test_loader, criterion, epoch, batch_count, device):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        loss = 0
        correct = 0
        for img, label in tqdm(test_loader, total=len(test_loader)):
            img = img.float()
            img, label = img.to(device, dtype = torch.float), label.to(device, dtype = torch.long)
            output = model(img)
            batch_loss = criterion(output, label)
            loss +=batch_loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

        val_loss = loss/batch_count    
        test_acc = correct / len(test_loader.dataset)

        print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch, 100. * test_acc))

    return (test_acc, val_loss)

       

def main():
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    valid = path_label_creator(IMAGE_PATH, False)
    transformer = transforms.Resize((max_Width,max_Height))
    dog = CustomImageDataset(valid, transform=transformer, image_path=IMAGE_PATH, pca_enabled=False)
    train_loader = DataLoader(dog, batch_size = 32, shuffle=True)
    test_loader = DataLoader(dog, batch_size = 32, shuffle=True)

    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 120)
    optimizer = SGD(model.classifier.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0005) 
    scheduler = ReduceLROnPlateau(optimizer, patience = 4, factor = 0.1, mode = 'min')
    criterion = nn.CrossEntropyLoss()   
    device = torch.device('cuda')

    
    model = model.to(device)
    for epoch in range(1,10):
        train_model(model,train_loader,optimizer,criterion,epoch,len(train_loader),device)
        eval_model(model,test_loader,criterion,epoch,len(test_loader),device)
    
     




main()

