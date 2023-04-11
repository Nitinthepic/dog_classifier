
from os.path import dirname, join
import argparse
from sklearn.model_selection import train_test_split
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
from sklearn.decomposition import PCA
import torchvision.models as models
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd

def arg_creator():
    parser = argparse.ArgumentParser("Used for configuring operating params")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--predict_mode",action='store_true')
    parser.add_argument("--predict_img_path",type=str)
    parser.add_argument("--load_checkpoint",type=str,default=None)
    parser.add_argument("--store_output",type=str,default=None)
    return parser.parse_args()

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

ANNOT_PATH = os.path.join('data','Annotation')
IMAGE_PATH = os.path.join('data','Images')
max_Width = 227
max_Height = 227
NUM_CLASSES = 120
device = torch.device('cpu')
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

reverse_breed_dict = dict()

def path_label_creator(img_dir):
    path_label_list = list()
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


def train_model(model, train_loader, optimizer, criterion, epoch):
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
    train_loss = loss/len(train_loader)
    print('Training loss for epoch {} is {:.4f}'.format(epoch, train_loss))

    return train_loss

def eval_model(model, test_loader, criterion, epoch):
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
            loss += batch_loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

        val_loss = loss/len(test_loader)    
        test_acc = correct / len(test_loader.dataset)

        print('[Validation Set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch, 100. * test_acc))

    return (test_acc, val_loss)

def test_img(model, dataset):
    model.eval()
    for img, label in tqdm(dataset, total=len(dataset)):
            img = img.float()
            img, label = img.to(device, dtype = torch.float), label.to(device, dtype = torch.long)
            output = model(img)
            pred = output.max(1, keepdim=True)[1]
            return pred


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_filters = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x  
    

def predict_img(args, model, transformer):
    valid = [(args.predict_img_path,9-10)]
    dog = CustomImageDataset(valid, transform=transformer, image_path=IMAGE_PATH, pca_enabled=False)
        

    predict_test = DataLoader(dog, batch_size = 128, shuffle=True, num_workers=2)
        
    print(reverse_breed_dict[(test_img(model,predict_test)).item()])

def train_val_loop(args,model,transformer):
    image_pathlist = path_label_creator(IMAGE_PATH)
    image_train, image_val = train_test_split(image_pathlist,
                                                train_size=args.train_size,
                                                shuffle=True)

    train_ds = CustomImageDataset(image_train,
                                transform=transformer, 
                                image_path=IMAGE_PATH, 
                                pca_enabled=False)
    
                
    val_ds = CustomImageDataset(image_val,
                                transform=transformer, 
                                image_path=IMAGE_PATH, 
                                pca_enabled=False)
        
    train_loader = DataLoader(train_ds,
                            batch_size = args.batch_size,
                            shuffle=True,
                            num_workers=2)
        
    validation_loader = DataLoader(val_ds,
                                batch_size = args.batch_size,
                                shuffle=True,
                                num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    val_loss = np.inf

    if args.store_output is not None:
        df = pd.DataFrame()

    for epoch in range(1,args.epochs+1):
        train_model(model,train_loader,optimizer,criterion,epoch)
        acc, val_loss = eval_model(model,validation_loader,criterion,epoch)   
        if val_loss <= best_loss:
            print('Validation loss has reduced from {:.4f} to {:.4f}'.format(best_loss, val_loss))
            print('Saving model')
            best_loss = val_loss
            torch.save(model.state_dict(),
            os.path.join("checkpoint_folder",f"epoch_{epoch}_"+'finetuning.pth'))
            print("Finished Saving!!!")  
        if args.store_output is not None:
            df = df.append({'Epoch': epoch, 'Accuracy': acc, 'Validation Loss': val_loss})
    if args.store_output is not None:
       df.to_csv(args.store_output+'.csv')

def main():
    args = arg_creator()
    global reverse_breed_dict
    reverse_breed_dict = {v: k for k, v in breed_dict.items()}
    global device
    device = torch.device(args.device)    
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    transformer = transforms.Resize((max_Width,max_Height),antialias=True)
    model = ResNet50(NUM_CLASSES)
    model = model.to(device)

    if args.load_checkpoint is not None:
            model.load_state_dict(torch.load(args.load_checkpoint,map_location=device))

    if args.predict_mode:
        predict_img(args,model,transformer)
    else:
        train_val_loop(args,model,transformer)
     



if __name__ == '__main__':
    main()

