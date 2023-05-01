import argparse
import random
import os

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import models, transforms

from tqdm import tqdm

import numpy as np

import pandas as pd


def arg_creator():
    """
    Uses argument parser to get environment variables
    """
    parser = argparse.ArgumentParser(
        prog="Dog Classifier",
        description="Used for configuring operating params for either training the model or having it classify a dog",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Picks the device to train the model, defaults to cpu, can use cuda",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="sets the batch size, defaults to 32"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to run training, defaults to 32",
    )
    parser.add_argument(
        "--max_width", type=int, default=227, help="width (in pixels) of the picture"
    )
    parser.add_argument(
        "--max_height", type=int, default=227, help="height (in pixels) of the picture"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="number of workers for dataloading"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="the proportion of the dataset to be used for training",
    )
    parser.add_argument(
        "--predict_mode",
        action="store_true",
        help="when set, model does not train, but instead will predict the breed of a given image, if set, must also set predict_img_path",
    )
    parser.add_argument(
        "--predict_img_path",
        type=str,
        help="location of the image for the breed will be predicted, must be inside data/Images",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="filename/location of a pth checkpoint file to be loaded",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default=os.path.join("data", "Images"),
        help="filename/location of the images for the dataset",
    )
    parser.add_argument(
        "--store_output",
        type=str,
        default=None,
        help="if set, this flag's input will be the filename for a csv with accuracy, loss, and epoch",
    )
    return parser.parse_args()


class CustomImageDataset(Dataset):
    def __init__(
        self,
        dog_list,
        image_path,
        transform=None,
        target_transform=None,
        pca_enabled=True,
    ):
        self.dog_list = dog_list
        self.transform = transform
        self.source_data_path = image_path
        self.image_path = image_path if not pca_enabled else f"{image_path}-pca"
        self.target_transform = target_transform
        self.max_image_size = 1250  # Max image size is 1250x1250
        self.pca_enabled = pca_enabled

    def __len__(self):
        return len(self.dog_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.dog_list[idx][0])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.dog_list[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


args = arg_creator()
IMAGE_PATH = args.img_path
MAX_WIDTH = args.max_width
MAX_HEIGHT = args.max_height
NUM_CLASSES = 120
device = torch.device(args.device)
breed_dict = {
    "silky_terrier": 0,
    "Scottish_deerhound": 1,
    "Chesapeake_Bay_retriever": 2,
    "Ibizan_hound": 3,
    "wire": 4,
    "Saluki": 5,
    "cocker_spaniel": 6,
    "schipperke": 7,
    "borzoi": 8,
    "Pembroke": 9,
    "komondor": 10,
    "Staffordshire_bullterrier": 11,
    "standard_poodle": 12,
    "Eskimo_dog": 13,
    "English_foxhound": 14,
    "golden_retriever": 15,
    "Sealyham_terrier": 16,
    "Japanese_spaniel": 17,
    "miniature_schnauzer": 18,
    "malamute": 19,
    "malinois": 20,
    "Pekinese": 21,
    "giant_schnauzer": 22,
    "Mexican_hairless": 23,
    "Doberman": 24,
    "standard_schnauzer": 25,
    "dhole": 26,
    "German_shepherd": 27,
    "Bouvier_des_Flandres": 28,
    "Siberian_husky": 29,
    "Norwich_terrier": 30,
    "Irish_terrier": 31,
    "Norfolk_terrier": 32,
    "Saint_Bernard": 33,
    "Border_terrier": 34,
    "briard": 35,
    "Tibetan_mastiff": 36,
    "bull_mastiff": 37,
    "Maltese_dog": 38,
    "Kerry_blue_terrier": 39,
    "kuvasz": 40,
    "Greater_Swiss_Mountain_dog": 41,
    "Lakeland_terrier": 42,
    "Blenheim_spaniel": 43,
    "basset": 44,
    "West_Highland_white_terrier": 45,
    "Chihuahua": 46,
    "Border_collie": 47,
    "redbone": 48,
    "Irish_wolfhound": 49,
    "bluetick": 50,
    "miniature_poodle": 51,
    "Cardigan": 52,
    "EntleBucher": 53,
    "Norwegian_elkhound": 54,
    "German_short": 55,
    "Bernese_mountain_dog": 56,
    "papillon": 57,
    "Tibetan_terrier": 58,
    "Gordon_setter": 59,
    "American_Staffordshire_terrier": 60,
    "vizsla": 61,
    "kelpie": 62,
    "Weimaraner": 63,
    "miniature_pinscher": 64,
    "boxer": 65,
    "chow": 66,
    "Old_English_sheepdog": 67,
    "pug": 68,
    "Rhodesian_ridgeback": 69,
    "Scotch_terrier": 70,
    "Shih": 71,
    "affenpinscher": 72,
    "whippet": 73,
    "Sussex_spaniel": 74,
    "otterhound": 75,
    "flat": 76,
    "English_setter": 77,
    "Italian_greyhound": 78,
    "Labrador_retriever": 79,
    "collie": 80,
    "cairn": 81,
    "Rottweiler": 82,
    "Australian_terrier": 83,
    "toy_terrier": 84,
    "Shetland_sheepdog": 85,
    "African_hunting_dog": 86,
    "Newfoundland": 87,
    "Walker_hound": 88,
    "Lhasa": 89,
    "beagle": 90,
    "Samoyed": 91,
    "Great_Dane": 92,
    "Airedale": 93,
    "bloodhound": 94,
    "Irish_setter": 95,
    "keeshond": 96,
    "Dandie_Dinmont": 97,
    "basenji": 98,
    "Bedlington_terrier": 99,
    "Appenzeller": 100,
    "clumber": 101,
    "toy_poodle": 102,
    "Great_Pyrenees": 103,
    "English_springer": 104,
    "Afghan_hound": 105,
    "Brittany_spaniel": 106,
    "Welsh_springer_spaniel": 107,
    "Boston_bull": 108,
    "dingo": 109,
    "soft": 110,
    "curly": 111,
    "French_bulldog": 112,
    "Irish_water_spaniel": 113,
    "Pomeranian": 114,
    "Brabancon_griffon": 115,
    "Yorkshire_terrier": 116,
    "groenendael": 117,
    "Leonberg": 118,
    "black": 119,
}
reverse_breed_dict = {v: k for k, v in breed_dict.items()}


def path_label_creator(img_dir):
    """
    Given the directory where the images are stored, returns a list of all the images' individual paths and their label; images should be in their own folder, and folder name should start with n0, folder name becomes the label

    Parameters:
    img_dir (str): where the images are stored

    Returns:
    list: a list of all the images indivdual paths and their labels, as a tuple

    """
    path_label_list = []
    for element in os.listdir(img_dir):
        if element.startswith("n0"):
            for img_id in os.listdir(os.path.join(img_dir, element)):
                path = os.path.join(element, img_id)
                try:
                    label = element.split("-")[1]
                    label = breed_dict[label]
                    path_label_list.append((path, label))
                except KeyError:
                    print(f"{path} is missing certain label")
    return path_label_list


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    Given a model and a pytorch DataLoader, trains the model
    Parameters:
    model (torch.nn.module): The model to be trained
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): An optimizer, usually Adam
    criterion (nn.CrossEntropyLoss) : Loss function used to train the model
    epoch (int): Current epoch number

    Returns:
    float: the training loss for this epoch
    """
    model.train()
    train_loss = 0.0
    loss = 0
    correct = 0
    for img, label in tqdm(train_loader, total=len(train_loader)):
        img = img.float()
        img, label = img.to(device, dtype=torch.float), label.to(
            device, dtype=torch.long
        )
        optimizer.zero_grad()
        output = model(img)
        batch_loss = criterion(output, label)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

    train_loss = loss / len(train_loader)
    train_acc = 100 * (correct / len(train_loader.dataset))
    print("Training loss for epoch {} is {:.4f}".format(epoch, train_loss))

    return (train_acc, train_loss)


def eval_model(model, test_loader, criterion, epoch):
    """
    Given a model and a pytorch DataLoader, evaluates the model
    Parameters:
    model (torch.nn.module): The model to be trained
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): An optimizer, usually Adam
    epoch (int): Current epoch number

    Returns:
    tuple: the accuracy for this epoch and its validation loss
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        loss = 0
        correct = 0
        for img, label in tqdm(test_loader, total=len(test_loader)):
            img = img.float()
            img, label = img.to(device, dtype=torch.float), label.to(
                device, dtype=torch.long
            )
            output = model(img)
            batch_loss = criterion(output, label)
            loss += batch_loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

        val_loss = loss / len(test_loader)
        test_acc = 100 * (correct / len(test_loader.dataset))

        print(f"[Validation Set] Epoch: {epoch}, Accuracy: {test_acc}")

    return (test_acc, val_loss)


def test_img(model, dataset):
    """
    Given a model and a pytorch DataLoader, uses model to predict the breed of the dog
    Parameters:
    model (torch.nn.module): The model to use for prediction
    train_loader (pytorch data loader): Data Loader with only one image
    Returns:
    tensor: the softmax output of each breed
    """
    model.eval()
    for img, label in tqdm(dataset, total=len(dataset)):
        img = img.float()
        img, label = img.to(device, dtype=torch.float), label.to(
            device, dtype=torch.long
        )
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


def predict_img(model, transformer):
    """
    Given a model and a pytorch transformer, setups up the dataset/loader, uses model to predict the breed of the dog, and converts to string name/label
    Parameters:
    model (torch.nn.module): The model to be used for prediction
    transformer: Transforms the image

    Returns:
    string: the name of the highest probability breed
    """
    valid = [(args.predict_img_path, 9 - 10)]
    dog = CustomImageDataset(
        valid, transform=transformer, image_path=IMAGE_PATH, pca_enabled=False
    )

    predict_test = DataLoader(dog, batch_size=128, shuffle=True, num_workers=2)

    print(reverse_breed_dict[(test_img(model, predict_test)).item()])


def train_val_loop(model, transformer):
    """
    Given a model and a pytorch transformer, splits into train/validate sets,  setups up the dataset/loader, and trains/validates model

    Can save output to csv, and saves epochs where model improves (lower vloss)

    Parameters:
    model (torch.nn.module): The model to be used for prediction
    transformer: Transforms the image
    """
    image_pathlist = path_label_creator(IMAGE_PATH)
    image_train, image_val = train_test_split(
        image_pathlist, train_size=args.train_size, shuffle=True
    )

    train_ds = CustomImageDataset(
        image_train, transform=transformer, image_path=IMAGE_PATH, pca_enabled=False
    )

    val_ds = CustomImageDataset(
        image_val, transform=transformer, image_path=IMAGE_PATH, pca_enabled=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    validation_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = 1e99

    if args.store_output is not None:
        training_df = pd.DataFrame()

    for epoch in range(1, args.epochs + 1):
        train_acc, train_loss = train_model(model, train_loader, optimizer, criterion, epoch)
        acc, val_loss = eval_model(model, validation_loader, criterion, epoch)

        if val_loss <= best_loss:
            print(f"Validation loss has reduced from {best_loss} to {val_loss}")
            print("Saving model")
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join("checkpoint_folder", f"epoch_{epoch}_" + "finetuning.pth"),
            )
            print("Finished Saving!!!")

        if args.store_output is not None:
            new_row = pd.DataFrame(
                [
                    {
                        "Epoch": epoch,
                        "Training Loss": train_loss,
                        "Traing Accuracy": train_acc,
                        "Validation Accuracy": acc,
                        "Validation Loss": val_loss,
                    }
                ]
            )
            training_df = pd.concat([training_df, new_row])

    if args.store_output is not None:
        training_df.to_csv(args.store_output + ".csv")


def main():
    """
    Sets the random seed, establishes cuda params, creates transformer and model, loads checkpoint (if set), either trains model or predicts given image(depends on CLA set)
    """
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    transformer = transforms.Resize((MAX_WIDTH, MAX_HEIGHT), antialias=True)
    model = ResNet50(NUM_CLASSES)
    model = model.to(device)

    if args.load_checkpoint is not None:
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))

    if args.predict_mode:
        predict_img(model, transformer)
    else:
        train_val_loop(model, transformer)


if __name__ == "__main__":
    main()
