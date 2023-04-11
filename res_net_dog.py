import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder

num_classes = 120
batch_size = 32

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_filters = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x

def train_model(model, optimizer, criterion, train_loader, val_loader):
    num_epochs = 10
    x = 0
    for epoch in range(num_epochs):
        # Train for one epoch
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(x)
            x += 1

        # Evaluate on the validation set
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}: val accuracy = {accuracy:.4f}")

def main():
    print("Starting")

    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root='data/Test', transform=transform)
    val_dataset = ImageFolder(root='data/Test', transform=transform)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ResNet50(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, optimizer, criterion, train_loader, val_loader)    


if __name__ == "__main__":
    main()
