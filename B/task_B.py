import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, path, subset = 'train'):
        data = np.load(path)

        if subset == 'train':
            self.images = torch.tensor(data['train_images'], dtype = torch.float32).permute(0, 3, 1, 2)
            self.labels = torch.tensor(data['train_labels'], dtype = torch.long).squeeze(1)
        elif subset == 'test':
            self.images = torch.tensor(data['test_images'], dtype = torch.float32).permute(0, 3, 1, 2)
            self.labels = torch.tensor(data['test_labels'], dtype = torch.long)
        elif subset == 'val':
            self.images = torch.tensor(data['val_images'], dtype = torch.float32).permute(0, 3, 1, 2)
            self.labels = torch.tensor(data['val_labels'], dtype = torch.long).squeeze(1)
        else:
            raise ValueError("Use 'train', 'test' or 'val' to specify subsets")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # Adjusted for 28x28 input size
        self.fc2 = nn.Linear(128, 9)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(-1, 64 * 3 * 3)  # Adjusted for 28x28 input size
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    size = len(dataloader.dataset)

    for i, data in enumerate(dataloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 1000 == 0:   
            loss, current = loss.item(), (i + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation(dataloader, model, criterion, device):
    model.eval()
    batch = len(dataloader)
    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            test_loss += loss.item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device")

    train_images = CustomImageDataset('./Datasets/pathmnist.npz', 'train')
    test_images = CustomImageDataset('./Datasets/pathmnist.npz', 'test')
    val_images = CustomImageDataset('./Datasets/pathmnist.npz', 'val')

    train_dataloader = DataLoader(train_images, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(val_images, batch_size = 32, shuffle = True)
    
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = 0.9)

    for i in range(10):
        print(f"Epoch {i+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer, device)
        validation(val_dataloader, model, criterion, device)
        scheduler.step()
    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_images:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == "__main__":
    main()
    