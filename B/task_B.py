import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

train_images = CustomImageDataset('./Datasets/pathmnist.npz', 'train')
test_images = CustomImageDataset('./Datasets/pathmnist.npz', 'test')
val_images = CustomImageDataset('./Datasets/pathmnist.npz', 'val')

dataloader = DataLoader(train_images, batch_size = 32, shuffle = True)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_images:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')