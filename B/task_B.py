import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt

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

        x = x.view(-1, 64 * 3 * 3)   # Adjusted for 28x28 input size
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TaskB:
    def __init__(self, dir):
        if(torch.cuda.is_available()):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        print(f"Using {self._device} device")
        torch.manual_seed(42)
        np.random.seed(42)

        self._train_images = CustomImageDataset(dir, 'train')
        self._test_images = CustomImageDataset(dir, 'test')
        self._val_images = CustomImageDataset(dir, 'val')

    def execution(self):
        train_dataloader = DataLoader(self._train_images, batch_size = 32, shuffle = True)
        val_dataloader = DataLoader(self._val_images, batch_size = 32, shuffle = True)

        model = CNN().to(self._device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = 0.9)

        self.epoch = range(1, 11)
        self.train_loss, self.val_loss, self.val_correct = [], [], []

        for i in self.epoch:
            print(f"Epoch {i}\n-------------------------------")
            avg_train_loss = self._train(train_dataloader, model, criterion, optimizer)
            avg_val_loss, avg_val_correct = self._validation(val_dataloader, model, criterion)
            scheduler.step()

            self.train_loss.append(avg_train_loss)
            self.val_loss.append(avg_val_loss)
            self.val_correct.append(avg_val_correct)

        print('Finished Training')
        self._pred_values, self._true_values = self._prediction(model, self._test_images)

    def evaluation(self):
        self._loss_accuracy_plots(self.epoch, self.train_loss, self.val_loss, self.val_correct)
        self._confusion_matrix_plot(self._true_values, self._pred_values)
        self._metrics_plots(self._true_values, self._pred_values)

    def _train(self, dataloader, model, criterion, optimizer):
        model.train()
        batch = len(dataloader)
        size = len(dataloader.dataset)
        total_loss = 0

        for i, data in enumerate(dataloader, 0):
            images, labels = data
            images, labels = images.to(self._device), labels.to(self._device)
            pred = model(images)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if i % 1000 == 0:   
                current = (i + 1) * len(images)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            
        avg_loss = total_loss/ batch
        
        return avg_loss

    def _validation(self, dataloader, model, criterion):
        model.eval()
        batch = len(dataloader)
        size = len(dataloader.dataset)

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self._device), labels.to(self._device)
                pred = model(images)
                loss = criterion(pred, labels)
                test_loss += loss.item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        avg_loss = test_loss/ batch
        avg_correct = correct/ size
        print(f"Test Error: \nAccuracy: {(100*avg_correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

        return avg_loss, avg_correct

    def _prediction(self, model, test_images):
        correct = 0
        total = 0
        pred_values = []
        true_values = []

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_images:
                images, labels = data
                images, labels = images.to(self._device), labels.to(self._device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pred_values.append(predicted.item())
                true_values.append(labels.item())

        self.pred_accuracy = 100* correct // total
        print(f'Accuracy of the network on the 10000 test images: {self.pred_accuracy} %')
        
        return pred_values, true_values

    def _loss_accuracy_plots(self, epoch, train_loss, val_loss, val_correct):
        plt.figure()
        plt.plot(epoch, train_loss, label = "Training Loss")
        plt.plot(epoch, val_loss, label = "Validation Loss")
        plt.title("Training and Validation Loss Against Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("./B/Training and Validation Loss Against Epoch.PNG")

        plt.figure()
        plt.plot(epoch, val_correct, label = "Validation Accuracy")
        plt.title("Validation Accuracy Against Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("./B/Validation Accuracy Against Epoch.PNG")

    def _confusion_matrix_plot(self, true_values, pred_values):
        conf_matrix = ConfusionMatrixDisplay(confusion_matrix(true_values, pred_values))
        conf_matrix.plot(cmap= "plasma")
        conf_matrix.figure_.savefig("./B/Confusion Matrix.PNG")

    def _metrics_plots(self, true_values, pred_values):
        micro_precision, micro_recall, micro_f1score, _ = precision_recall_fscore_support(true_values, pred_values, average='micro')
        macro_precision, macro_recall, macro_f1score, _ = precision_recall_fscore_support(true_values, pred_values, average='macro')
        class_precision, class_recall, class_f1score, _ = precision_recall_fscore_support(true_values, pred_values, average=None)
        
        precision = np.append(class_precision, np.append(micro_precision, macro_precision))
        recall = np.append(class_recall, np.append(micro_recall, macro_recall))
        f1score = np.append(class_f1score, np.append(micro_f1score, macro_f1score))

        classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "Micro", "Macro"]
        ticks = np.arange(len(classes))
        plt.figure()
        plt.bar(ticks, precision, 0.2, label = "Precision")
        plt.bar(ticks + 0.2, recall, 0.2, label = "Recall")
        plt.bar(ticks + 0.4, f1score, 0.2, label = "F1")
        plt.xlabel("Classes")
        plt.ylabel("Score")
        plt.title("Performace Metrics for 9 Classes")
        plt.xticks(ticks + 0.2, classes)
        plt.legend()
        plt.savefig("./B/Performance Metrics.PNG")

def main():
    CNN_model = TaskB("./Datasets/pathmnist.npz")
    CNN_model.execution()
    CNN_model.evaluation()
    #epoch, train_loss, val_loss, val_correct, pred_accuracy

if __name__ == "__main__":
    main()
    