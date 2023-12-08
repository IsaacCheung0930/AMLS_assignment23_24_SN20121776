import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    """
    A class to load custom images for a CNN model.

    Attributes:
        path    : The directory of the file
        subset  : The type of data (train/ test/ validation)
    """
    def __init__(self, path, subset = 'train'):
        """
        Load and reshape the images and labels as Pytorch tensors. 

        Parameters:
            path    : The directory of the file
            subset  : The type of data (train/ test/ validation)
        """
        # Load the data from the .npz file
        data = np.load(path)

        # Select the subset of the data
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
        """
        Hidden method for the Pytorch DataLoader function.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Hidden method for the Pytorch DataLoader function.
        """
        return self.images[index], self.labels[index]
    
class CNN(nn.Module):
    """
    A class for CNN model that inherits from the Pytorch module
    """
    def __init__(self):
        """
        Define the CNN model architecture.
        """
        super().__init__()
        # The CNN is defined to have 3 convolution layers, 3x3 filters and 1 pixel stride
        # Input channel is 3 for R, G, B pixels
        # Intermediate channels are power of 2, from 16 to 32 to 64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride=1, padding=1)
        
        # A ReLU function is used for the activation function
        self.relu = nn.ReLU()

        # Each pooling layer has 2x2 filters and 2 pixels stride
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2 Fully connected layers reduces the intermediate channels to 9 output channels
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        """
        Define the procedures of forward propagation

        Parameter:
            x   : The CNN model

        Return:
            x   : The forward propagated CNN model
        """
        # Pooling layer and activation function are wrapped around each convolution layer
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # The model is flattened before the fully connected layers
        x = x.view(-1, 64 * 3 * 3)   
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class TaskB:
    """
    A class for the CNN model for task B.

    Attributes:
        dir : Directory of the dataset
    """
    def __init__(self, dir):
        """
        Initiate the class instance. Activate Cuda device if possible.

        Parameters:
            dir (string): Directory of the dataset
        """
        # Detect availability of cuda device
        if(torch.cuda.is_available()):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        print(f"Using {self._device} device")

        # Set random seed to 42 for reproducible results
        torch.manual_seed(42)
        np.random.seed(42)

        # Load train, test and val images using the CustomImageDataset class
        self._train_images = CustomImageDataset(dir, 'train')
        self._test_images = CustomImageDataset(dir, 'test')
        self._val_images = CustomImageDataset(dir, 'val')

    def execution(self, load=True, overwrite=False):
        """
        Perform model training, validation and prediction using the loaded dataset. 

        Parameters:
            load      : Load model from save if 'True'
            overwrite : Overwrite existing save if 'True'
        """
        model = CNN().to(self._device)
        self._load = load

        if self._load == True:
            model.load_state_dict(torch.load("./B/CNN_model.pth"))
            print("CNN model loaded from './B/CNN_model.pth'")

        else:
            # Separate the train and validation images to batches using the DataLoader funcion
            train_dataloader = DataLoader(self._train_images, batch_size = 32, shuffle = True)
            val_dataloader = DataLoader(self._val_images, batch_size = 32, shuffle = True)

            # Define model parameters
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scheduler = lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = 0.9)

            # Define variables and maximum number of epochs
            self.epoch = range(1, 31)
            self._train_loss, self._val_loss, self._val_correct = [], [], []
            best_state, best_epoch, best_val_loss, best_accuracy = {}, 1, float('inf'), 0

            # Begin model training
            for i in self.epoch:
                print(f"Epoch {i}\n-------------------------------")
                # Model training
                avg_train_loss = self._train(train_dataloader, model, criterion, optimizer)
                # Model valiation
                avg_val_loss, avg_val_correct = self._validation(val_dataloader, model, criterion)
                # Adjust learning rate
                scheduler.step()

                # Record training, validation loss and validation accuracy for analysis
                self._train_loss.append(avg_train_loss)
                self._val_loss.append(avg_val_loss)
                self._val_correct.append(avg_val_correct)

                # Save the state of the model with the best validation performance
                if (avg_val_loss < best_val_loss) & (avg_val_correct > best_accuracy):
                    best_val_loss = avg_val_loss
                    best_val_accuracy = avg_val_correct
                    best_epoch = i
                    # Generate the metrics for the best model state
                    self.best_metrics = {"epoch": best_epoch,
                                         "train_loss": format(avg_train_loss, ".3f"),
                                         "val_loss": format(best_val_loss, ".3f"),
                                         "val_acc": format(best_val_accuracy*100, ".1f")}
                    best_state = model.state_dict()

            # Load the state of the model with the best valiation performance
            model.load_state_dict(best_state)
            print(f'Finished Training, best epoch: {best_epoch}')

            # Overwrite the saved model if overwrite is 'True'
            if overwrite == True:
                torch.save(model.state_dict(), "./B/CNN_model.pth")
                print("Model saved")

        # Predict the classes of the test images using the best model
        self._pred_values, self._true_values = self._prediction(model, self._test_images)

    def evaluation(self):
        """
        Evaluate the performance of the model. 
        Output loss-accuracy plots, confusion matrix and metrics plots.
        """

        # If an old model is loaded, no training data is available hence no loss-accuracy plot
        if self._load == False:
            self._loss_accuracy_plots(self.epoch, self._train_loss, self._val_loss, self._val_correct)
        self._confusion_matrix_plot(self._true_values, self._pred_values)
        self._metrics_plots(self._true_values, self._pred_values)

    def _train(self, dataloader, model, criterion, optimizer):
        """
        Train the CNN model with the model parameters provided.

        Parameters:
            dataloader  : Batches of training data generated by the Pytorch DataLoader
            model       : The class instance that inherits from the Pytorch module
            criterion   : The loss function of the model
            optimizer   : The learning rate and momentum of the model
        
        Return:
            avg_loss    : The average training loss of a batch of training data
        """
        # Set the model to training mode
        model.train()

        batch = len(dataloader)
        size = len(dataloader.dataset)
        total_loss = 0

        # Iterate through the batch of training data
        for i, data in enumerate(dataloader):
            images, labels = data
            images, labels = images.to(self._device), labels.to(self._device)

            # Forward propagation to find the training loss
            pred = model(images)
            loss = criterion(pred, labels)

            # Backward propagation to perform gradient descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # Output the training loss
            if i % 1000 == 0:   
                current = (i + 1) * len(images)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            
        avg_loss = total_loss/ batch
        
        return avg_loss

    def _validation(self, dataloader, model, criterion):
        """
        Validate the CNN model using the validation set.

        Parameters:
            dataloader  : Batches of training data generated by the Pytorch DataLoader
            model       : The CNN model that inherits from the Pytorch module
            criterion   : The loss function of the model
        
        Returns:
            avg_loss    : The average validation loss per batch
            avg_accuracy: The average validation accuracy
        """
        # Set the model to evaluation mode
        model.eval()

        batch, size = len(dataloader), len(dataloader.dataset)
        test_loss, correct = 0, 0

        # Disable gradient descent during validation
        with torch.no_grad():
            # Iterate through all images and labels in a batch of validation data
            for images, labels in dataloader:
                images, labels = images.to(self._device), labels.to(self._device)

                # Predict the labels using the model and calculate the loss
                outputs = model(images)
                _, pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Count the number of correctly classified images
                correct += (pred == labels).sum().item()


        # Calculate and print the average validation accuracy and loss
        avg_loss = test_loss/ batch
        avg_accuracy = correct/ size

        print(f"Validation: \nAccuracy: {(100*avg_accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

        return avg_loss, avg_accuracy

    def _prediction(self, model, test_images):
        """
        Predict the labels of given test images using the trained CNN model.

        Parameters:
            model       : The CNN model that inherits from the Pytorch module
            test_images : Test images loaded from the dataset
        
        Returns:
            pred_values : The predicted labels 
            true_values : The true labels
        """
        correct, total = 0, 0
        pred_values, true_values = [], []

        # Disable gradient descent during prediction
        with torch.no_grad():
            # Iterate through all images in the dataset
            for data in test_images:
                images, labels = data
                images, labels = images.to(self._device), labels.to(self._device)

                # Predict the labels
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Record the predicted and true labels for analysis
                pred_values.append(predicted.item())
                true_values.append(labels.item())

        # Calculate and print the prediction accuracy
        self.pred_accuracy = 100* correct // total
        print(f'Accuracy of the network on the 10000 test images: {self.pred_accuracy} %')
        
        return pred_values, true_values

    def _loss_accuracy_plots(self, epoch, train_loss, val_loss, val_correct):
        """
        Plot the training and validation related plots

        Parameters:
            epoch       : The maximum epoch 
            train_loss  : The average training loss of a batch of training data
            val_loss    : The average validation loss per batch
            val_correct : The average validation accuracy
        """
        # Plot the training and validation loss against epoch
        plt.figure()
        plt.plot(epoch, train_loss, label = "Training Loss", marker="*")
        plt.plot(epoch, val_loss, label = "Validation Loss", marker="*")
        plt.title("Training and Validation Loss Against Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.savefig("./B/Plots/Training and Validation Loss Against Epoch.PNG")

        # Plot the validation accuracy against epoch
        plt.figure()
        plt.plot(epoch, val_correct, label = "Validation Accuracy", marker="*")
        plt.title("Validation Accuracy Against Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.legend()
        plt.savefig("./B/Plots/Validation Accuracy Against Epoch.PNG")

    def _confusion_matrix_plot(self, true_values, pred_values):
        """
        Plot the confusion matrix using the sklearn library.

        Parameters:
            true_values : The true labels
            pred_values : The predicted labels 
        """
        # Plot the confusion matrix
        conf_matrix = ConfusionMatrixDisplay(confusion_matrix(true_values, pred_values))
        conf_matrix.plot(cmap= "plasma")
        conf_matrix.figure_.savefig("./B/Plots/Confusion Matrix.PNG")

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
        plt.grid()
        plt.title("Performace Metrics for 9 Classes")
        plt.xticks(ticks + 0.2, classes)
        plt.legend()
        plt.savefig("./B/Plots/Performance Metrics.PNG")

def main():
    """
    Use this function if the script is run from this file.
    """

    load = False
    CNN_model = TaskB("./Datasets/pathmnist.npz")
    CNN_model.execution(load=load, overwrite=False)
    CNN_model.evaluation()
    if load == False:
        print(f"Total number of epoch: \n{CNN_model.epoch[-1]}")
        print(f"Best epoch (lowest validation loss & highest accuracy): \n{CNN_model.best_metrics['epoch']}")
        print(f"Training loss and validation loss: \n{CNN_model.best_metrics['train_loss']}, {CNN_model.best_metrics['val_loss']}")
        print(f"Validation accuracy: \n{CNN_model.best_metrics['val_acc']}%")
    print(f"Prediction accuracy: \n{CNN_model.pred_accuracy}%")
    
if __name__ == "__main__":
    main()
    