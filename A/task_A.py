import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics

class PNEUMONIA:
    def __init__(self, data):
        self.train_images = data['train_images']
        self.test_images = data['train_images']
        self.val_images = data['val_images']

        self.train_labels = data['train_labels']
        self.test_labels = data['train_labels']
        self.val_labels = data['val_labels']


    def model_accuracy(self):
        return metrics.accuracy_score(self.test_labels, self.pred_labels)
    
    def __image_flattening(self):
        self.train_images_flattened = self.train_images.flatten()
        self.test_images_flattened = self.test_images.flatten()
        self.val_images_flattened = self.val_images.flatten()
        self.train_labels_flattened = self.train_labels.flatten()
        self.test_labels_flattened = self.train_labels.flatten()
        self.val_labels_flattened = self.val_labels.flatten()

    def support_vector_machines(self):
        self.__image_flattening()
        clf = svm.SVC(kernel = 'linear')
        clf.fit(self.train_images, self.test_labels)
        self.pred_labels = clf.predict(self.test_images)
    
    def print_images(self):
        print(self.train_images[0])

data = np.load('AMLS_assignment23_24/Datasets/pneumoniamnist.npz')
pneumonia = PNEUMONIA(data)
pneumonia.support_vector_machines()
print("Accuracy:", pneumonia.model_accuracy())