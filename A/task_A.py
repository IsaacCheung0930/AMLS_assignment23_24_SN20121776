import numpy as np
from sklearn import svm
from sklearn import metrics


def main():
    data = np.load('./Datasets/pneumoniamnist.npz')

    train_images = data['train_images']
    test_images = data['test_images']
    val_images = data['val_images']

    train_labels = data['train_labels'].ravel()
    test_labels = data['test_labels'].ravel()
    val_labels = data['val_labels'].ravel()
    
    train_images = train_images.reshape((train_images.shape[0], train_images[0].size))
    test_images = test_images.reshape((test_images.shape[0], test_images[0].size))
    val_images = val_images.reshape((val_images.shape[0], val_images[0].size))
    
    clf = svm.SVC(kernel = 'rbf')
    clf.fit(train_images, train_labels)
    pred_labels = clf.predict(test_images)

    print(metrics.accuracy_score(test_labels, pred_labels))
    

main()
    

