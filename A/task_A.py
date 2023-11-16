import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

def load_data():
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

    return train_images, test_images, val_images, train_labels, test_labels, val_labels

def rbf_parameters(train_images, train_labels):
    # Default C = 1, gamma = 6.92e-7
    C_range = np.logspace(0, 3, 5)
    gamma_range = np.logspace(-7, -5, 5)
    parameter = {'gamma' : gamma_range, 'C' : C_range}
    grid = GridSearchCV(SVC(), param_grid = parameter, cv = 3, scoring = 'accuracy', n_jobs = -1, verbose = 2)
    grid.fit(train_images, train_labels)

    return grid.best_params_, grid.best_score_


def support_vector_machine(function, train_images, train_labels, test_images):
    clf = SVC(kernel = function)
    clf.fit(train_images, train_labels)
    pred_labels = clf.predict(test_images)
    return pred_labels

def accuracy_score(test_labels, pred_labels):
    score = metrics.accuracy_score(test_labels, pred_labels)

    return score

def main():
    train_images, test_images, val_images, train_labels, test_labels, val_labels = load_data()

    # ONLY call the functions below when hyperparameter tuning is necessary
    #best_parameter, best_score = rbf_parameters(train_images, train_labels)
    #print(best_parameter, best_score)

    pred_labels = support_vector_machine('rbf', train_images, train_labels, test_images)
    print(accuracy_score(test_labels, pred_labels))
    

main()
    

