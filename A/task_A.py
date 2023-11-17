import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

def load_data():
    # Load data from .npz file.
    data = np.load('./Datasets/pneumoniamnist.npz')

    train_images = data['train_images']
    test_images = data['test_images']
    val_images = data['val_images']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    val_labels = data['val_labels']

    # Reshaping the data to match the syntax of sklearn.
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()
    val_labels = val_labels.ravel()
    train_images = train_images.reshape((train_images.shape[0], train_images[0].size))
    test_images = test_images.reshape((test_images.shape[0], test_images[0].size))
    val_images = val_images.reshape((val_images.shape[0], val_images[0].size))

    return train_images, test_images, val_images, train_labels, test_labels, val_labels

def hyper_parameters(val_images, val_labels, kernel):
    # Setting range for hyperparameters. Default C = 1, gamma = 6.92e-7; Best C = 5.62, best gamma = 1e-6
    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-7, -5, 5)
    d_range = [2, 3, 4]
    r_range = np.logspace(-3, 3, 7)
    
    # Performing gridsearch for best parameters, kernel defaults to be linear
    if kernel == 'rbf':
        grid_parameter = {'C' : C_range, 'gamma' : gamma_range, 'random_state': [42]}
    elif kernel == 'poly':
        grid_parameter = {'C' : C_range, 'degree' : d_range, 'coef0': r_range, 'random_state': [42]}
    else:
        grid_parameter = {'C' : C_range, 'random_state': [42]}

    grid = GridSearchCV(SVC(kernel = kernel), param_grid = grid_parameter, cv = 3, scoring = 'accuracy', n_jobs = -1)
    grid.fit(val_images, val_labels)

    return grid.best_params_, grid.best_score_

def cross_validation(train_images, train_labels, kernel, best_parameter):
    # Initiate a SVM model with the best parameter from gridsearch.
    cv_model = SVC(kernel = kernel, **best_parameter)

    # Setup parameters for cross validation. 
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    cv_score = cross_val_score(cv_model, train_images, train_labels, cv = kf)

    return cv_score

def support_vector_machine(train_images, train_labels, test_images, kernel, best_parameter):
    # Setup a SVM model with the best kernel parameter from gridsearch.
    clf = SVC(kernel = kernel, **best_parameter)
    clf.fit(train_images, train_labels)
    pred_labels = clf.predict(test_images)

    return pred_labels

def performance_evaluation(test_labels, pred_labels):
    # Evaluate the performance of the model.
    pred_score = metrics.accuracy_score(test_labels, pred_labels)

    return pred_score

def main():
    train_images, test_images, val_images, train_labels, test_labels, val_labels = load_data()

    # ONLY call the functions below when hyperparameter tuning is necessary
    best_parameter, best_score = hyper_parameters(val_images, val_labels, 'sigmoid')
    print('RBF kernel; Best hyperparameter and its accuracy score: ', best_parameter, best_score)

    # Perform cross validation
    cv_score = cross_validation(train_images, train_labels, 'sigmoid', best_parameter)
    print('RBF kernel; Cross validation accuracy score: ', cv_score)

    # Prediction and Evaluation
    pred_labels = support_vector_machine(train_images, train_labels, test_images, 'sigmoid', best_parameter)
    pred_score = performance_evaluation(test_labels, pred_labels)
    print('RBF kernel; Prediction accuracy score: ', pred_score)
    


if __name__ == "__main__":
    main()
    

