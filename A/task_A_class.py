import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

class SVM:
    def __init__(self, kernel, path):
        # Load data from .npz file.
        data = np.load(path)

        train_images = data['train_images']
        test_images = data['test_images']
        val_images = data['val_images']
        train_labels = data['train_labels']
        test_labels = data['test_labels']
        val_labels = data['val_labels']

        # Reshaping the data to match the syntax of sklearn.
        self.train_labels = train_labels.ravel()
        self.test_labels = test_labels.ravel()
        self.val_labels = val_labels.ravel()
        self.train_images = train_images.reshape((train_images.shape[0], train_images[0].size))
        self.test_images = test_images.reshape((test_images.shape[0], test_images[0].size))
        self.val_images = val_images.reshape((val_images.shape[0], val_images[0].size))

        # Define the kernel used for the SVM.
        self.kernel = kernel

    def train_model(self):
        self.__hyperparameters()
        self.__cross_validation()

        return self.best_parameter, self.best_score, self.cv_score
    
    def predict_model(self):
        self.__prediction()
        self.__evaluation()
        return self.pred_score
    
    def __hyperparameters(self):
        # Setting range for hyperparameters.
        C_range = np.logspace(-3, 3, 7)
        gamma_range = np.logspace(-7, -5, 5)
        d_range = [2, 3, 4]
        r_range = np.logspace(-3, 3, 7)
        
        # Performing gridsearch for best parameters, kernel defaults to be linear
        if self.kernel == 'rbf':
            grid_parameter = {'C' : C_range, 'gamma' : gamma_range, 'random_state': [42]}
        elif self.kernel == 'poly':
            grid_parameter = {'C' : C_range, 'degree' : d_range, 'coef0': r_range, 'random_state': [42]}
        else:
            grid_parameter = {'C' : C_range, 'random_state': [42]}

        grid = GridSearchCV(SVC(kernel = self.kernel), param_grid = grid_parameter, cv = 3, scoring = 'accuracy', n_jobs = -1)
        grid.fit(self.val_images, self.val_labels)

        self.best_parameter = grid.best_params_
        self.best_score = grid.best_score_

    def __cross_validation(self):
        # Initiate a SVM model with the best parameter from gridsearch.
        cv_model = SVC(kernel = self.kernel, **self.best_parameter)

        # Setup parameters for cross validation. 
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        self.cv_score = cross_val_score(cv_model, self.train_images, self.train_labels, cv = kf)

    def __prediction(self):
        # Setup a SVM model with the best kernel parameter from gridsearch.
        clf = SVC(kernel = self.kernel, **self.best_parameter)
        clf.fit(self.train_images, self.train_labels)
        self.pred_labels = clf.predict(self.test_images)

    def __evaluation(self):
        # Evaluate the performance of the model.
        self.pred_score = metrics.accuracy_score(self.test_labels, self.pred_labels)

def main():
    linear_SVM = SVM('rbf', './Datasets/pneumoniamnist.npz')
    best_parameter, best_score, cv_score = linear_SVM.train_model()
    pred_score = linear_SVM.predict_model()

    print('Best hyperparameter and its accuracy score: ', best_parameter, best_score)
    print('Cross validation accuracy score: ', cv_score)
    print('Prediction accuracy score: ', pred_score)

    
if __name__ == "__main__":
    main()
    

