import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
        self.__train_labels = train_labels.ravel()
        self.__test_labels = test_labels.ravel()
        self.__val_labels = val_labels.ravel()
        self.__train_images = train_images.reshape((train_images.shape[0], train_images[0].size))
        self.__test_images = test_images.reshape((test_images.shape[0], test_images[0].size))
        self.__val_images = val_images.reshape((val_images.shape[0], val_images[0].size))

        # Define the kernel used for the SVM.
        self.__kernel = kernel

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
        if self.__kernel == 'rbf':
            grid_parameter = {'C' : C_range, 'gamma' : gamma_range, 'random_state': [42]}
        elif self.__kernel == 'poly':
            grid_parameter = {'C' : C_range, 'degree' : d_range, 'coef0': r_range, 'random_state': [42]}
        else:
            grid_parameter = {'C' : C_range, 'random_state': [42]}

        grid = GridSearchCV(SVC(kernel = self.__kernel), param_grid = grid_parameter, cv = 3, scoring = 'accuracy', n_jobs = -1)
        grid.fit(self.__val_images, self.__val_labels)

        self.best_parameter = grid.best_params_
        self.best_score = grid.best_score_

    def __cross_validation(self):
        # Initiate a SVM model with the best parameter from gridsearch.
        cv_model = SVC(kernel = self.__kernel, **self.best_parameter)

        # Setup parameters for cross validation. 
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        self.cv_score = cross_val_score(cv_model, self.__train_images, self.__train_labels, cv = kf)

    def __prediction(self):
        # Setup a SVM model with the best kernel parameter from gridsearch.
        clf = SVC(kernel = self.__kernel, **self.best_parameter)
        clf.fit(self.__train_images, self.__train_labels)
        self.__pred_labels = clf.predict(self.__test_images)

    def __evaluation(self):
        # Evaluate the performance of the model.
        self.pred_score = metrics.accuracy_score(self.__test_labels, self.__pred_labels)


def main():
    tree = TREE('./Datasets/pneumoniamnist.npz')
    #best_parameter, best_score, cv_score = tree.train_model()
    pred_score = tree.predict_model()

    #print('Best hyperparameter and its accuracy score: ', best_parameter, best_score)
    #print('Cross validation accuracy score: ', cv_score)
    print('Prediction accuracy score: ', pred_score)

    
if __name__ == "__main__":
    main()
    

class TREE():
    def __init__(self, path):
        # Load data from .npz file.
        data = np.load(path)

        train_images = data['train_images']
        test_images = data['test_images']
        val_images = data['val_images']
        train_labels = data['train_labels']
        test_labels = data['test_labels']
        val_labels = data['val_labels']

        # Reshaping the data to match the syntax of sklearn.
        self.__train_labels = train_labels.ravel()
        self.__test_labels = test_labels.ravel()
        self.__val_labels = val_labels.ravel()
        self.__train_images = train_images.reshape((train_images.shape[0], train_images[0].size))
        self.__test_images = test_images.reshape((test_images.shape[0], test_images[0].size))
        self.__val_images = val_images.reshape((val_images.shape[0], val_images[0].size))
    
    def predict_model(self):
        self.__prediction()
        self.__evaluation()
        return self.pred_score

    def __prediction(self):
        # Setup a SVM model with the best kernel parameter from gridsearch.
        clf = DecisionTreeClassifier()
        clf.fit(self.__train_images, self.__train_labels)
        self.__pred_labels = clf.predict(self.__test_images)
        print(clf.get_depth(), clf.get_n_leaves(), clf.get_params())
        # Default depth = 25, default leaves = 180
    def __evaluation(self):
        # Evaluate the performance of the model.
        self.pred_score = metrics.accuracy_score(self.__test_labels, self.__pred_labels)



