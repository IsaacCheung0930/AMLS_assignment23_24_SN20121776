import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

class TaskA:
    """
    A class for the SVM model for task A,

    Attribute:
        kernel  : Type of kernel to use in the SVM model (linear, poly, rbf)
        path    : The directory of the file
    """
    def __init__(self, kernel, path):
        """
        Load and reshape the data from the .npz file

        parameters:
            kernel  : Type of kernel to use in the SVM model (linear, poly, rbf)
            path    : The directory of the file
        """
        # Load data from the .npz file.
        data = np.load(path)

        # Sort the dataset into different variables
        train_images = data['train_images']
        test_images = data['test_images']
        val_images = data['val_images']
        train_labels = data['train_labels']
        test_labels = data['test_labels']
        val_labels = data['val_labels']

        # Reshape the variables to match the syntax of sklearn.
        self._train_labels = train_labels.ravel()
        self._test_labels = test_labels.ravel()
        self._val_labels = val_labels.ravel()
        self._train_images = train_images.reshape((train_images.shape[0], train_images[0].size))
        self._test_images = test_images.reshape((test_images.shape[0], test_images[0].size))
        self._val_images = val_images.reshape((val_images.shape[0], val_images[0].size))

        # Define the kernel used for the SVM.
        self._kernel = kernel

    def execution(self):
        """
        Perform model training, validation and prediction using the loaded dataset. 
        """
        self._hyperparameters(self._kernel)
        self._cross_validation(self._kernel, self.best_parameter)
        self._prediction(self._kernel, self.best_parameter)
    
    def evaluation(self):
        """
        Evaluate the performance of the model. 
        Output hyperparameter plots, and calculate the metrics of the trained model. 
        """
        if self._kernel == 'rbf':
            self._hyperparameter_plots(self._grid_parameters, self._grid_search_results)
        self.pred_score = accuracy_score(self._test_labels, self._pred_labels)
        self.precision, self.recall, self.f1score, _ = precision_recall_fscore_support(self._test_labels, self._pred_labels, average='binary')
        self._confusion_matrix_plot(self._test_labels, self._pred_labels)

    def _hyperparameters(self, kernel):
        """
        Search for the best hyperparameter using the validation set. 
        GridSearchCV function from sklearn is used.

        Parameter:
            kernel  : The kernel of the SVM model
        """
        # Define the range for hyperparameters
        C_range = np.logspace(-3, 3, 7)
        gamma_range = np.logspace(-7, -5, 5)
        d_range = [2, 3, 4]
        r_range = np.logspace(-3, 3, 7)
        n_range = [50, 100, 200]
        lr_range = [0.01, 0.1, 1]

        # Rearrange the hyperparameters in dictionary. Kernel defaults to be linear
        if kernel == 'rbf':
            grid_parameters = {'C' : C_range, 'gamma' : gamma_range, 'random_state': [42]}
        elif kernel == 'poly':
            grid_parameters = {'C' : C_range, 'degree' : d_range, 'coef0': r_range, 'random_state': [42]}
        elif kernel == 'tree':
            grid_parameters = {'n_estimators': n_range, 'learning_rate': lr_range}
        else:
            grid_parameters = {'C' : C_range, 'random_state': [42]}
        
        # Perform gridsearch for best parameters
        if kernel == 'tree':
            tree = DecisionTreeClassifier(max_depth = 1)
            grid = GridSearchCV(AdaBoostClassifier(tree), param_grid=grid_parameters, cv = 3, scoring = 'accuracy', n_jobs = -1, verbose=2)
        else:
            grid = GridSearchCV(SVC(kernel = kernel), param_grid = grid_parameters, cv = 3, scoring = 'accuracy', n_jobs = -1, verbose=2)
        
        grid.fit(self._val_images, self._val_labels)
        # Record the parameters from gridsearch
        self._grid_parameters = grid_parameters
        self._grid_search_results = pd.DataFrame(grid.cv_results_)
        self.best_parameter = grid.best_params_
        self.best_score = grid.best_score_

    def _cross_validation(self, kernel, best_parameter):
        """
        Perform cross validation using the training set.

        Parameters:
            kernel          : The kernel of the SVM model
            best_parameter  : The best parameter from the gridsearch function
        """
        # Initiate a SVM model with the best parameters
        if kernel == 'tree':
            cv_model = AdaBoostClassifier(**best_parameter)
        else:
            cv_model = SVC(kernel = kernel, **best_parameter)

        # Initiate a 5-fold cross validation and record the accuracy
        print(self.best_parameter)
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        self.cv_score = cross_val_score(cv_model, self._train_images, self._train_labels, cv = kf, n_jobs=-1, verbose=2)

    def _prediction(self, kernel, best_parameter):
        """
        Predict the labels of the test dataset using the trained SVM model.

        Parameters:
            kernel          : The kernel of the SVM model
            best_parameter  : The best parameter from the gridsearch function
        """
        # Initiate a SVM model with the best kernel parameter
        if kernel == 'tree':
            clf = AdaBoostClassifier(**best_parameter)
        else:
            clf = SVC(kernel = kernel, **best_parameter)

        # Fit the test data in the model and predict the labels
        clf.fit(self._train_images, self._train_labels)
        self._pred_labels = clf.predict(self._test_images)

    def _hyperparameter_plots(self, grid_parameters, grid_search_results):
        """
        Plot the relationship of hyperparameters to the mean test score.

        Parameters:
            grid_parameters     : The predefined parameters used in gridsearch
            grid_search_results : The output dictionary of gridsearch
        """
        # Filter and reshape the grid parameters
        C_values, gamma_values = grid_parameters['C'], grid_parameters['gamma']
        C, gamma = np.meshgrid(C_values, gamma_values)
        mean_test_score = np.array(grid_search_results['mean_test_score']).reshape(len(gamma_values), len(C_values))

        # Plot a contour map to showcase the hyperparameter relations
        plt.figure()
        contour = plt.contourf(C, gamma, mean_test_score, cmap='plasma')
        plt.colorbar(contour, label='Mean Test Score')
        plt.xlabel('C')
        plt.ylabel('Gamma')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Mean Test Score for different combinations of hyperparameters")
        plt.savefig("./A/Plots/RBF Kernel Mean Test Scores.PNG")

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
        conf_matrix.figure_.savefig("./A/Plots/Confusion Matrix.PNG")

def main():
    """
    Use this function if the script is run from this file.
    """
    tree = TaskA('tree','./Datasets/pneumoniamnist.npz')
    tree.execution()
    tree.evaluation()
    print(f"Best AdaBoost parameter & accuracy: \n{tree.best_parameter}, {(100*tree.best_score):>0.1f}%")
    print(f"5-fold cross validation accuracy: \n", [format(i, '>0.1f') for i in [x*100 for x in tree.cv_score]], "%")
    print(f"Prediction accuracy: \n{(100*tree.pred_score):>0.1f}%")
    print(f"Performance metrics:\nPrecision: {(tree.precision):>0.1f}\nRecall: {(tree.recall):>0.1f}\nF1score: {(tree.f1score):>0.1f}")
    
    '''
    svm = TaskA('rbf','./Datasets/pneumoniamnist.npz')
    svm.execution()
    svm.evaluation()
    print(f"Best SVM parameter & accuracy: \n{svm.best_parameter}, {(100*svm.best_score):>0.1f}%")
    print(f"5-fold cross validation accuracy: \n", [format(i, '>0.1f') for i in [x*100 for x in svm.cv_score]], "%")
    print(f"Prediction accuracy: \n{(100*svm.pred_score):>0.1f}%")
    print(f"Performance metrics:\nPrecision: {(svm.precision):>0.1f}\nRecall: {(svm.recall):>0.1f}\nF1score: {(svm.f1score):>0.1f}")
    '''

if __name__ == "__main__":
    main()
    