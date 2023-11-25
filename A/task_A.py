import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

class TaskA:
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
        self._train_labels = train_labels.ravel()
        self._test_labels = test_labels.ravel()
        self._val_labels = val_labels.ravel()
        self._train_images = train_images.reshape((train_images.shape[0], train_images[0].size))
        self._test_images = test_images.reshape((test_images.shape[0], test_images[0].size))
        self._val_images = val_images.reshape((val_images.shape[0], val_images[0].size))

        # Define the kernel used for the SVM.
        self._kernel = kernel

    def execution(self):
        self._hyperparameters(self._kernel)
        self._cross_validation(self._kernel, self.best_parameter)
        self._prediction(self._kernel, self.best_parameter)
    
    def evaluation(self):
        # Hyperparameter scores, Pred_accuracy, precision, recall, f1score, confusion matrix
        if self._kernel == 'rbf':
            self._hyperparameter_plots(self._grid_parameters, self._grid_search_results)
        self.pred_score = accuracy_score(self._test_labels, self._pred_labels)
        self.precision, self.recall, self.f1score, _ = precision_recall_fscore_support(self._test_labels, self._pred_labels, average='binary')
        self._confusion_matrix_plot(self._test_labels, self._pred_labels)

    def _hyperparameters(self, kernel):
        # Setting range for hyperparameters.
        C_range = np.logspace(-3, 3, 7)
        gamma_range = np.logspace(-7, -5, 5)
        d_range = [2, 3, 4]
        r_range = np.logspace(-3, 3, 7)
        
        # Performing gridsearch for best parameters, kernel defaults to be linear
        if kernel == 'rbf':
            grid_parameters = {'C' : C_range, 'gamma' : gamma_range, 'random_state': [42]}
        elif kernel == 'poly':
            grid_parameters = {'C' : C_range, 'degree' : d_range, 'coef0': r_range, 'random_state': [42]}
        else:
            grid_parameters = {'C' : C_range, 'random_state': [42]}

        grid = GridSearchCV(SVC(kernel = kernel), param_grid = grid_parameters, cv = 3, scoring = 'accuracy', n_jobs = -1)
        grid.fit(self._val_images, self._val_labels)

        self._grid_parameters = grid_parameters
        self._grid_search_results = pd.DataFrame(grid.cv_results_)
        self.best_parameter = grid.best_params_
        self.best_score = grid.best_score_

    def _cross_validation(self, kernel, best_parameter):
        # Initiate a SVM model with the best parameter from gridsearch.
        cv_model = SVC(kernel = kernel, **best_parameter)

        # Setup parameters for cross validation. 
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        self.cv_score = cross_val_score(cv_model, self._train_images, self._train_labels, cv = kf)
        #self.cv_score = [format(score, '>0.1f') for score in cv_score]

    def _prediction(self, kernel, best_parameter):
        # Setup a SVM model with the best kernel parameter from gridsearch.
        clf = SVC(kernel = kernel, **best_parameter)
        clf.fit(self._train_images, self._train_labels)
        self._pred_labels = clf.predict(self._test_images)

    def _hyperparameter_plots(self, grid_parameters, grid_search_results):
        C_values, gamma_values = grid_parameters['C'], grid_parameters['gamma']
        C, gamma = np.meshgrid(C_values, gamma_values)
        mean_test_score = np.array(grid_search_results['mean_test_score']).reshape(len(gamma_values), len(C_values))

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
        conf_matrix = ConfusionMatrixDisplay(confusion_matrix(true_values, pred_values))
        conf_matrix.plot(cmap= "plasma")
        conf_matrix.figure_.savefig("./A/Plots/Confusion Matrix.PNG")

def main():
    svm = TaskA('rbf','./Datasets/pneumoniamnist.npz')
    svm.execution()
    svm.evaluation()
    print(f"Best SVM parameter & accuracy: \n{svm.best_parameter}, {(100*svm.best_score):>0.1f}%")
    print(f"5-fold cross validation accuracy: \n", [format(i, '>0.1f') for i in [x*100 for x in svm.cv_score]], "%")
    print(f"Prediction accuracy: \n{(100*svm.pred_score):>0.1f}%")
    print(f"Performance metrics:\nPrecision: {(svm.precision):>0.1f}\nRecall: {(svm.recall):>0.1f}\nF1score: {(svm.f1score):>0.1f}")



    
if __name__ == "__main__":
    main()
    