from A.task_A import TaskA
from B.task_B import TaskB

def main():
    # Instance for task A
    svm = TaskA('rbf','./Datasets/pneumoniamnist.npz')
    svm.execution()
    svm.evaluation()
    print(f"Best SVM parameter & accuracy: \n{svm.best_parameter}, {(100*svm.best_score):>0.1f}%")
    print(f"5-fold cross validation accuracy: \n{(svm.cv_score):>0.1f}%")
    print(f"Prediction accuracy: \n{(100*svm.pred_score)}%")
    print(f"Performance metrics:\nPrecision: {(svm.precision):>0.1f}\nRecall: {(svm.recall):>0.1f}\nF1score: {(svm.f1score):>0.1f}")

    # Instance for task B
    CNN_model = TaskB("./Datasets/pathmnist.npz")
    CNN_model.execution()
    CNN_model.evaluation()

if __name__ == "__main__":
    main()