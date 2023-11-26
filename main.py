from A.task_A import TaskA
from B.task_B import TaskB

def main():
    # Instance for task A
    svm = TaskA('rbf','./Datasets/pneumoniamnist.npz')
    svm.execution()
    svm.evaluation()
    print(f"Best SVM parameter & accuracy: \n{svm.best_parameter}, {(100*svm.best_score):>0.1f}%")
    print(f"5-fold cross validation accuracy: \n", [format(i, '>0.1f') for i in [x*100 for x in svm.cv_score]], "%")
    print(f"Prediction accuracy: \n{(100*svm.pred_score):>0.1f}%")
    print(f"Performance metrics:\nPrecision: {(svm.precision):>0.1f}\nRecall: {(svm.recall):>0.1f}\nF1score: {(svm.f1score):>0.1f}")
    # Instance for task B
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