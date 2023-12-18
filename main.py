from A.task_A import Svm, Tree
from B.task_B import Cnn

def main():
    # Instance for task A Decision Tree.
    tree = Tree('./Datasets/pneumoniamnist.npz')
    # Train and evaluate the Decision Tree model.
    tree.execution()
    tree.evaluation()
    # Export the results to a .CSV file.
    tree.export()
    
    # Instance for task A SVM. Kernel can be selected as 'linear', 'rbf' or 'poly'
    svm = Svm('linear', './Datasets/pneumoniamnist.npz')
    # Train and evaluate the SVM model.
    svm.execution()
    svm.evaluation()
    # Export the results to a .CSV file.
    svm.export()

    
    # Instance for task B CNN.
    load = True
    CNN_model = Cnn("./Datasets/pathmnist.npz")
    # Train and evaluate the CNN model
    CNN_model.execution(load=load, overwrite=False)
    CNN_model.evaluation()
    # Export the results to a .CSV file if applicable.
    # Most of the training metrics are unavailable if a pre-trained model is used.
    if load == False:
        CNN_model.export()
    else:
        print(f"Prediction accuracy: \n{CNN_model.pred_accuracy}%")

if __name__ == "__main__":
    main()