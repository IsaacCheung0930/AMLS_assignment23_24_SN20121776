from A.task_A import Svm, Tree
from B.task_B import Cnn

def main():
    '''
    User Menu for task A and task B.
    '''
    while True:
        option = input("Task Selection Menu:\nA: Task A\nB: Task B\nE: Exit\n")
        if option == "A":
            while True:
                model = input("Selection Menu for Task A:\n1: Tree\n2: Linear SVM\n3: Poly SVM\n4: RBF SVM\nE: Return\n")
                if model == "1":
                    # Instance for task A Decision Tree.
                    tree = Tree('./Datasets/pneumoniamnist.npz')
                    # Train and evaluate the Decision Tree model.
                    tree.execution()
                    tree.evaluation()
                    # Export the results to a .CSV file.
                    tree.export()
                    break

                elif model == "2" or model == "3" or model == "4":
                    # Kernel can be selected as 'linear', 'poly' or 'rbf'.
                    kernel = {2: "linear", 3: "poly", 4: "rbf"}
                    # Instance for task A SVM. 
                    svm = Svm(kernel[int(model)], './Datasets/pneumoniamnist.npz')
                    # Train and evaluate the SVM model.
                    svm.execution()
                    svm.evaluation()
                    # Export the results to a .CSV file.
                    svm.export()
                    break
                
                elif model == "E":
                    print("Return to previous step.")
                    break

                else:
                    print("Invalid entry. Please try again")

        elif option == "B":
            while True:
                model = input("Selection Menu for Task B:\n1: Load old model\n2: Train new model\nE: Return\n")
                if model == "1" or model == "2":
                    # Choose if the pre-trained model is loaded.
                    load = {1: True, 2: False}
                    # Instance for task B CNN. 
                    CNN_model = Cnn("./Datasets/pathmnist.npz")
                    # Train and evaluate the CNN model
                    CNN_model.execution(load=load[int(model)], overwrite=False)
                    CNN_model.evaluation()
                    # Export the results to a .CSV file if applicable.
                    # Most of the training metrics are unavailable if a pre-trained model is used.
                    if load[int(model)] == False:
                        CNN_model.export()
                    else:
                        print(f"Prediction accuracy: \n{CNN_model.pred_accuracy}%")
                    break

                elif model == "E":
                    print("Return to previous step.")
                    break

                else:
                    print("Invalid entry. Please try again")

        elif option == "E":
            print("Program terminated.")
            break

        else:
            print("Invalid entry. Please try again")

if __name__ == "__main__":
    main()