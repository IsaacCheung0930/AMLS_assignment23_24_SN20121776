from A.task_A import TaskA
from B.task_B import TaskB

def main():
    # Instance for task A
    

    # Instance for task B
    CNN_model = TaskB("./Datasets/pathmnist.npz")
    CNN_model.execution()
    CNN_model.evaluation()

if __name__ == "__main__":
    main()