# AMLS_assignment23_24
## Description
This is an assignment for ELEC0134 Advanced Machine Learning System. 

## Project Organisation
AMLS_23-24_SN20121776
- A
    - task_A.py 
    - Plots
    - Results
- B
    - task_B.py
    - Plots
    - Results
- Datasets
- .gitignore
- main.py
- README.md
- requirement.txt

1. Task_A.py consists of a SVM model and Task_B.py consists of a CNN model.
2. Figures generated within the code are saved in the Plots folder for each task.
3. Exported data (performance metrics) are saved in the Results folder for each task 
4. The dataset folder holds the pneumoniamnist.npz and pathmnist.npz for task A and B respectively.
5. The .gitignore file prevents unnecessary files from being uploaded.
6. Both task A and B can be run at once using main.py.
8. All packages installed in the virtual environement is listed in requirement.txt.

## Running the Project
This project (both tasks) can be run using the main.py, or individually using the respective .py files. Note that the SVM model in Task A allows for different kernels. The default kernel is 'linear', but can be changed to 'rbf' or 'poly' if necessary. The performance metrics of the models can be viewed using the .CSV files in the Results folder; while the generated plots can be found in the Plots folder.

## Required Packages
- numpy           1.26.2
- pandas          2.0.3
- matplotlib      3.7.2
- scikit-learn    1.1.3
- torch           2.1.0
\
More detailed package list can be found in requirement.txt.