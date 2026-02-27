# MLProject_WineQualityPrediction
This project focuses on binary classification of wine quality using machine learning models implemented from scratch: Support Vector Machine (SVM) and Logistic Regression (LR)

The dataset contains physicochemical properties of red and white wines.
The original quality score was transformed into a binary classification problem:

- Good (1) → quality ≥ 6
- Bad (0) → quality < 6
  
The objective is to compare linear and kernel-based classifiers and analyze their bias–variance behavior.

Project structure:
├── DataExploration&Preprocessing.py
├── SupportVectorMachine_function.py
├── LogisticRegression_function.py
├── CrossValidation_function.py
├── Svm&Lr_linear.py
├── kernel_polynomial.py
├── Kernel_gaussian.py
├── Graphs.py
├── project_results_complete.json

Methodology:

Data Preprocessing:
- Outlier removal using the Interquartile Range (IQR) method
- Stratified 80/20 train-test split
- Feature standardization computed exclusively on the training set
- Identical transformation applied to the test set to prevent data leakage

Implemented Models:
- Linear Models: 
  Soft-margin Support Vector Machine (hinge loss) and 
  Logistic Regression (logistic loss)

- Kernel Extensions:
  Polynomial kernel and
  Gaussian (RBF) kernel

Hyperparameters were selected using cross-validation on the training set.

Evaluation:
Performance was assessed using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve and Area Under the Curve (AUC)
- Confusion Matrix
- Learning Curves for bias–variance analysis
