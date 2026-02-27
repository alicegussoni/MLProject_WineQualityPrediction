# SVM e LR LINEAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
X_train = pd.read_csv('X_train_scaled.csv').values
X_test  = pd.read_csv('X_test_scaled.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test  = pd.read_csv('y_test.csv').values.flatten()

y_train_bin = np.where(y_train == 1, 1, -1)
y_test_bin  = np.where(y_test == 1, 1, -1)

if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)
if X_test.ndim == 1:
    X_test = X_test.reshape(-1, 1)

# 1. SUPPORT VECTOR MACHINE
from SupportVectorMachine_function import SupportVectorMachine


# Tuning hyperparameters
LearningRate = [0.0001, 0.001, 0.002]
ParameterLambda = [0.0001, 0.001, 0.01]
N_iter = [50, 100]

best_parameters = {}
best_accuracy = 0 

from CrossValidation_function import Cross_Validation
for LR in LearningRate:
     for PL in ParameterLambda:
         for it in N_iter:
             current_accuracy = Cross_Validation(
                 SupportVectorMachine, 
                 X_train, 
                 y = y_train_bin, 
                 k_fold = 5, 
                 LearningRate = LR, 
                 ParameterLambda = PL, 
                 N_iter = it
             )
             print(f"LR: {LR}, PL: {PL}, N_iter: {it}, current accuracy: {current_accuracy}")
             if current_accuracy > best_accuracy:
                 best_accuracy = current_accuracy
                 best_parameters_SVM = {
                     'Learning Rate': LR,
                     'Parameter Lambda': PL,
                     'N iter': it
                 }

print(f"Best accuracy: {best_accuracy}")
print(f"Best Parameters: {best_parameters_SVM}")



# SVM final 
w_best_SVM, b_best_SVM, *extras = SupportVectorMachine(
    X_train, 
    y_train_bin, 
    LearningRate = best_parameters_SVM['Learning Rate'],
    ParameterLambda = best_parameters_SVM['Parameter Lambda'],
    N_iter = best_parameters_SVM['N iter'],
    X_val = None,     
    y_val = None  
)
history_loss_SVM = extras[0] if len(extras) > 0 else []

linear_output = np.dot(X_test, w_best_SVM) + b_best_SVM
y_pred_test_SVM = np.sign(linear_output)

test_accuracy_SVM = np.mean(y_pred_test_SVM == y_test_bin)
print(f"Final accuracy on test set: {test_accuracy_SVM:.4f}")


#plot

plt.figure(figsize=(8,5))
plt.plot(history_loss_SVM, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SVM Linear - Training Loss")
plt.legend()
plt.grid(True)
plt.show()


# 2. LOGISTIC REGRESSION
from LogisticRegression_function import LogisticRegression

# Tuning hyperparameters
LearningRate = [0.0001, 0.001, 0.002]
ParameterLambda = [0.0001, 0.001, 0.01]
N_iter = [50, 100] 

best_parameters_lr = {}
best_accuracy_lr = 0

for LR in LearningRate:
    for PL in ParameterLambda:
        for it in N_iter:
            current_accuracy = Cross_Validation(
                LogisticRegression, 
                X_train, 
                y = y_train_bin, 
                k_fold = 5, 
                LearningRate = LR, 
                lambd = PL, 
                epoches = it
            )
            print(f"LR: {LR}, Lambda: {PL}, Epoche: {it}, current accuracy: {current_accuracy:.4f}")
            
            if current_accuracy > best_accuracy_lr:
                best_accuracy_lr = current_accuracy
                best_parameters_lr = {
                    'Learning Rate': LR,
                    'Lambda': PL,
                    'Epochs': it
                }

print(f"Best LR accuracy: {best_accuracy_lr:.4f}")
print(f"Best parameters LR: {best_parameters_lr}")


# LR final
w_best_LR, b_best_LR, *extras = LogisticRegression(
    X_train, 
    y_train_bin, 
    LearningRate = best_parameters_lr['Learning Rate'],
    lambd = best_parameters_lr['Lambda'],
    epoches = best_parameters_lr['Epochs'],
    X_val = None,      
    y_val = None 
)
history_loss_LR = extras[0] if len(extras) > 0 else []


linear_output_LR = np.dot(X_test, w_best_LR) + b_best_LR
y_pred_test_LR = np.sign(linear_output_LR)

test_accuracy_LR = np.mean(y_pred_test_LR == y_test_bin)

print(f"LR final accuracy on test set: {test_accuracy_LR :.4f}")


#plot

plt.figure(figsize=(8,5))
plt.plot(history_loss_LR, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Logistic Regression Linear - Training Loss")
plt.legend()
plt.grid(True)
plt.show()



# 3. analysis
print("Accuracy SVM vs LR: ")
print(f"SVM: {test_accuracy_SVM:.4f}")
print(f"LG: {test_accuracy_LR:.4f}")

print("Confusion matrix SVM: ")
TP_svm = np.sum((y_pred_test_SVM ==  1) & (y_test_bin ==  1))
TN_svm = np.sum((y_pred_test_SVM == -1) & (y_test_bin == -1))
FP_svm = np.sum((y_pred_test_SVM ==  1) & (y_test_bin == -1))
FN_svm = np.sum((y_pred_test_SVM == -1) & (y_test_bin ==  1))

confusion_matrix = np.array([[TP_svm, FP_svm],
                             [FN_svm, TN_svm]])

print("Confusion Matrix")
print("          Pred +1   Pred -1")
print(f"True +1     {TP_svm}        {FN_svm}")
print(f"True -1     {FP_svm}        {TN_svm}")


print("Confusion matrix LR: ")
TP_lr = np.sum((y_pred_test_LR ==  1) & (y_test_bin ==  1))
TN_lr = np.sum((y_pred_test_LR == -1) & (y_test_bin == -1))
FP_lr = np.sum((y_pred_test_LR ==  1) & (y_test_bin == -1))
FN_lr = np.sum((y_pred_test_LR == -1) & (y_test_bin ==  1))

confusion_matrix = np.array([[TP_lr, FP_lr],
                             [FN_lr, TN_lr]])

print("Confusion Matrix")
print("          Pred +1   Pred -1")
print(f"True +1     {TP_lr}        {FN_lr}")
print(f"True -1     {FP_lr}        {TN_lr}")


precision_SVM = TP_svm / (TP_svm + FP_svm) if (TP_svm + FP_svm) > 0 else 0
print(f"Precision SVM: {precision_SVM:.4f}")
precision_LR = TP_lr / (TP_lr + FP_lr) if (TP_lr + FP_lr) > 0 else 0
print(f"Precision LR : {precision_LR:.4f}")

print("Recall: ")
recall_SVM = TP_svm / (TP_svm + FN_svm) if (TP_svm + FN_svm) > 0 else 0
print(f"Recall SVM : {recall_SVM:.4f}")
recall_LR = TP_lr / (TP_lr + FN_lr) if (TP_lr + FN_lr) > 0 else 0
print(f"Recall LR : {recall_LR:.4f}")

print("F1-score: ")
f1_score_SVM = 2 * (precision_SVM * recall_SVM) / (precision_SVM + recall_SVM) if (precision_SVM + recall_SVM) > 0 else 0
print(f"F1-score SVM : {f1_score_SVM:.4f}")
f1_score_LR = 2 * (precision_LR * recall_LR) / (precision_LR + recall_LR) if (precision_LR + recall_LR) > 0 else 0
print(f"F1-score LR : {f1_score_LR:.4f}")

print(f"LR Lineare - Acc: {test_accuracy_LR:.4f}, Prec: {precision_LR:.4f}, Rec: {recall_LR:.4f}, F1: {f1_score_LR:.4f}")
print(f"SVM Lineare - Acc: {test_accuracy_SVM:.4f}, Prec: {precision_SVM:.4f}, Rec: {recall_SVM:.4f}, F1: {f1_score_SVM:.4f}")


# 4. data saved
import json
import os

file_path = "project_results_complete.json"

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        final_json = json.load(f)
else:
    final_json = {
        "linear_models": {},
        "kernel_models": {"predictions": {}}
    }
final_json["linear_models"] = {
    "LR_Linear": {
        "accuracy": float(test_accuracy_LR),
        "precision": float(precision_LR),
        "recall": float(recall_LR),
        "f1": float(f1_score_LR),
        "loss_history": [float(l) for l in history_loss_LR]
    },
    "SVM_Linear": {
        "accuracy": float(test_accuracy_SVM),
        "precision": float(precision_SVM),
        "recall": float(recall_SVM),
        "f1": float(f1_score_SVM),
        "loss_history": [float(l) for l in history_loss_SVM]
    },
    "predictions": {
        "y_test_real": y_test_bin.tolist(),
        "y_pred_linear_lr": y_pred_test_LR.tolist(),
        "y_pred_linear_svm": y_pred_test_SVM.tolist(),
        "y_score_linear_lr": linear_output_LR.tolist(),
        "y_score_linear_svm": linear_output.tolist()
    }
}

with open(file_path, "w") as f:
    json.dump(final_json, f, indent=4)

print("Linear models saved")