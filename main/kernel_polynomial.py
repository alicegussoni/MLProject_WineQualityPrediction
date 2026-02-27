# POLYNOMIAL KERNEL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

np.random.seed(42)

X_train = pd.read_csv('X_train_scaled.csv').values
X_test  = pd.read_csv('X_test_scaled.csv').values
y_train = pd.read_csv('y_train.csv').values.flatten()
y_test  = pd.read_csv('y_test.csv').values.flatten()

y_train_bin = np.where(y_train == 1, 1, -1)
y_test_bin  = np.where(y_test == 1, 1, -1)


# Polynomial Kernel
def polynomial_kernel(X1, X2, degree, gamma=0.01, coef0=1):
    X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
    return (gamma * (X1 @ X2.T) + coef0) ** degree

# 1. SVM Kernel

def SVM_Kernel(X, y, ParameterLambda, N_iter, k_function,
               X_val=None, y_val=None):
    y = y.flatten()
    n = X.shape[0]
    alpha = np.zeros(n)
    b = 0.0
    K = k_function(X, X)

    loss_history = []
    val_loss_history = []

    for t in range(1, N_iter + 1):

        eta = 1.0 / (ParameterLambda * t)

        i = np.random.randint(0, n)

        f_i = np.dot(alpha * y, K[i]) + b

        alpha *= (1 - eta * ParameterLambda)

        if y[i] * f_i < 1:
            alpha[i] += eta
            b += eta * y[i]


        margins = np.dot(K, alpha * y) + b

        hinge = np.maximum(0, 1 - y * margins)
        hinge_loss = np.mean(hinge)
        reg_loss = 0.5 * ParameterLambda * np.dot(alpha * y, np.dot(K, alpha * y))

        loss_history.append(hinge_loss + reg_loss)

        if X_val is not None and y_val is not None:
            y_val = y_val.flatten()
            K_val = k_function(X_val, X)
            z_v = np.dot(K_val, alpha * y) + b
            hinge_val = np.maximum(0, 1 - y_val * z_v)
            reg_val = 0.5 * ParameterLambda * np.dot(alpha * y, np.dot(K, alpha * y))
            val_loss_history.append(np.mean(hinge_val) + reg_val)

    return alpha, b, loss_history, val_loss_history

def kernel_predict_SVM(X_test, X_train, y_train, alpha, b, k_function):
    K_new = k_function(X_test, X_train)
    return np.dot(K_new, alpha * y_train) + b



# 2. LR Kernel

def logistic_function(z):
    return 1 / (1 + np.exp(-np.clip(z, -20, 20)))


def LR_Kernel(X, y, LearningRate, epoches, lambd, k_function,
              X_val=None, y_val=None):

    y = y.flatten()
    m = X.shape[0]
    alpha = np.zeros(m)
    K = k_function(X, X)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epoches):

        eta = LearningRate / (1 + epoch)

        Ka = np.dot(K, alpha)  
        
        for i in range(m):
            z_i = Ka[i]
            update = logistic_function(-y[i] * z_i)

            reg_gradient = lambd * Ka[i] 

            alpha[i] += eta * (update * y[i] - reg_gradient)

        z_t = np.dot(K, alpha)
        log_loss = np.mean(np.log1p(np.exp(-y * z_t)))
        reg_loss = 0.5 * lambd * np.dot(alpha, np.dot(K, alpha))

        train_loss_history.append(log_loss + reg_loss)

        if X_val is not None and y_val is not None:
            K_val = k_function(X_val, X)
            z_v = np.dot(K_val, alpha)
            log_val = np.mean(np.log1p(np.exp(-y_val * z_v)))
            val_loss_history.append(log_val)

    return alpha, train_loss_history, val_loss_history

def kernel_predict_LR(X_test, X_train, alpha, k_function):
    K_new = k_function(X_test, X_train)
    z = np.dot(K_new, alpha)
    return logistic_function(z)


#3. Cross Validation

def Cross_Validation(algorithm, X, y, k_fold, is_kernel=False, **parameters):

    accuracies = []
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_fold)

    for i in range(k_fold):

        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_fold) if j != i])

        X_train_cv, y_train_cv = X[train_idx], y[train_idx]
        X_val_cv, y_val_cv = X[val_idx], y[val_idx]

        k_func = parameters["k_function"]

        if "SVM" in algorithm.__name__:
            alpha, b, *_ = algorithm(X_train_cv, y_train_cv, **parameters)
            scores = kernel_predict_SVM(X_val_cv, X_train_cv, y_train_cv, alpha, b, k_func)

            y_pred = np.sign(scores)
        else:
            alpha, *_ = algorithm(X_train_cv, y_train_cv, **parameters)
            probs = kernel_predict_LR(X_val_cv, X_train_cv, alpha, k_func)
            y_pred = np.where(probs >= 0.5, 1, -1)

        acc = np.mean(y_pred == y_val_cv)
        accuracies.append(acc)

    return np.mean(accuracies)


# 4. Hyperparameters tuning
# LR Kernel
learning_rates = [0.001, 0.003, 0.01]  
lambdas_lr = [0.01, 0.1]      
epochs_list = [70]           
degrees = [2, 3]

best_accuracy_polynomial_LR = 0
best_parameters_polynomial_LR = {}

for lr in learning_rates:
    for lb in lambdas_lr:
        for ep in epochs_list:
            for d in degrees:

                def current_kernel(X1, X2):
                    return polynomial_kernel(X1, X2, degree=d)

                acc = Cross_Validation(
                    LR_Kernel,
                    X_train,
                    y_train_bin,
                    k_fold=3,
                    is_kernel=True,
                    LearningRate=lr,
                    lambd=lb,
                    epoches=ep,
                    k_function=current_kernel
                )

                print(f"LR Poly -> lr:{lr}, lambda:{lb}, ep:{ep}, deg:{d} = {acc:.4f}")

                if acc > best_accuracy_polynomial_LR:
                    best_accuracy_polynomial_LR = acc
                    best_parameters_polynomial_LR = {
                        "Learning Rate": lr,
                        "Lambda": lb,
                        "Epochs": ep,
                        "Degree": d
                    }


# SVM Kernel
lambdas = [1e-6, 1e-5, 1e-4]
n_iters = [1000]           
degrees = [2, 3]

best_accuracy_polynomial_SVM = 0
best_parameters_polynomial_SVM = {}

for L in lambdas:
    for it in n_iters:
        for d in degrees:

            def current_kernel(X1, X2):
                return polynomial_kernel(X1, X2, degree=d)

            acc = Cross_Validation(
                SVM_Kernel,
                X_train,
                y_train_bin,
                k_fold=3,
                is_kernel=True,
                ParameterLambda=L,
                N_iter=it,
                k_function=current_kernel
            )

            print(f"SVM Poly -> L:{L}, iter:{it}, deg:{d} = {acc:.4f}")

            if acc > best_accuracy_polynomial_SVM:
                best_accuracy_polynomial_SVM = acc
                best_parameters_polynomial_SVM = {
                    "Lambda": L,
                    "N iter": it,
                    "Degree": d
                }



# 5. Final training
# SVM Kernel
best_d_svm = best_parameters_polynomial_SVM["Degree"]

def final_kernel_svm(X1, X2):
    return polynomial_kernel(X1, X2, degree=best_d_svm)

alpha_svm, b_svm, train_svm, val_svm = SVM_Kernel(
    X_train,
    y_train_bin,
    best_parameters_polynomial_SVM["Lambda"],
    2000, 
    final_kernel_svm,
    X_val=None,
    y_val=None
)

y_score_svm = kernel_predict_SVM(X_test, X_train, y_train_bin, alpha_svm, b_svm, final_kernel_svm)
y_pred_svm = np.sign(y_score_svm)


# LR Kernel
best_d_lr = best_parameters_polynomial_LR["Degree"]

def final_kernel_lr(X1, X2):
    return polynomial_kernel(X1, X2, degree=best_d_lr)

alpha_lr, train_lr, val_lr = LR_Kernel(
    X_train,
    y_train_bin,
    best_parameters_polynomial_LR["Learning Rate"],
    150,
    best_parameters_polynomial_LR["Lambda"],
    final_kernel_lr,
    X_val=None,
    y_val=None
)

y_prob_lr = kernel_predict_LR(X_test, X_train, alpha_lr, final_kernel_lr)
y_pred_lr = np.where(y_prob_lr >= 0.5, 1, -1)


# metrics
def compute_metrics(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == -1) & (y_true == -1))
    FP = np.sum((y_pred == 1) & (y_true == -1))
    FN = np.sum((y_pred == -1) & (y_true == 1))

    acc = np.mean(y_pred == y_true)
    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    return acc, prec, rec, f1


acc_svm, prec_svm, rec_svm, f1_svm = compute_metrics(y_test_bin, y_pred_svm)
acc_lr, prec_lr, rec_lr, f1_lr = compute_metrics(y_test_bin, y_pred_lr)

print(f"SVM Polynomial Kernel - Acc: {acc_svm:.4f}, Prec: {prec_svm:.4f}, Rec: {rec_svm:.4f}, F1: {f1_svm:.4f}")
print(f"LR Polynomial Kernel - Acc: {acc_lr:.4f}, Prec: {prec_lr:.4f}, Rec: {rec_lr:.4f}, F1: {f1_lr:.4f}")



# plot

plt.figure(figsize=(8,5))
plt.plot(train_svm, label="Training Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SVM Polynomial Kernel - Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))
plt.plot(train_lr, label="Training Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Logistic Regression Polynomial Kernel - Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# 6. Data save
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

final_json["kernel_models"]["SVM_Polynomial"] = {
    "accuracy": float(acc_svm),
    "precision": float(prec_svm),
    "recall": float(rec_svm),
    "f1": float(f1_svm),
    "loss_history": train_svm
}

final_json["kernel_models"]["LR_Polynomial"] = {
    "accuracy": float(acc_lr),
    "precision": float(prec_lr),
    "recall": float(rec_lr),
    "f1": float(f1_lr),
    "loss_history": train_lr
}

final_json["kernel_models"]["predictions"].update({
    "y_pred_polynomial_svm": y_pred_svm.tolist(),
    "y_score_polynomial_svm": y_score_svm.tolist(),
    "y_prob_polynomial_lr": y_prob_lr.tolist()
})

with open(file_path, "w") as f:
    json.dump(final_json, f, indent=4)

print("Polynomial models saved")