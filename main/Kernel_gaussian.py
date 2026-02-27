# GAUSSIAN KERNEL
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

def gaussian_kernel(X1, X2, gamma):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)

    sq_dists = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-sq_dists / (2 * gamma))#



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


def LR_Kernel(X, y, LearningRate, epoches, lambd, k_function, X_val=None, y_val=None):
    m = X.shape[0]
    alpha = np.zeros(m)
    K_matrix = k_function(X, X)
    train_loss_history = []
    val_loss_history = []
    y = y.flatten()


    for epoch in range(epoches):
        Ka = np.dot(K_matrix, alpha)

        for i in range(m):
            z_i = Ka[i]

            update_factor = logistic_function(-(y[i] * z_i))

            reg_gradient = lambd * Ka[i]

            alpha[i] = alpha[i] + LearningRate * (update_factor * y[i] - reg_gradient)

        z_t = np.dot(K_matrix, alpha)
        log_t = np.mean(np.log1p(np.exp(-np.clip(y * z_t, -20, 20))))
        reg_loss = 0.5 * lambd * np.dot(alpha, np.dot(K_matrix, alpha))
        train_loss_history.append(log_t + reg_loss)

        if X_val is not None and y_val is not None:
            K_val = k_function(X_val, X)
            z_v = np.dot(K_val, alpha)
            log_v = np.mean(np.log1p(np.exp(-np.clip(y_val * z_v, -20, 20))))
            val_loss_history.append(log_v + reg_loss)

    return alpha, train_loss_history, val_loss_history



def kernel_predict_LR(X_test, X_train, alpha, k_function):
    K_new = k_function(X_test, X_train)
    z = np.dot(K_new, alpha)
    return logistic_function(z) 



# 3. Cross Validation
def Cross_Validation(algorithm, X, y, k_fold, is_kernel=False, **parameters):
    accuracies = []
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_fold)

    for i in range(k_fold):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_fold) if j != i])

        X_train_cv, y_train_cv = X[train_idx], y[train_idx]
        X_val_cv, y_val_cv = X[val_idx], y[val_idx]

        if is_kernel:
            k_func = parameters.get('k_function')
            

            if "SVM" in algorithm.__name__:
                alpha, b, *_ = algorithm(X_train_cv, y_train_cv, **parameters)
                y_score = kernel_predict_SVM(X_val_cv, X_train_cv, y_train_cv, alpha, b, k_func)

                y_pred = np.sign(y_score)

            else:
                alpha, *_ = algorithm(X_train_cv, y_train_cv, **parameters)
                y_prob = kernel_predict_LR(X_val_cv, X_train_cv, alpha, k_func)
                y_pred = np.where(y_prob >= 0.5, 1, -1)
        else:
            w_final, b_final = algorithm(X_train_cv, y_train_cv, **parameters)
            scores = np.dot(X_val_cv, w_final) + b_final
            y_pred = np.sign(scores)

        acc = np.mean(y_pred == y_val_cv)
        accuracies.append(acc)

    return np.mean(accuracies)


# 4. Hyperparameters tuning
# SVM Kernel
lambdas = [0.001, 0.01]       
n_iters = [2000]        
gamma = [0.1, 0.5] 


best_accuracy_gaussian_SVM = 0
best_parameters_gaussian_SVM = {}


for g in gamma:
    for L in lambdas:
        for it in n_iters:
            def current_gau_kernel(X1, X2):
                return gaussian_kernel(X1, X2, gamma=g)
            acc = Cross_Validation(
                SVM_Kernel, 
                X_train, 
                y_train_bin,
                k_fold=3, 
                is_kernel=True,
                ParameterLambda=L, 
                N_iter=it, 
                k_function=current_gau_kernel
            )
            
            print(f"Lambda: {L}, Iterations: {it}, Gamma: {g} -> Accuracy CV: {acc:.4f}")
            
            if acc > best_accuracy_gaussian_SVM:
                best_accuracy_gaussian_SVM = acc
                best_parameters_gaussian_SVM = {
                    'Parameter Lambda': L, 
                    'N iter': it,
                    'Gamma': g
                }


print(f"Best accuracy Gaussian SVM: {best_accuracy_gaussian_SVM:.4f}")
print(f"Best parameters: {best_parameters_gaussian_SVM}")


# LR Kernel
learning_rates = [0.003, 0.01]
lambdas_lr = [0.01, 0.1]
gamma = [0.1, 0.5]
epochs_list = [100]


best_accuracy_gaussian_LR = 0
best_parameters_gaussian_LR = {}


for g in gamma:
    for lr in learning_rates:
        for lb in lambdas_lr:
            for ep in epochs_list:
                def current_gau_kernel(X1, X2):
                    return gaussian_kernel(X1, X2, gamma=g)
                acc = Cross_Validation(
                    LR_Kernel, 
                    X_train, 
                    y_train_bin, 
                    k_fold=3, 
                    is_kernel=True,
                    LearningRate=lr, 
                    lambd=lb,
                    epoches=ep,
                    k_function=current_gau_kernel
                )
                
                print(f"LR: {lr}, Lambda: {lb}, Epochs: {ep}, Gamma: {g} -> Accuracy CV: {acc:.4f}")
                
                if acc > best_accuracy_gaussian_LR:
                    best_accuracy_gaussian_LR = acc
                    best_parameters_gaussian_LR = {
                        'Learning Rate': lr, 
                        'Lambda': lb, 
                        'Epochs': ep,
                        'Gamma': g
                    }
print(f"Best accuracy Gaussian LR: {best_accuracy_gaussian_LR:.4f}")
print(f"Best parameters: {best_parameters_gaussian_LR}")



# 5. final
# SVM Kernel
best_g_svm = best_parameters_gaussian_SVM["Gamma"]

def final_gau_kernel_svm(X1, X2):
    return gaussian_kernel(X1, X2, gamma=best_g_svm)

alpha_final_Gaussian_SVM, b_final_Gaussian_SVM, *extras = SVM_Kernel(
    X = X_train,
    y = y_train_bin,
    ParameterLambda = best_parameters_gaussian_SVM["Parameter Lambda"],
    N_iter = 5000, 
    k_function = final_gau_kernel_svm,
    X_val = None,      
    y_val = None   
)


history_loss_gaussian_SVM = extras[0] if len(extras) > 0 else []


y_score_Gaussian_SVM = kernel_predict_SVM(
    X_test, X_train,
    y_train_bin,
    alpha_final_Gaussian_SVM,
    b_final_Gaussian_SVM,
    final_gau_kernel_svm
)

y_pred_Gaussian_SVM = np.sign(y_score_Gaussian_SVM)



TP_Gaussian_SVM = np.sum((y_pred_Gaussian_SVM == 1) & (y_test_bin == 1))
TN_Gaussian_SVM = np.sum((y_pred_Gaussian_SVM == -1) & (y_test_bin == -1))
FP_Gaussian_SVM = np.sum((y_pred_Gaussian_SVM == 1) & (y_test_bin == -1))
FN_Gaussian_SVM = np.sum((y_pred_Gaussian_SVM == -1) & (y_test_bin == 1))

acc_Gaussian_SVM = np.mean(y_pred_Gaussian_SVM == y_test_bin)
prec_Gaussian_SVM = TP_Gaussian_SVM / (TP_Gaussian_SVM + FP_Gaussian_SVM) if (TP_Gaussian_SVM + FP_Gaussian_SVM) > 0 else 0
rec_Gaussian_SVM = TP_Gaussian_SVM / (TP_Gaussian_SVM + FN_Gaussian_SVM) if (TP_Gaussian_SVM + FN_Gaussian_SVM) > 0 else 0
f1_Gaussian_SVM = 2 * (prec_Gaussian_SVM * rec_Gaussian_SVM) / (prec_Gaussian_SVM + rec_Gaussian_SVM) if (prec_Gaussian_SVM + rec_Gaussian_SVM) > 0 else 0

print(f"SVM Gaussian Kernel - Acc: {acc_Gaussian_SVM:.4f}, Prec: {prec_Gaussian_SVM:.4f}, Rec: {rec_Gaussian_SVM:.4f}, F1: {f1_Gaussian_SVM:.4f}")



#plot

plt.figure(figsize=(8,5))
plt.plot(history_loss_gaussian_SVM, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SVM Gaussian Kernel - Training Loss")
plt.legend()
plt.grid(True)
plt.show()



# LR Kernel
best_lr = best_parameters_gaussian_LR["Learning Rate"]
best_lb = best_parameters_gaussian_LR["Lambda"]
best_ep = best_parameters_gaussian_LR["Epochs"]
best_g  = best_parameters_gaussian_LR["Gamma"]

def final_gau_kernel_lr(X1, X2):
    return gaussian_kernel(X1, X2, gamma=best_g)

alpha_final_Gaussian_LR, history_loss_gaussian_LR, history_val_loss_gaussian_LR = LR_Kernel(
    X = X_train,
    y = y_train_bin,
    LearningRate = best_lr , 
    lambd = best_lb, 
    epoches = best_ep,        
    k_function = final_gau_kernel_lr,
    X_val = None,
    y_val = None
)


y_prob_Gaussian_LR = kernel_predict_LR(X_test, X_train, alpha_final_Gaussian_LR, final_gau_kernel_lr)
y_pred_Gaussian_LR = np.where(y_prob_Gaussian_LR >= 0.5, 1, -1)


TP_Gaussian_LR = np.sum((y_pred_Gaussian_LR ==  1) & (y_test_bin ==  1))
TN_Gaussian_LR = np.sum((y_pred_Gaussian_LR == -1) & (y_test_bin == -1))
FP_Gaussian_LR = np.sum((y_pred_Gaussian_LR ==  1) & (y_test_bin == -1))
FN_Gaussian_LR = np.sum((y_pred_Gaussian_LR == -1) & (y_test_bin ==  1))

test_accuracy_Gaussian_LR = np.mean(y_pred_Gaussian_LR == y_test_bin)
precision_Gaussian_LR = TP_Gaussian_LR / (TP_Gaussian_LR + FP_Gaussian_LR) if (TP_Gaussian_LR + FP_Gaussian_LR) > 0 else 0
recall_Gaussian_LR = TP_Gaussian_LR / (TP_Gaussian_LR + FN_Gaussian_LR) if (TP_Gaussian_LR + FN_Gaussian_LR) > 0 else 0
f1_score_Gaussian_LR = 2 * (precision_Gaussian_LR * recall_Gaussian_LR) / (precision_Gaussian_LR + recall_Gaussian_LR) if (precision_Gaussian_LR + recall_Gaussian_LR) > 0 else 0

print(f"LR Gaussian Kernel - Acc: {test_accuracy_Gaussian_LR:.4f}, Prec: {precision_Gaussian_LR:.4f}, Rec: {recall_Gaussian_LR:.4f}, F1: {f1_score_Gaussian_LR:.4f}")


#plot
plt.figure(figsize=(8,5))
plt.plot(history_loss_gaussian_LR, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Logistic Regression Gaussian Kernel - Training Loss")
plt.legend()
plt.grid(True)
plt.show()



# Save data
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

final_json["kernel_models"]["SVM_Gaussian"] = {
    "accuracy": float(acc_Gaussian_SVM),
    "precision": float(prec_Gaussian_SVM),
    "recall": float(rec_Gaussian_SVM),
    "f1": float(f1_Gaussian_SVM),
    "loss_history": [float(l) for l in history_loss_gaussian_SVM],
    "best_params": best_parameters_gaussian_SVM
}

final_json["kernel_models"]["LR_Gaussian"] = {
    "accuracy": float(test_accuracy_Gaussian_LR),
    "precision": float(precision_Gaussian_LR),
    "recall": float(recall_Gaussian_LR),
    "f1": float(f1_score_Gaussian_LR),
    "loss_history": [float(l) for l in history_loss_gaussian_LR],
    "best_params": best_parameters_gaussian_LR
}

final_json["kernel_models"]["predictions"].update({
    "y_prob_gaussian_lr": y_prob_Gaussian_LR.tolist(),
    "y_pred_gaussian_svm": y_pred_Gaussian_SVM.tolist(),
    "y_score_gaussian_svm": y_score_Gaussian_SVM.tolist()
})

with open(file_path, "w") as f:
    json.dump(final_json, f, indent=4)

print("Gaussian models saved")
