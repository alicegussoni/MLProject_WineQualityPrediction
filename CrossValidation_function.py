# CROSS VALIDATION (for linear svm and lr)
import numpy as np 

def Cross_Validation(algorithm, X, y, k_fold, **parameters):
    accuracies = []
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold = np.array_split(indices, k_fold)

    for i in range(k_fold):
        validation_fold = fold[i]
        train_idx = np.concatenate([fold[j] for j in range(k_fold) if j != i])

        X_train_cv, y_train_cv = X[train_idx], y[train_idx]
        X_val_cv, y_val_cv = X[validation_fold], y[validation_fold]

        results = algorithm(X_train_cv, y_train_cv, **parameters)
        
        w_final = results[0]
        b_final = results[1]
        
        scores = np.dot(X_val_cv, w_final) + b_final
        predictions = np.sign(scores)
        
        acc = np.mean(predictions == y_val_cv)
        accuracies.append(acc)

    return np.mean(accuracies)