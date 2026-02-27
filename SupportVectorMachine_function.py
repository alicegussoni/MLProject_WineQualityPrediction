# SUPPORT VECTOR MACHINE FUNCTION
import numpy as np 

def SupportVectorMachine(X, y, LearningRate, ParameterLambda, N_iter, 
                         X_val=None, y_val=None):

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    train_loss_history = []
    val_loss_history = []

    for epoch in range(N_iter):
        for i in range(n_samples):

            margin = y[i] * (np.dot(X[i], w) + b)

            if margin < 1:
                w = w - LearningRate * (ParameterLambda * w - y[i] * X[i])
                b = b + LearningRate * y[i]
            else:
                w = w - LearningRate * (ParameterLambda * w)

        # TRAIN LOSS
        margins = y * (X @ w + b)
        hinge_loss = np.mean(np.maximum(0, 1 - margins))
        reg_loss = 0.5 * ParameterLambda * np.dot(w, w)
        train_loss_history.append(float(hinge_loss + reg_loss))

        # VALIDATION LOSS
        if X_val is not None and y_val is not None:
            margins_val = y_val * (X_val @ w + b)
            hinge_val = np.mean(np.maximum(0, 1 - margins_val))
            val_loss = hinge_val + 0.5 * ParameterLambda* np.dot(w, w)
            val_loss_history.append(float(val_loss))

    return w, b, train_loss_history, val_loss_history
