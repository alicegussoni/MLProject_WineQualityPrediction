# LOGISTIC REGRESSION FUNCTION
import numpy as np

#sigmoid function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

def LogisticRegression(X, y, LearningRate, epoches, lambd, X_val=None, y_val=None):
    m, n = X.shape 
    w = np.zeros(n)
    loss_history = []
    val_loss_history = []
    
    for epoch in range(epoches):
        for i in range(m):
            x_t = X[i]
            y_t = y[i]
            margin = y_t * np.dot(w, x_t)
            
            update_factor = logistic_function(-margin)
            
            w = w + LearningRate * (update_factor * y_t * x_t - lambd * w)

        margins_train = y * np.dot(X, w)
        current_loss = np.mean(np.log1p(np.exp(-margins_train))) + 0.5 * lambd * np.dot(w, w)
        loss_history.append(float(current_loss))
        
        if X_val is not None and y_val is not None:
            margins_val = y_val * np.dot(X_val, w)
            current_val_loss = np.mean(np.log1p(np.exp(-margins_val))) + 0.5 * lambd * np.dot(w, w)
            val_loss_history.append(float(current_val_loss))
            
    return w, 0, loss_history, val_loss_history



def predict_probability(X, w):
    z = np.dot(X, w)
    return logistic_function(z)