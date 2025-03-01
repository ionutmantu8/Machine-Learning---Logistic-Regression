import numpy as np

def sigmoid(z):
    """ Compute sigmoid activation. """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def compute_cost(X, y, w, b, lambda_=1):
    """ Compute cost for logistic regression with regularization. """
    m = len(y)
    f_wb = sigmoid(np.dot(X, w) + b)
    f_wb = np.clip(f_wb, 1e-10, 1 - 1e-10)
    cost = -np.mean(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost

def compute_gradient(X, y, w, b, lambda_):
    """ Compute gradients for logistic regression. """
    m = len(y)
    err = sigmoid(np.dot(X, w) + b) - y
    dj_dw = (np.dot(X.T, err) / m) + (lambda_ / m) * w
    dj_db = np.mean(err)
    return dj_db, dj_dw

def predict(X, w, b):
    """ Predicts binary class labels. """
    return sigmoid(np.dot(X, w) + b) >= 0.5
