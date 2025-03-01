import numpy as np
import copy
from model import compute_cost, compute_gradient

def gradient_descent(X, y, w, b, alpha, num_iters, lambda_=0.1):
    """ Train logistic regression model using gradient descent. """
    J_history = []
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % (num_iters // 10) == 0:
            J_history.append(compute_cost(X, y, w, b, lambda_))
            print(f"Iteration {i}: Cost {J_history[-1]:.6f}")
    return w, b

def train_model(X, y, alpha=0.1, iterations=10000):
    w = np.zeros(X.shape[1])
    b = 0.
    return gradient_descent(X, y, w, b, alpha, iterations)
