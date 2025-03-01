import numpy as np
from model import sigmoid, predict
from train import train_model

# Training data
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Train model
w_final, b_final = train_model(X_train, y_train)

# Test data
X_test = np.array([[1, 1.5], [2.5, 1], [0.5, 0.5], [3, 2.5]])
y_expected = np.array([0, 1, 0, 1])

predictions = predict(X_test, w_final, b_final)

print("\nTesting Results:")
for i in range(len(X_test)):
    print(f"Point: {X_test[i]} | Prediction: {int(predictions[i])} | Expected: {y_expected[i]}")
