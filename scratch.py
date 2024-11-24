import pandas as pd
import numpy as np
import copy
import math
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv('preprocessed_dataset.csv')

# Select the features and target
X = dataset[['Rooms', 'Bathrooms', 'Car Parks', 'SizeValue']]  # Add other features if needed
y = dataset['Price']

# Normalize/Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to NumPy arrays
X_train = X_scaled
y_train = y.to_numpy()

# Cost function: Mean Squared Error for linear regression
def compute_cost(X, y, w, b):
    m = X.shape[0]  # Number of training examples
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b  # Model's prediction
        cost += (f_wb - y[i]) ** 2  # Squared error
    total_cost = cost / (2 * m)
    return total_cost

# Compute gradient function
def compute_gradient(X, y, w, b):
    m, n = X.shape  # m = number of examples, n = number of features
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b  # Model's prediction
        error = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Gradient descent function
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(num_iters):
        # Calculate gradient
        dj_dw, dj_db = gradient_function(X, y, w, b)
        # Update parameters
        w -= alpha * dj_dw
        b -= alpha * dj_db
        # Save cost
        if i < 100000:
            cost = cost_function(X, y, w, b)
            J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {cost:8.2f}")
    return w, b, J_history

# Initialize parameters
initial_w = np.zeros(X_train.shape[1])  # Weights initialized to zeros (shape: number of features)
initial_b = 0.0  # Bias initialized to zero
iterations = 1000  # Number of iterations for gradient descent
alpha = 0.001  # Lower learning rate to prevent large jumps

# Run gradient descent to optimize the parameters
w_final, b_final, J_history = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

# Final weights and bias
print(f"Final weights: {w_final}")
print(f"Final bias: {b_final}")

# Example: Predict house price in KL for a new set of features
X_new = np.array([3, 2, 1, 1500])  # Example: 3 rooms, 2 bathrooms, 1 car park, 1500 sq ft
X_new_scaled = scaler.transform([X_new])  # Don't forget to scale the new data
predicted_price = np.dot(X_new_scaled, w_final) + b_final
print(f"Predicted price: {predicted_price}")
