# Simple Stochastic Gradient Descent (SGD) Example
# Goal: Fit a line y = w*x + b to data using SGD

import random

# Step 1: Create some sample data (y = 2x + 1)
X = [1, 2, 3, 4, 5]
Y = [2*x + 1 for x in X]

# Step 2: Initialize parameters (weights)
w = 0.0   # slope
b = 0.0   # intercept

# Step 3: Set hyperparameters
learning_rate = 0.01
epochs = 50  # number of passes through the data

# Step 4: Define the prediction function
def predict(x):
    return w * x + b

# Step 5: Define the loss function (Mean Squared Error)
def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

# Step 6: Training loop using SGD
for epoch in range(epochs):
    total_loss = 0

    # Go through each data point one by one (stochastic)
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        # --- Forward pass ---
        y_pred = predict(x)

        # --- Compute gradients ---
        # derivative of MSE w.r.t w = -2 * x * (y - y_pred)
        # derivative of MSE w.r.t b = -2 * (y - y_pred)
        dw = -2 * x * (y - y_pred)
        db = -2 * (y - y_pred)

        # --- Update parameters ---
        # w = w - learning_rate * gradient
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # --- Track loss ---
        total_loss += mse(y, y_pred)

    # Show progress
    avg_loss = total_loss / len(X)
    print(f"Epoch {epoch+1:02d}: w = {w:.4f}, b = {b:.4f}, loss = {avg_loss:.4f}")

# Step 7: Final results
print("\nTraining complete!")
print(f"Final model: y = {w:.2f}x + {b:.2f}")

# Test the trained model
test_x = 6
test_y = predict(test_x)
print(f"Prediction for x = {test_x}: y = {test_y:.2f}")
