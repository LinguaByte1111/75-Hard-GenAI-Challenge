# Simple Adam Optimizer Example (from scratch)
# Goal: Fit a line y = w*x + b to data using Adam

import math

# Step 1: Create sample data (y = 2x + 1)
X = [1, 2, 3, 4, 5]
Y = [2 * x + 1 for x in X]

# Step 2: Initialize parameters
w = 0.0   # slope
b = 0.0   # intercept

# Step 3: Hyperparameters for Adam
learning_rate = 0.01
beta1 = 0.9     # exponential decay rate for first moment (mean)
beta2 = 0.999   # exponential decay rate for second moment (variance)
epsilon = 1e-8  # small constant to avoid division by zero
epochs = 50

# Step 4: Initialize moment variables
m_w, v_w = 0.0, 0.0  # first & second moment for w
m_b, v_b = 0.0, 0.0  # first & second moment for b

# Step 5: Define helper functions
def predict(x):
    return w * x + b

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

# Step 6: Training loop using Adam
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        # --- Forward pass ---
        y_pred = predict(x)
        loss = mse(y, y_pred)

        # --- Compute gradients ---
        dw = -2 * x * (y - y_pred)
        db = -2 * (y - y_pred)

        # --- Update biased first moment estimate (mean) ---
        m_w = beta1 * m_w + (1 - beta1) * dw
        m_b = beta1 * m_b + (1 - beta1) * db

        # --- Update biased second raw moment estimate (variance) ---
        v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
        v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

        # --- Compute bias-corrected moment estimates ---
        m_w_hat = m_w / (1 - beta1 ** (epoch + 1))
        m_b_hat = m_b / (1 - beta1 ** (epoch + 1))
        v_w_hat = v_w / (1 - beta2 ** (epoch + 1))
        v_b_hat = v_b / (1 - beta2 ** (epoch + 1))

        # --- Parameter updates ---
        w = w - learning_rate * m_w_hat / (math.sqrt(v_w_hat) + epsilon)
        b = b - learning_rate * m_b_hat / (math.sqrt(v_b_hat) + epsilon)

        total_loss += loss

    avg_loss = total_loss / len(X)
    print(f"Epoch {epoch+1:02d}: w = {w:.4f}, b = {b:.4f}, loss = {avg_loss:.4f}")

# Step 7: Final results
print("\nTraining complete with Adam!")
print(f"Final model: y = {w:.2f}x + {b:.2f}")

# Test the trained model
test_x = 6
test_y = predict(test_x)
print(f"Prediction for x = {test_x}: y = {test_y:.2f}")
