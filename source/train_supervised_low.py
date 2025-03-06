"""
Course: Machine Learning 2023/2024
Students:
Alberti Andrea    0622702370    a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355    c.attianese13@studenti.unisa.it
Capaldo Vincenzo  0622702347    v.capaldo7@studenti.unisa.it
Esposito Paolo    0622702292    p.esposito57@studenti.unisa.it
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.nets import SupervisedNeuralNetwork

# Read the dataset
dataset = pd.read_csv('saved_dataset/dataset_low.csv', header=None)

# Manually assign column names
dataset.columns = ['y', 'z', 'j3', 'j5', 'j7']

# Display the first few rows of the dataset to understand its structure
print(dataset.head())
print(dataset.shape)

# Prepare the data
X = dataset[['y', 'z']].values  # Use y and z as the input
y = dataset[['j3', 'j5', 'j7']].values  # Use j3, j5, and j7 as the output

# Split the data into training (70%), validation (15%), and test sets (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)  # Fit and transform the training data
X_val = scaler_X.transform(X_val)  # Transform the validation data
X_test = scaler_X.transform(X_test)  # Transform the test data

y_train = scaler_y.fit_transform(y_train)  # Fit and transform the training data
y_val = scaler_y.transform(y_val)  # Transform the validation data
y_test = scaler_y.transform(y_test)  # Transform the test data

# Save the scalers for future use
joblib.dump(scaler_X, 'saved_models/supervised/low_scaler_X.pkl')
joblib.dump(scaler_y, 'saved_models/supervised/low_scaler_y.pkl')

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Initialize the neural network model
model = SupervisedNeuralNetwork(2, 3)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model with loss function monitoring
epochs = 2000
early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear the gradients

    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute the training loss
    loss.backward()  # Backward pass to compute gradients
    optimizer.step()  # Update the model parameters

    train_losses.append(loss.item())  # Record the training loss

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_outputs = model(X_val)  # Forward pass on validation data
        val_loss = criterion(val_outputs, y_val)  # Compute the validation loss
        val_losses.append(val_loss.item())  # Record the validation loss

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    # Early stopping mechanism
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), 'saved_models/supervised/model_low.pth')  # Save the best model
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping")
            break

# Load the best saved model
model.load_state_dict(torch.load('saved_models/supervised/model_low.pth'))

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test)  # Forward pass on test data
    test_loss = criterion(test_outputs, y_test)  # Compute the test loss
    print(f"Mean Squared Error on test set: {test_loss.item()}")

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
