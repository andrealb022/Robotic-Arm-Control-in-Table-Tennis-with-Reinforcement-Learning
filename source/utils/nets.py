"""
Course: Machine Learning 2023/2024
Students:
Alberti Andrea    0622702370    a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355    c.attianese13@studenti.unisa.it
Capaldo Vincenzo  0622702347    v.capaldo7@studenti.unisa.it
Esposito Paolo    0622702292    p.esposito57@studenti.unisa.it
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants for weight and bias initialization
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

# Utility function for initializing network weights
def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing weights of actor and critic networks"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    # Calculate the range for uniform initialization
    w = 1. / np.sqrt(fan_in)
    # Initialize the weights uniformly between -w and w
    nn.init.uniform_(tensor, -w, w)

# Class for the Actor neural network
class Actor(nn.Module):
    def __init__(self, in_state, hidden_size, n_actions):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        # Define the network layers
        self.fc1 = nn.Linear(in_state, self.hidden_size[0])  # First fully connected layer
        self.fc2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])  # Second fully connected layer
        self.fc3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])  # Third fully connected layer
        self.out = nn.Linear(self.hidden_size[2], n_actions)  # Output layer

        # Layer normalization
        self.bn1 = nn.LayerNorm(self.hidden_size[0])  # Normalize the first layer
        self.bn2 = nn.LayerNorm(self.hidden_size[1])  # Normalize the second layer
        self.bn3 = nn.LayerNorm(self.hidden_size[2])  # Normalize the third layer

        # Initialize the weights of the layers
        fan_in_uniform_init(self.fc1.weight)
        fan_in_uniform_init(self.fc1.bias)
        fan_in_uniform_init(self.fc2.weight)
        fan_in_uniform_init(self.fc2.bias)
        fan_in_uniform_init(self.fc3.weight)
        fan_in_uniform_init(self.fc3.bias)
        nn.init.uniform_(self.out.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.out.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, in_state):
        x = in_state
        # Layer 1
        x = self.fc1(x)  # Pass input through the first fully connected layer
        x = self.bn1(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation function

        # Layer 2
        x = self.fc2(x)  # Pass through the second fully connected layer
        x = self.bn2(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation function

        # Layer 3
        x = self.fc3(x)  # Pass through the third fully connected layer
        x = self.bn3(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation function

        # Output layer
        actions = torch.tanh(self.out(x))  # Pass through the output layer and apply Tanh activation function
        return actions  # Return the computed actions

# Class for the Critic neural network
class Critic(nn.Module):
    def __init__(self, in_state, hidden_size, n_actions):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        # Layer 1
        self.fc1 = nn.Linear(in_state, self.hidden_size[0])  # First fully connected layer
        self.bn1 = nn.LayerNorm(self.hidden_size[0])  # Normalize the first layer

        # Layer 2
        # The actions will be inserted in this layer
        self.fc2 = nn.Linear(self.hidden_size[0] + n_actions, self.hidden_size[1])  # Second fully connected layer with action concatenation
        self.bn2 = nn.LayerNorm(self.hidden_size[1])  # Normalize the second layer

        self.fc3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])  # Third fully connected layer
        self.bn3 = nn.LayerNorm(self.hidden_size[2])  # Normalize the third layer
        # Output layer (single value)
        self.out = nn.Linear(self.hidden_size[2], 1)  # Output layer

        # Initialize the weights of the layers
        fan_in_uniform_init(self.fc1.weight)
        fan_in_uniform_init(self.fc1.bias)
        fan_in_uniform_init(self.fc2.weight)
        fan_in_uniform_init(self.fc2.bias)
        fan_in_uniform_init(self.fc3.weight)
        fan_in_uniform_init(self.fc3.bias)
        nn.init.uniform_(self.out.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.out.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, in_state, actions):
        in_state = in_state.float()
        actions = actions.float()
        x = in_state
        # Layer 1
        x = self.fc1(x)  # Pass input through the first fully connected layer
        x = self.bn1(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation function

        # Layer 2
        x = torch.cat((x, actions), 1)  # Concatenate state with actions
        x = self.fc2(x)  # Pass through the second fully connected layer
        x = self.bn2(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation function

        # Layer 3
        x = self.fc3(x)  # Pass through the third fully connected layer
        x = self.bn3(x)  # Apply layer normalization
        x = F.relu(x)  # Apply ReLU activation function

        # Output layer
        v = self.out(x)  # Pass through the output layer
        return v  # Return the Q value

# Class for the Supervised Neural Network
class SupervisedNeuralNetwork(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(SupervisedNeuralNetwork, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(dim_input, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 64)  # Third fully connected layer
        self.fc4 = nn.Linear(64, dim_output)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Pass input through the first layer and apply ReLU
        x = torch.relu(self.fc2(x))  # Pass through the second layer and apply ReLU
        x = torch.relu(self.fc3(x))  # Pass through the third layer and apply ReLU
        x = self.fc4(x)  # Pass through the output layer
        return x  # Return the computed output
