import numpy as np
from model_params import n, A, C, B, V_threshold, D, p

results_doc = []
for _ in range(0, 50):
    from run_model import training_pairs
    results_doc = results_doc + [element for element in training_pairs if not A.contains_point(element[0])] # set removal
print(len(results_doc))

### ----- Neural Network stuff ------ ###

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = F.relu(self.fc2(x))
        return x


# Custom loss function
def custom_loss(V_x, E_V_x_prime, epsilon):
    return torch.sum(F.relu(E_V_x_prime - V_x + epsilon))


# Example usage
input_size = n
hidden_size = 5
output_size = 1
epsilon = 0.000001
learning_rate = 0.0001
num_epochs = 500

# Instantiate the neural network
model = SimpleNN(input_size, hidden_size, output_size)

X = torch.tensor(np.array([i[0] for i in results_doc]), dtype=torch.float32).squeeze(-1)

X_prime = torch.tensor(np.array([i[1] for i in results_doc]), dtype=torch.float32).squeeze(-1)

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    V_x = model(X)
    V_x_prime = torch.stack([model(i) for i in X_prime])
    E_V_x_prime = torch.mean(V_x_prime, dim=1)

    # Compute loss
    loss = custom_loss(V_x, E_V_x_prime, epsilon)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    V_x = model(X)
    V_x_prime = torch.stack([model(i) for i in X_prime])
    E_V_x_prime = torch.mean(V_x_prime, dim=1)
    final_loss = custom_loss(V_x, E_V_x_prime, epsilon)
    print(f'Final Loss: {final_loss.item():.4f}')

# model_weights = {}
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         model_weights[name] = param.data.numpy()
