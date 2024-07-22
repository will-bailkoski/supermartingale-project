import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_params import n, m, A
from verify_function_z3 import verify_model
from transition_kernel import transition_kernel

from run_model import training_pairs, generate_model_params
results_doc = [element for element in training_pairs if not A.contains_point(element[0])]  # set removal


# Define the neural network (unchanged)
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# Custom loss function (unchanged)
def custom_loss(V_x, E_V_x_prime, epsilon):
    return torch.sum(F.relu(E_V_x_prime - V_x + epsilon))


# Example usage
input_size = n
hidden_size = 10
output_size = 1
epsilon = 0.00000001
learning_rate = 0.001
num_epochs = 500
max_iterations = 50

# Instantiate the neural network
model = SimpleNN(input_size, hidden_size, output_size)

# Initialize training data
X = torch.tensor(np.array([i[0] for i in results_doc]), dtype=torch.float32).squeeze(-1)
X_prime = torch.tensor(np.array([i[1] for i in results_doc]), dtype=torch.float32).squeeze(-1)


# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training and verification loop
for iteration in range(max_iterations):
    print(f"Training iteration {iteration + 1}")
    print(len(X), len(X_prime))

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
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        V_x = model(X)
        V_x_prime = torch.stack([model(i) for i in X_prime])
        E_V_x_prime = torch.mean(V_x_prime, dim=1)
        final_loss = custom_loss(V_x, E_V_x_prime, epsilon)
        print(f'Final Loss: {final_loss.item():.6f}')

    # Verification step
    model_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_weights[name] = param.data.numpy()

    C, D, p, B, V_threshold, X_initial, r = generate_model_params(n, m)
    is_sat, counter_example = verify_model(input_size, hidden_size,
                                           A, C, D, p, B, V_threshold, r, epsilon,
                                           model_weights['fc1.weight'], model_weights['fc2.weight'],
                                           model_weights['fc1.bias'], model_weights['fc2.bias'])
    if not is_sat:
        print("Verification successful. Model is correct.")
        break
    else:
        print(f"Verification failed. Retraining with counter-example: {counter_example}")
        # Add counter-example to training data
        X = torch.cat([X, torch.tensor([counter_example], dtype=torch.float32)])
        X_prime_new = torch.stack([torch.tensor(transition_kernel(np.array([counter_example]).T, C, B, r), dtype=torch.float32)])
        X_prime = torch.cat([X_prime, X_prime_new.squeeze(-1)])

if iteration == max_iterations - 1:
    print("Max iterations reached. Model could not be verified.")

# The final model is now stored in the 'model' variable
