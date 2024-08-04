"""This is the main file that executes the Counter-Example Guided Inductive Synthesis (CEGIS) for the supermartingale
training"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_params import n, m, A, min_bound, max_bound
from verify_function_z3 import verify_model
from verify_function_milp import verify_model_gurobi
from function_application import transition_kernel
from find_lipschitz import calculate_lipschitz_constant

from visualise import plot_weight_distribution, plot_output_distribution, plot_weight_changes, plot_loss_curve, plot_decision_boundary

from run_model import training_pairs, generate_model_params
results_doc = [element for element in training_pairs if not A.contains_point(element[0])]  # set removal

torch.set_printoptions(precision=20)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def sample_within_area(state):
    og_state = state
    flag = True
    while flag:
        state = og_state
        for xi in state:
            new = xi[0] + random.uniform(-2, 2)
            if new <= max_bound and new >= min_bound:
                xi[0] = new
        flag = A.contains_point(state)
    return state


# Example usage
input_size = n
hidden_size = 7
output_size = 1
epsilon = 0.001
learning_rate = 0.001
num_epochs = 100
max_iterations = 200


# Instantiate the neural network
model = SimpleNN(input_size, hidden_size, output_size)

# Initialize training data
X = torch.tensor(np.array([i[0] for i in results_doc]), dtype=torch.float32).squeeze(-1)
X_prime = torch.tensor(np.array([i[1] for i in results_doc]), dtype=torch.float32).squeeze(-1)


# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

weight_history = {'fc1': [], 'fc2': []}
loss_history = []
sat_history = []

# Training and verification loop
for iteration in range(max_iterations):
    print(f"Training iteration {iteration + 1}")
    print(len(X), len(X_prime))

    model.train()

    # Training loop
    for epoch in range(num_epochs):
        V_x = model(X)
        V_x_prime = torch.stack([model(i) for i in X_prime])
        E_V_x_prime = torch.mean(V_x_prime, dim=1)


        loss = torch.sum(F.relu(E_V_x_prime - V_x + epsilon))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(
                        f'{name} - Max Gradient: {param.grad.abs().max().item():.4f}, Mean Gradient: {param.grad.abs().mean().item():.4f}')

        weight_history['fc1'].append(model.fc1.weight.detach().mean(dim=1).numpy())
        weight_history['fc2'].append(model.fc2.weight.detach().mean(dim=1).numpy())
        loss_history.append(loss.item())
        condition = E_V_x_prime - V_x + epsilon
        satisfied = (condition <= 0).float()
        sat_history.append(torch.mean(satisfied).item())

        # if epoch % 100 == 0:
        #     plot_weight_distribution(model, epoch)
        #     plot_output_distribution(model, X, epoch)
        #     if X.shape[1] == 2:  # Only for 2D input
        #         plot_decision_boundary(model, X, epoch)

    # scheduler.step(loss)

    # Evaluation
    model.eval()
    with torch.no_grad():
        V_x = model(X)
        V_x_prime = torch.stack([model(i) for i in X_prime])
        E_V_x_prime = torch.mean(V_x_prime, dim=1)
        final_loss = torch.sum(F.relu(E_V_x_prime - V_x + epsilon))
        condition = E_V_x_prime - V_x + epsilon
        satisfied = (condition <= 0).float()
        satisfaction_rate = torch.mean(satisfied).item()
        print(f'Final Loss: {final_loss.item()}')
        print(f"Condition satisfaction rate: {satisfaction_rate:.2%}")

    # Verification step
    model_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_weights[name] = param.data.numpy()

    C, _, _, B, _, _, r = generate_model_params(n, m)
    is_sat, counter_example = verify_model(input_size, hidden_size,
                                           A, C, B, r, epsilon,
                                           model_weights['fc1.weight'], model_weights['fc2.weight'],
                                           model_weights['fc1.bias'], model_weights['fc2.bias'])
    # is_sat, counter_example = verify_model_gurobi(input_size, hidden_size,
    #                                        A, C, B, r, epsilon,
    #                                        model_weights['fc1.weight'], model_weights['fc2.weight'],
    #                                        model_weights['fc1.bias'], model_weights['fc2.bias'])
    if not is_sat:
        print("Verification successful. Model is correct.")
        break
    else:
        print(f"Verification failed. Retraining with counter-example: {counter_example}")

        counter_examples = np.array([sample_within_area(np.array([counter_example]).T) for _ in range(30)])
        # Add counter-example to training data
        X = torch.cat([X, torch.tensor(counter_examples, dtype=torch.float32).squeeze(-1)])
        for example in counter_examples:
            X_prime_new = torch.stack([torch.tensor(transition_kernel(example, C, B, r), dtype=torch.float32)])
            X_prime = torch.cat([X_prime, X_prime_new.squeeze(-1)])
        print(len(X), len(X_prime))

    if iteration % 5 == 0:
        plot_weight_changes(weight_history)
        plot_loss_curve(loss_history, sat_history)

if iteration == max_iterations - 1:
    print("Max iterations reached. Model could not be verified.")

print("\n\n")
with torch.no_grad():
    V_x = model(X)
    V_x_prime = torch.stack([model(i) for i in X_prime])
    E_V_x_prime = torch.mean(V_x_prime, dim=1)
    final_loss = torch.sum(F.relu(E_V_x_prime - V_x + epsilon))
    condition = E_V_x_prime - V_x + epsilon
    satisfied = (condition <= 0).float()
    satisfaction_rate = torch.mean(satisfied).item()
    print(f'Final Loss: {final_loss.item()}')
    print(f"Condition satisfaction rate: {satisfaction_rate:.2%}")

    print(model_weights)

    print("Lipschitz constant:", calculate_lipschitz_constant(model_weights['fc1.weight'],
                                           model_weights['fc1.bias'], model_weights['fc2.weight'],
                                           model_weights['fc2.bias']))

    torch.save(model.state_dict(), 'model_weights.pth')

plot_weight_changes(weight_history)
plot_loss_curve(loss_history, sat_history)


