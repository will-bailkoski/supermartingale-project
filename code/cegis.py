"""This is the main file that executes Counter-Example Guided Inductive Synthesis (CEGIS) for training the
supermartingale"""

import random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from MAB_algorithm import mab_algorithm
from find_lipschitz import calculate_lipschitz_constant
from find_lower import find_reward_lower_bound
from find_upper import find_reward_upper_bound
from function_application import transition_kernel, v_x
from model_params import n, C, B, r, min_bound, max_bound
from verify_function_milp import verify_model_gurobi
from verify_function_z3 import verify_model
from visualise import plot_weight_changes, plot_loss_curve

results_doc = [(ex, transition_kernel(ex, C, B, r)) for ex in [np.array([np.random.uniform(min_bound, max_bound, n)]).T for _ in range(500)] if not ex[0][0] < 0 and ex[1][0] < 0]

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
            if max_bound >= new >= min_bound:
                xi[0] = new
        flag = state[0][0] < 0 and state[1][0] < 0
    return state


input_size = n
hidden_size = 7
output_size = 1
epsilon = 0.5
learning_rate = 0.01
num_epochs = 100
max_iterations = 200

# Instantiate the neural network
model = SimpleNN(input_size, hidden_size, output_size)
model.double()

# Initialize training data
X = torch.tensor(np.array([i[0] for i in results_doc]), dtype=torch.float64).squeeze(-1)
X_prime = torch.tensor(np.array([i[1] for i in results_doc]), dtype=torch.float64).squeeze(-1)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

weight_history = {'fc1': [], 'fc2': []}
loss_history = []
sat_history = []

# Training and verification loop
for iteration in range(max_iterations):
    print(f"Training iteration {iteration + 1}")

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
        weight_history['fc1'].append(model.fc1.weight.detach().mean(dim=1).numpy())
        weight_history['fc2'].append(model.fc2.weight.detach().mean(dim=1).numpy())
        loss_history.append(loss.item())
        condition = E_V_x_prime - V_x + epsilon
        satisfied = torch.mean((condition <= 0).float()).item()
        sat_history.append(satisfied)


    # Evaluation
    model.eval()
    print(f'Final Loss: {loss.item()}')
    print(f"Condition satisfaction rate: {satisfied:.2%}")

    # Verification step
    model_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_weights[name] = param.data.numpy()

    # is_sat, counter_example = verify_model(input_size, hidden_size,
    #                                        A, C, B, r, epsilon,
    #                                        model_weights['fc1.weight'], model_weights['fc2.weight'],
    #                                        model_weights['fc1.bias'], model_weights['fc2.bias'])
    is_sat, counter_example = verify_model_gurobi(input_size, hidden_size,
                                                  C, B, r, epsilon,
                                                  model_weights['fc1.weight'], model_weights['fc2.weight'],
                                                  model_weights['fc1.bias'], model_weights['fc2.bias'])
    if not is_sat:
        print("Verification successful. Model is correct.")
        break
    else:
        print(f"Verification failed. Retraining with counter-example: {counter_example}")

        counter_examples = np.array([sample_within_area(np.array([counter_example]).T) for _ in range(50)])
        # Add counter-example to training data
        X = torch.cat([X, torch.tensor(counter_examples, dtype=torch.float32).squeeze(-1)])
        for example in counter_examples:
            X_prime_new = torch.stack([torch.tensor(transition_kernel(example, C, B, r), dtype=torch.float32)])
            X_prime = torch.cat([X_prime, X_prime_new.squeeze(-1)])

    if iteration % 10 == 0 or iteration == 3:
        plot_weight_changes(weight_history)
        plot_loss_curve(loss_history, sat_history)

if iteration == max_iterations - 1:
    print("Max iterations reached. Model could not be verified.")

print("\n\n")
print(model_weights)

gurobi, _ = verify_model_gurobi(input_size, hidden_size,
                                C, B, r, epsilon,
                                model_weights['fc1.weight'], model_weights['fc2.weight'],
                                model_weights['fc1.bias'], model_weights['fc2.bias'])
z3, _ = verify_model(input_size, hidden_size,
                     C, B, r, epsilon,
                     model_weights['fc1.weight'], model_weights['fc2.weight'],
                     model_weights['fc1.bias'], model_weights['fc2.bias'])

if gurobi == z3:
    print("Models are the same!!")
else:
    print("Models arent the same")
assert z3 and gurobi, "model is not valid"

P = partial(transition_kernel, C=C, B=B, r=r)
V = partial(v_x,
            W1=model_weights['fc1.weight'],
            W2=model_weights['fc2.weight'],
            B1=np.array([model_weights['fc1.bias']]).T,
            B2=np.array([model_weights['fc2.bias']]))

result = mab_algorithm(
    initial_bounds=[(min_bound, max_bound)] * n,
    dynamics=P,
    certificate=V,
    lipschitz=calculate_lipschitz_constant(input_size, hidden_size, C, B, r,
                                           model_weights['fc1.weight'], model_weights['fc2.weight'],
                                           model_weights['fc1.bias'], model_weights['fc2.bias'],
                                           (min_bound, max_bound), 0),
    beta=abs(find_reward_upper_bound(input_size, hidden_size, C, B, r,
                                     model_weights['fc1.weight'], model_weights['fc2.weight'],
                                     model_weights['fc1.bias'], model_weights['fc2.bias'])
             - find_reward_lower_bound(input_size, hidden_size, C, B, r,
                                       model_weights['fc1.weight'], model_weights['fc2.weight'],
                                       model_weights['fc1.bias'], model_weights['fc2.bias'])),
    max_iterations=50000
)
print("\n\n\n\n", result)
torch.save(model.state_dict(), 'model_weights.pth')

plot_weight_changes(weight_history)
plot_loss_curve(loss_history, sat_history)
