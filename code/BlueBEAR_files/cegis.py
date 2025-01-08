"""This is the main file that executes Counter-Example Guided Inductive Synthesis (CEGIS) for training the
supermartingale on the cascade system"""
import random
import time

import numpy as np
from numpy.linalg import svd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import partial

from property_tests import is_in_invariant_set
from MAB_algorithm import mab_algorithm


def v_x(x, weights, biases):
    """
    Compute forward pass through a neural network with arbitrary number of layers.

    Parameters:
    x (np.ndarray): Input vector of shape (n, 1)
    weights (list): List of weight matrices [W1, W2, ..., Wn]
    biases (list): List of bias vectors [B1, B2, ..., Bn]

    Returns:
    float: Output of the neural network
    """
    # Input validation
    assert len(weights) == len(biases), "Number of weight matrices must match number of bias vectors"
    assert len(weights) >= 1, "Must have at least one layer"

    # Forward pass
    activation = x
    for W, B in zip(weights, biases):
        # Compute layer output
        z = np.dot(W, activation) + B
        activation = np.maximum(0, z)  # ReLU activation

    return activation[0][0]


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, width, output_size, depth):
        """
        A neural network with variable depth.
        :param input_size: Number of input features.
        :param width: Number of units in each hidden layer.
        :param output_size: Number of output features.
        :param depth: Number of hidden layers.
        """
        super(SimpleNN, self).__init__()

        # Input layer
        layers = [nn.Linear(input_size, width)]

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(width, width))

        # Output layer
        layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_size))

        # Store the layers in a Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def find_supermartingale(domain_bounds, num_players, transition_kernel, lipschitz_P, network_width, network_depth, confidence, reward_boundary, epsilon=0.5, learning_rate=0.01, training_its=150, verifier_attempts=2000000):
    torch.set_printoptions(precision=20)
    np.set_printoptions(threshold=20)

    network_run_times = []
    network_it_nums = []
    verifier_run_times = []
    verifier_it_nums = []
    verifier_avg_it_times = []
    verifier_tree_depth = []
    verifier_regions_nums = []
    alpha_history = []
    beta_history = []
    loss_history = []


    def sample_within_area(state):
        og_state = [xi.copy() for xi in state]  # Copy the original state to avoid mutating it
        flag = False

        while not flag:
            state = [xi.copy() for xi in og_state]  # Reset state to original
            for i, xi in enumerate(state):
                # Use the bounds for the i-th dimension
                min_bound, max_bound = domain_bounds[i]
                new = xi[0] + random.uniform(-0.01, 0.01)
                if min_bound <= new <= max_bound:  # Check if new value is within the bounds
                    xi[0] = new
            flag = not is_in_invariant_set(state)  # Ensure state is outside the invariant set

        return state

    # Define total loss with range punishment
    def total_loss(V_x, E_V_x_prime, lambda_):
        # Original task loss
        task_loss = torch.sum(F.relu(E_V_x_prime - V_x + epsilon))

        # Compute range penalty for V_x (current outputs)
        range_loss = 0  #TODO

        # Combine task loss and range penalty
        return task_loss + lambda_ * range_loss

    # Initialize the neural network
    model = SimpleNN(input_size=num_players, width=network_width, output_size=1, depth=network_depth)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Initial training examples
    results_doc = [(ex, transition_kernel(ex)) for ex in [np.array([np.random.uniform(min_bound, max_bound) for min_bound, max_bound in domain_bounds]).reshape(-1, 1) for _ in range(500)] if not is_in_invariant_set(ex)]
    X = torch.tensor(np.array([i[0] for i in results_doc]), dtype=torch.float64).squeeze(-1)
    X_prime = torch.tensor(np.array([i[1] for i in results_doc]), dtype=torch.float64).squeeze(-1)


    #weight_history = {'fc1': [], 'fc2': []}

    # Training and verification loop
    for iteration in range(verifier_attempts):
        print(f"Training iteration {iteration + 1}")
        network_start_time = time.process_time()
        # Training loop
        model.train()
        epoch = 0
        regulariser_strength = 0 #TODO
        while epoch < training_its:
            V_x = model(X)
            V_x_prime = torch.stack([model(i) for i in X_prime])
            E_V_x_prime = torch.mean(V_x_prime, dim=1)

            # Use the new total loss function with range penalty
            loss = total_loss(V_x, E_V_x_prime, regulariser_strength)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 50 == 0:
                print(f'Epoch [{epoch}/{training_its}], Loss: {loss.item()}')
            loss_history.append(loss.item())
            condition = E_V_x_prime - V_x + epsilon
            satisfied = torch.mean((condition <= 0).float()).item()

            epoch += 1
            if satisfied == 1:
                print(f'Epoch [{epoch}/{training_its}], Loss: {loss.item()}')
                break

        # Verification
        network_run_times.append(network_start_time - time.process_time())
        network_it_nums.append(epoch)
        model.eval()
        #print(f'Final Loss: {loss.item()}')
        #print(f"Condition satisfaction rate: {satisfied:.2%}")

        with torch.no_grad():

            # overestimation of the lipschitz constant
            L = 1
            model_weights = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    model_weights[name] = param.data.numpy()
                    if len(param.shape) == 2:  # Check if it is a weight matrix (not a bias vector)
                        _, singular_values, _ = svd(param, full_matrices=False)
                        spectral_norm = np.max(singular_values)
                        L *= spectral_norm

            print(model_weights.keys())

            V = partial(v_x, weights=[model_weights[f'network.{i*2}.weight'] for i in range(network_depth + 1)],
                        biases=[model_weights[f'network.{i*2}.bias'] for i in range(network_depth + 1)])

            # import inspect
            # # Get the signature of the function
            # signature = inspect.signature(reward_boundary)
            #
            # # Print the signature
            # print("Expected arguments:", signature)

            ub, max_point = reward_boundary(weights=[model_weights[f'network.{i*2}.weight'] for i in range(network_depth + 1)],
                                            biases=[model_weights[f'network.{i*2}.bias'] for i in range(network_depth + 1)],
                                            upper=True)

            lb, min_point = reward_boundary(weights=[model_weights[f'network.{i*2}.weight'] for i in range(network_depth + 1)],
                                            biases=[model_weights[f'network.{i*2}.bias'] for i in range(network_depth + 1)],
                                            upper=False)

            verify_start_time = time.time()
            is_sat, counter_example, its, avg_time, tree_depth, num_regions = mab_algorithm(
                initial_bounds=domain_bounds,
                dynamics=transition_kernel,
                certificate=V,
                lipschitz=L * lipschitz_P + L,
                reward_range=abs(ub - lb),
                max_iterations=1000000000,
                tolerance=epsilon * 0.5,
                confidence=confidence
            )
            time.sleep(0.6)
            verifier_run_times.append(verify_start_time - time.time())
            verifier_it_nums.append(its)
            verifier_avg_it_times.append(avg_time)
            verifier_tree_depth.append(tree_depth)
            verifier_regions_nums.append(num_regions)
            alpha_history.append(L * lipschitz_P + L)
            beta_history.append(abs(ub - lb))

        if is_sat:
            break
        else:
            #counter_examples = np.array([sample_within_area(np.array([counter_example]).T) for _ in range(10)])
            X = torch.cat([X, torch.tensor([counter_example], dtype=torch.float64)])
            X_prime_new = torch.stack([torch.tensor(transition_kernel(np.array([counter_example]).T), dtype=torch.float64)])
            X_prime = torch.cat([X_prime, X_prime_new.squeeze(-1)])

    if iteration == verifier_attempts - 1:
        print("Max iterations reached. Model could not be verified.")
        return False, network_run_times, network_it_nums, verifier_run_times, verifier_it_nums, verifier_avg_it_times, verifier_tree_depth, verifier_regions_nums, alpha_history, beta_history, loss_history, model_weights
    else:
        return True, network_run_times, network_it_nums, verifier_run_times, verifier_it_nums, verifier_avg_it_times, verifier_tree_depth, verifier_regions_nums, alpha_history, beta_history, loss_history, model_weights
