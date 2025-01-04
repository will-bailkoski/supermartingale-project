"""This is the main file that executes Counter-Example Guided Inductive Synthesis (CEGIS) for training the
supermartingale on the cascade system"""
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import partial

from generate_cascade_params import is_in_invariant_set
from MAB_algorithm import mab_algorithm


def e_r_x(x, C, B, r, W1, W2, B1, B2, epsilon):
    return e_v_p_x(x, C, B, r, W1, W2, B1, B2) - v_x(x, W1, W2, B1, B2) - epsilon


def e_v_p_x(x, C, B, r, W1, W2, B1, B2):

    # assert x.shape == (n, 1), f"Input x must have shape (2, 1), but has shape {x.shape}"
    # assert W1.shape == (h, n), f"Input x must have shape (7, 2), but has shape {W1.shape}"
    # assert B1.shape == (h, 1), f"Input x must have shape (7, 1), but has shape {B1.shape}"
    # assert W2.shape == (1, h), f"Input x must have shape (1, 7), but has shape {W2.shape}"
    # assert B2.shape == (1, 1), f"Input x must have shape (1, 1), but has shape {B2.shape}"

    p_x = transition_kernel(x, C, B, r)
    v_p_xs = [v_x(i, W1, W2, B1, B2) for i in p_x]
    return sum(v_p_xs) / 4


def v_x(x, W1, W2, B1, B2):

    # assert x.shape == (n, 1), f"Input x must have shape (2, 1), but has shape {x.shape}"
    # assert W1.shape == (h, n), f"Input x must have shape (7, 2), but has shape {W1.shape}"
    # assert B1.shape == (h, 1), f"Input x must have shape (7, 1), but has shape {B1.shape}"
    # assert W2.shape == (1, h), f"Input x must have shape (1, 7), but has shape {W2.shape}"
    # assert B2.shape == (1, 1), f"Input x must have shape (1, 1), but has shape {B2.shape}"

    z1 = np.dot(W1, x) + B1
    a1 = np.maximum(0, z1)
    z2 = np.dot(W2, a1) + B2
    return np.maximum(0, z2)[0][0]



# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, width):
        """
        A neural network with variable depth.
        :param input_size: Number of input features.
        :param hidden_size: Number of units in each hidden layer.
        :param output_size: Number of output features.
        :param width: Number of hidden layers.
        """
        super(SimpleNN, self).__init__()

        # Input layer
        layers = [nn.Linear(input_size, hidden_size)]

        # Hidden layers
        for _ in range(width - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))

        # Store the layers in a Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def find_supermartingale(domain_bounds, num_players, transition_kernel, network_width, network_depth, confidence, reward_boundary, epsilon = 0.5, learning_rate = 0.01, training_its = 150, verifier_attempts = 2000000):
    torch.set_printoptions(precision=20)
    np.set_printoptions(threshold=20)


    def sample_within_area(state):
        og_state = state
        flag = False
        while not flag:
            state = og_state
            for xi in state:
                new = xi[0] + random.uniform(-0.01, 0.01)
                if max_bound >= new >= min_bound: #TODO
                    xi[0] = new
            flag = not is_in_invariant_set(state)
        return state


    # Define total loss with range punishment
    def total_loss(V_x, E_V_x_prime, lambda_):
        # Original task loss
        task_loss = torch.sum(F.relu(E_V_x_prime - V_x + epsilon))

        # Compute range penalty for V_x (current outputs)
        range_loss = 0  #TODO

        # Combine task loss and range penalty
        return task_loss + lambda_ * range_loss

    def test_model_with_params(model_state, input_data):
        # Create a new instance of the model
        model = SimpleNN(input_size=num_players, hidden_size=network_depth, output_size=1, width=network_width)

        # Load the provided state into the new model
        model.load_state_dict(model_state)

        # Set the model to evaluation mode
        model.eval()

        # Pass input data through the model
        with torch.no_grad():
            outputs = model(input_data)

        return outputs

    # Partial function that creates a test function with the current network's parameters
    def create_partial_test_function(model):
        # Get a copy of the model's current state
        model_state = model.state_dict()

        # Create a partial function that binds the model_state
        partial_test = partial(test_model_with_params, model_state)
        return partial_test



    # Initialize the neural network
    model = SimpleNN(input_size=num_players, hidden_size=network_depth, output_size=1, width=network_width)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Initial training examples
    results_doc = [(ex, transition_kernel(ex)) for ex in [np.array([np.random.uniform(min_bound, max_bound) for min_bound, max_bound in domain_bounds]).reshape(-1, 1) for _ in range(500)] if not is_in_invariant_set(ex)]
    X = torch.tensor(np.array([i[0] for i in results_doc]), dtype=torch.float64).squeeze(-1)
    X_prime = torch.tensor(np.array([i[1] for i in results_doc]), dtype=torch.float64).squeeze(-1)


    weight_history = {'fc1': [], 'fc2': []}
    loss_history = []
    sat_history = []

    # Training and verification loop
    for iteration in range(verifier_attempts):
        print(f"Training iteration {iteration + 1}")
        #
        # if iteration % 50 == 0:
        #     loss_history = []
        #     sat_history = []
        #     weight_history = {'fc1': [], 'fc2': []}


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
            weight_history['fc1'].append(model.fc1.weight.detach().mean(dim=1).numpy())
            weight_history['fc2'].append(model.fc2.weight.detach().mean(dim=1).numpy())
            loss_history.append(loss.item())
            condition = E_V_x_prime - V_x + epsilon
            satisfied = torch.mean((condition <= 0).float()).item()
            sat_history.append(satisfied)

            epoch += 1
            if satisfied == 1:
                epoch = training_its + 100

        # Verification
        model.eval()
        print(f'Final Loss: {loss.item()}')
        print(f"Condition satisfaction rate: {satisfied:.2%}")

        with torch.no_grad():

            model_weights = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    model_weights[name] = param.data.numpy()

            V = partial(v_x, W1=W1, W2=W2, B1=B1, B2=B2)

            ub, max_point = reward_boundary(model_weights['fc1.weight'], model_weights['fc2.weight'],
                                                        model_weights['fc1.bias'], model_weights['fc2.bias'],
                                                        upper=True)

            lb, min_point = reward_boundary(model_weights['fc1.weight'], model_weights['fc2.weight'],
                                                        model_weights['fc1.bias'], model_weights['fc2.bias'],
                                                        upper=False)


            is_sat, counter_example = mab_algorithm(
                initial_bounds=domain_bounds,
                dynamics=transition_kernel,
                certificate=V,
                lipschitz=L,
                reward_range=abs(ub - lb),
                max_iterations=10000000000,
                tolerance=epsilon * 0.5,
                confidence=confidence
            )
        if is_sat:
            print("Verification successful. Model is correct.")
            break
        else:
            print(f"Verification failed. Retraining with counter-example: {counter_example}")

            counter_examples = np.array([sample_within_area(np.array([counter_example]).T) for _ in range(10)])
            # Add counter-example to training data
            #print(torch.tensor(counter_examples, dtype=torch.float64).squeeze(-1))
            #print(torch.tensor(counter_example, dtype=torch.float64))
            X = torch.cat([X, torch.tensor([counter_example], dtype=torch.float64)])
            # X = torch.cat([X, torch.tensor([counter_example], dtype=torch.float64), torch.tensor(counter_examples, dtype=torch.float64).squeeze(-1)])
            # for example in counter_examples:
            #     X_prime_new = torch.stack([torch.tensor(transition_kernel(example, C, B, r), dtype=torch.float64)])
            #     X_prime = torch.cat([X_prime, X_prime_new.squeeze(-1)])
            X_prime_new = torch.stack([torch.tensor(transition_kernel(np.array([counter_example]).T, C, B, r), dtype=torch.float64)])
            X_prime = torch.cat([X_prime, X_prime_new.squeeze(-1)])

        if iteration % 10 == 0 or iteration == 3:
            plot_weight_changes(weight_history)
            plot_loss_curve(loss_history, sat_history)

    if iteration == verifier_attempts - 1:
        print("Max iterations reached. Model could not be verified.")
        return

    # export params upon termination
    params = {
            'bounds': (min_bound, max_bound),
            'epsilon': epsilon,
            'verified': not is_sat,
            'C': C,
            'B': B,
            'r': r,
            'network weights': model_weights
        }

        # Save the dictionary to a file
        with open(f"cascade/{n}_players/params.pkl", 'wb') as f:
            pickle.dump(params, f)

        # torch.save(model.state_dict(), 'model_weights.pth')
        # np.save('covariance.npy', C)
        # np.save('saved_r.npy', r)
        # print("\n\n")
        # for key in model_weights.keys():
        #     for x in model_weights[key]:
        #         print(x)
        #     print("\n")
        # for each in C:
        #     print([x for x in each])
        #
        # gurobi, _ = verify_model_milp(input_size, hidden_size,
        #                               C, B, r, epsilon,
        #                               model_weights['fc1.weight'], model_weights['fc2.weight'],
        #                               model_weights['fc1.bias'], model_weights['fc2.bias'])
        # z3, _ = verify_model_sat(input_size, hidden_size,
        #                          C, B, r, epsilon,
        #                          model_weights['fc1.weight'], model_weights['fc2.weight'],
        #                          model_weights['fc1.bias'], model_weights['fc2.bias'])
        #
        # print(gurobi, z3)
        #
        # if gurobi == z3:
        #     print("Models are the same!!")
        # else:
        #     print("Models arent the same")
        # assert not z3 and not gurobi, "model is not valid"
        #
        # W1 = model_weights['fc1.weight']
        # W2 = model_weights['fc2.weight']
        # B1 = np.array([model_weights['fc1.bias']]).T
        # B2 = np.array([model_weights['fc2.bias']])
        #
        #
        #
        # P = partial(transition_kernel, C=C, B=B, r=r)
        # V = partial(v_x, W1=W1, W2=W2, B1=B1, B2=B2)
        # R = partial(e_r_x, C=C, B=B, r=r, W1=W1, W2=W2, B1=B1, B2=B2, epsilon=epsilon)
        #
        # points = np.array([np.random.uniform(min_bound, max_bound, 10000) for i in range(n)]).T
        # grad_f = grad(R)
        # gradient_norms = np.array([np.linalg.norm(grad_f(np.array([point]).T)) for point in points])
        # L = np.max(gradient_norms)
        #
        # result = mab_algorithm(
        #     initial_bounds=[(min_bound, max_bound)] * n,
        #     dynamics=P,
        #     certificate=V,
        #     lipschitz=L,
        #     # calculate_lipschitz_constant(input_size, hidden_size, C, B, r,
        #     #                                        model_weights['fc1.weight'], model_weights['fc2.weight'],
        #     #                                        model_weights['fc1.bias'], model_weights['fc2.bias'],
        #     #                                        (min_bound, max_bound), 0),
        #     beta=abs(find_reward_bound(input_size, hidden_size, C, B, r,
        #                                      model_weights['fc1.weight'], model_weights['fc2.weight'],
        #                                      model_weights['fc1.bias'], model_weights['fc2.bias'], epsilon, True)
        #              - find_reward_bound(input_size, hidden_size, C, B, r,
        #                                        model_weights['fc1.weight'], model_weights['fc2.weight'],
        #                                        model_weights['fc1.bias'], model_weights['fc2.bias'], epsilon)),
        #     verifier_attempts=10000000,
        #     epsilon=epsilon
        # )
        # print("\n\n\n\nWas the supermartingale validated?", result)
        # torch.save(model.state_dict(), 'model_weights.pth')
