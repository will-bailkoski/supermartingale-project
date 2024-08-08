import random
import numpy as np
import matplotlib.pyplot as plt

from function_application import transition_kernel

training_pairs = []


def generate_model_params(n, m):
    # C = np.random.uniform(0, 0.01, (n, n))
    # np.fill_diagonal(C, 0)

    C = np.array([[0., 0.0052315],
                  [0.00117426, 0.]])

    D = np.array([[0.06] * n] * m).T
    p = np.array([[10] * m]).T

    beta = np.array([0.4] * n)  # definable
    B = np.diag(beta)  # Failure costs
    beta = np.array([beta]).T

    V_threshold = np.array([[5] * n]).T  # Failure thresholds                           # definable
    X_initial = np.array([np.random.uniform(0, 30, n)]).T  # definable
    r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)

    return C, D, p, B, V_threshold, X_initial, r


def run_simulation_X(C, r, B, X_initial, time_steps):
    # Initialize the array to store equity values over time
    X = [None] * (time_steps)
    X[0] = X_initial

    for t in range(1, time_steps):
        next_possible_states = transition_kernel(X[t - 1], C, B, r)
        training_pairs.append((X[t - 1], next_possible_states))
        X[t] = np.array(random.choice(next_possible_states))
    return X


from model_params import n, m, time_steps

for i in range(500):
    C, D, p, B, V_threshold, X_initial, r = generate_model_params(n, m)

    # Equilibrium check
    # a = np.linalg.inv(np.eye(len(C)) - C)
    # beta = np.array([0.4] * n)
    # b = r - beta
    # if np.dot(a, b)[0][0] <= 0 and np.dot(a, b)[1][0] <= 0:
    #     print("equilibrium exists")
    #     if np.dot(a, r)[0][0] <= 0 and np.dot(a, r)[1][0] <= 0:
    #         print("equilibrium is unique")

    X = run_simulation_X(C, r, B, X_initial, time_steps + 1)

    results = np.array(X).T[0].T

# Plot the results
plt.figure(figsize=(10, 10))
plt.plot(results)
plt.axhline(y=0, color='r', linestyle='--', label='Failure Threshold')
plt.xlabel('Time')
plt.ylabel('Equity Values')
plt.legend([str(i + 1) for i in range(0, n)] + ['Failure Threshold'])

plt.grid(linestyle='-', linewidth='0.5', color='grey')

plt.show()
