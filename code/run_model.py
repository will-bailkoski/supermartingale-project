import random
import numpy as np
import matplotlib.pyplot as plt

from transition_kernel import transition_kernel

training_pairs = []

def run_simulation_X(C, r, B, X_initial, time_steps):

    # Initialize the array to store equity values over time
    X = [None] * (time_steps)
    X[0] = X_initial

    for t in range(1, time_steps):
        next_possible_states = transition_kernel(X[t-1], C, B, r)
        training_pairs.append((X[t-1], next_possible_states))
        X[t] = np.array(random.choice(next_possible_states))
    return X

from model_params import n, C, D, p, B, V_threshold, X_initial
r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)

# Equilibrium check???
# a = np.linalg.inv(np.eye(len(C)) - C)
# b = r - beta
# if np.dot(a, b)[0][0] >= 0 and np.dot(a, b)[1][0] >= 0:  # equilibrium check



time_steps = 10

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



