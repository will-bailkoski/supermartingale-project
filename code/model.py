import random

import numpy as np
import matplotlib.pyplot as plt

num_samples = 5
n = 20
m = 10

# for i in range(num_samples):
flag = False
while not flag:
    # Define the parameters
    C = []
    for i in range(n):
        C.append(np.random.uniform(0, 0.01, n))  # Example cross-holdings matrix
    C = np.array(C)
    np.fill_diagonal(C, 0)

    D = np.array([[0.03] * n] * m).T  # Market price of assets
    p = np.array([[10] * m]).T  # Initial market prices
    beta = np.array([0.4] * n)
    B = np.diag(beta)  # Failure costs
    beta = np.array([beta]).T
    V_threshold = np.array([[5] * n]).T  # Failure thresholds
    V_initial = np.array([np.random.uniform(0, 30, n)]).T  # Initial equity values

    X_initial = np.array([np.random.uniform(0, 30, n)]).T  # np.subtract(V_initial, V_threshold)
    r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)
    a = np.linalg.inv(np.eye(len(C)) - C)
    b = r - beta

    #if np.dot(a, b)[0][0] >= 0 and np.dot(a, b)[1][0] >= 0:  # equilibrium check
    flag = True


# Define the failure function phi
def phi(v_vbar):
    print(v_vbar)
    indicator = []
    for i in range(0, len(V_threshold)):
        if v_vbar[i][0] < 0:
            indicator.append([1])
        else:
            indicator.append([0])
    return indicator

# Stochasticity
def brownian_motion(state):

    for k in range(len(state)):
        if random.random() < 0.5:
            state[k] = state[k] - (0.1 * state[k])
        else:
            state[k] = state[k] + (0.1 * state[k])

    return state


def run_simulation(C, D, p, B, V_threshold, V_initial, time_steps):

    # Initialize the array to store equity values over time
    V = [None] * (time_steps)
    V[0] = V_initial.T

    for t in range(1, time_steps):
        #print(np.shape(np.dot(B, phi(V[t-1].T, V_threshold))))
        V[t] = (np.dot(C, V[t-1].T) + np.dot(D, p) - np.dot(B, phi(V[t-1].T - V_threshold))).T
        #V[t] = np.add(np.multiply(C, V[t - 1].T), np.subtract(np.multiply(D, p), np.multiply(B, phi(V[t - 1].T, V_threshold))).T)

    return V


def run_simulation_X(C, r, B, X_initial, time_steps):

    # Initialize the array to store equity values over time
    X = [None] * (time_steps)
    X[0] = X_initial

    for t in range(1, time_steps):


        Cx = np.dot(C, X[t-1])
        Bphi = np.dot(B, phi(X[t-1]))

        X[t] = Cx + r - Bphi  # basic equation in x(t)
        print("CYCLE")
        print(X[t])
        X[t] = brownian_motion(X[t])  # add noise



        # X[t] = np.array([brownian_motion((np.dot(C, X[t-1].T) + r - np.dot(B, phi(X[t-1].T))).T)[0]])

    return X


time_steps = 6
#results = run_simulation(C, D, p, B, V_threshold, V itial, time_steps + 1)
X = run_simulation_X(C, r, B, X_initial, time_steps + 1)

results = np.array(X).T[0].T

print(results)

# Plot the results
plt.figure(figsize=(10, 10))
plt.plot(results)
plt.axhline(y=0, color='r', linestyle='--', label='Failure Threshold')
plt.xlabel('Time')
plt.ylabel('Equity Values')
plt.legend([str(i + 1) for i in range(0, len(p))] + ['Failure Threshold'])

plt.grid(linestyle='-', linewidth='0.5', color='grey')

plt.show()
