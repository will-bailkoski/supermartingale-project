import numpy as np
import matplotlib.pyplot as plt
import itertools

outcomes = ['up', 'down']  # brownian motion
noise_pmf = [0.5, 0.5]

training_pairs = []


# Define the failure function phi
def phi(v_vbar):
    indicator = []
    for i in range(0, len(v_vbar)):
        if v_vbar[i][0] < 0:
            indicator.append([1])
        else:
            indicator.append([0])
    return indicator


# Stochasticity
def state_noise(state, outcome, old_state):

    global outcomes
    global training_pairs

    down = []
    up = []
    for k in range(len(state)):
        down.extend(state[k] - (0.1 * state[k]))
        up.extend(state[k] + (0.1 * state[k]))

    possible_states = list(itertools.product(up, down))
    training_pairs.append((old_state.T, possible_states))

    for k in range(len(state)):
        if outcome[k] == 'down':
            state[k] = state[k] - (0.05 * state[k])  # action 1
        else:
            state[k] = state[k] + (0.05 * state[k])  # action 2

    return state

def transition_kernel(C, previous_state, B, r, sample):
    Cx = np.dot(C, previous_state)
    Bphi = np.dot(B, phi(previous_state))

    return state_noise(Cx + r - Bphi, sample, previous_state)  # add noise

def run_simulation_X(C, r, B, X_initial, time_steps, samples):

    # Initialize the array to store equity values over time
    X = [None] * (time_steps)
    X[0] = X_initial

    for t in range(1, time_steps):
        X[t] = transition_kernel(C, X[t-1], B, r, samples[t-1])

    return X


from model_params import n, C, D, p, B, V_threshold, X_initial
r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)

# Equilibrium check???
# a = np.linalg.inv(np.eye(len(C)) - C)
# b = r - beta
# if np.dot(a, b)[0][0] >= 0 and np.dot(a, b)[1][0] >= 0:  # equilibrium check



time_steps = 10
samples = []
for i in range(time_steps):
    samples.append(list(np.random.choice(outcomes, size=n, p=noise_pmf)))


X = run_simulation_X(C, r, B, X_initial, time_steps + 1, samples)

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



