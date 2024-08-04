"""This file is for testing the results for a given state and a set of weights, either the immediate result for state
x or the expected results for x(t+1)"""

import numpy as np
from itertools import product

def e_v_p_x(x, C, B, r, W1, W2, B1, B2):
    p_x = transition_kernel([[x[0]], [x[1]]], C, B, r)

    v_p_x = []

    for xi in p_x:
        xs = [i[0] for i in xi]
        z1 = np.dot(W1, xs) + B1
        a1 = np.maximum(0, z1)
        z2 = np.dot(W2, a1) + B2
        v_p_x.append(np.maximum(0, z2)[0])

    print(v_p_x)

    return np.mean(v_p_x)

def v_x(x, W1, W2, B1, B2):
    z1 = np.dot(W1, x) + B1
    a1 = np.maximum(0, z1)
    z2 = np.dot(W2, a1) + B2
    return np.maximum(0, z2)[0]


# Transition kernel

# Failure function
def phi(v_vbar):
    indicator = []
    for i in range(0, len(v_vbar)):
        if v_vbar[i][0] < 0:
            indicator.append([1])
        else:
            indicator.append([0])
    return indicator


# Stochasticity
def state_noise(state):
    noise_dict = {}
    for k in range(len(state)):
        noise_dict[str(k)] = (state[k] * 0.9, state[k] * 1.1)

    return list(product(*noise_dict.values()))


def transition_kernel(previous_state, C, B, r):
    # returns old state and uniform list of possible states
    Cx = np.dot(C, previous_state)
    Bphi = np.dot(B, phi(previous_state))

    return np.array(state_noise(Cx + r - Bphi))  # add noise
