"""This defines the transition kernel P for the dynamic system and creates training pairs
(NOTE: this does not check for equilibrium set constraints)"""

import numpy as np
from itertools import product


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
