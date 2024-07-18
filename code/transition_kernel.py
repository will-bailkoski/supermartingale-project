# Should be used for the generation of training pairs

import numpy as np
from itertools import product


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
def state_noise(state):
    noise_dict = {}
    for k in range(len(state)):
        noise_dict["agent_" + str(k)] = (state[k] * 0.9, state[k] * 1.1)

    return list(product(*noise_dict.values()))


def transition_kernel(previous_state, C, B, r):
    # returns old state and uniform list of possible states
    Cx = np.dot(C, previous_state)
    Bphi = np.dot(B, phi(previous_state))

    return state_noise(Cx + r - Bphi)  # add noise
