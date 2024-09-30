"""This file provides a method of applying various functions, including executing a given realisation of the neural
network and applying the transition kernel to a state. The transition kernel can be used to generate training pairs,
and the others can be used to test neural network outputs. Every function takes the state x as a column vector"""

import numpy as np
from itertools import product
from model_params import n, m

def e_r_x(x, C, B, r, W1, W2, B1, B2, epsilon):
    return e_v_p_x(x, C, B, r, W1, W2, B1, B2) - v_x(x, W1, W2, B1, B2) - epsilon


def e_v_p_x(x, C, B, r, W1, W2, B1, B2):



    # assert x.shape == (2, 1), f"Input x must have shape (2, 1), but has shape {x.shape}"
    # assert W1.shape == (7, 2), f"Input W1 must have shape (7, 2), but has shape {W1.shape}"
    # assert B1.shape == (7, 1), f"Input B1 must have shape (7, 1), but has shape {B1.shape}"
    # assert W2.shape == (1, 7), f"Input W2 must have shape (1, 7), but has shape {W2.shape}"
    # assert B2.shape == (1, 1), f"Input B2 must have shape (1, 1), but has shape {B2.shape}"

    p_x = transition_kernel(x, C, B, r)
    v_p_xs = [v_x(i, W1, W2, B1, B2) for i in p_x]
    return sum(v_p_xs) / 4


def v_x(x, W1, W2, B1, B2):

    # assert x.shape == (2, 1), f"Input x must have shape (2, 1), but has shape {x.shape}"
    assert W1.shape == (16, 2), f"Input x must have shape (7, 2), but has shape {W1.shape}"
    # assert B1.shape == (7, 1), f"Input x must have shape (7, 1), but has shape {B1.shape}"
    # assert W2.shape == (1, 7), f"Input x must have shape (1, 7), but has shape {W2.shape}"
    # assert B2.shape == (1, 1), f"Input x must have shape (1, 1), but has shape {B2.shape}"


    z1 = np.dot(W1, x) + B1
    a1 = np.maximum(0, z1)
    z2 = np.dot(W2, a1) + B2
    return np.maximum(0, z2)[0][0]


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
    # returns old state and list of possible states
    assert previous_state.shape == (n, 1), \
        f"Input x must have shape ({n}, 1), but has shape {previous_state.shape}"
    assert C.shape == (n, n), \
        f"Input x must have shape ({n}, {n}), but has shape {previous_state.shape}"
    assert B.shape == (n, n), \
        f"Input x must have shape (2, 1), but has shape {previous_state.shape}"
    assert r.shape == (n, 1), \
        f"Input x must have shape ({n}, 1), but has shape {r.shape}"
    Cx = np.dot(C, previous_state)
    Bphi = np.dot(B, phi(previous_state))

    return np.array(state_noise(Cx + r - Bphi))  # add noise


