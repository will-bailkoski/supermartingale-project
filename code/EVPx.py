"""This file is for testing the results for a given state and a set of weights, either the immediate result for state
x or the expected results for x(t+1)"""

import numpy as np
from transition_kernel import transition_kernel
from run_model import generate_model_params
from model_params import n, m

C, _, _, B, _, _, r = generate_model_params(n, m)

def ReLu(n):
    if n > 0:
        return n
    else:
        return 0


def e_v_p_xt1(x, W1, W2, B1, B2):
    p_x = transition_kernel(x, C, B, r)

    v_p_x = []

    for xi in p_x:
        l1 = np.dot(xi.T, W1.T) + B1
        l1[0] = [ReLu(i) for i in l1[0]]
        l2 = np.dot(l1, W2.T) + B2
        v_p_x.append(ReLu(l2[0][0]))

    return np.mean(v_p_x)

def v_p_xt(x, W1, W2, B1, B2):

    l1 = np.dot(np.array(x).T, W1.T) + B1
    l1[0] = [ReLu(i) for i in l1[0]]
    l2 = np.dot(l1, W2.T) + B2

    return ReLu(l2[0][0])
