import torch
import numpy as np
from functools import partial
from numpy.linalg import svd
from property_tests import lipschitz_constant_statistical_test
import random
import time

from MAB_algorithm import mab_algorithm

def v_x(x, weights, biases):
    output = x.T[0]
    for weight, bias in zip(weights[:-1], biases[:-1]):
        output = np.maximum(np.dot(output, weight.T) + bias, 0)  # Linear layer + ReLU activation

    # For the last layer, no activation (or apply one if specified)
    output = output @ weights[-1].T + biases[-1]
    return np.maximum(output, 0)[0]


def verify_cycle(model_weights, network_depth, domain_bounds, P, alpha_p, epsilon, confidence, kappa):
    print("\n")
    with torch.no_grad():

        # overestimation of the lipschitz constant
        alpha_f = 1
        for key in model_weights.keys():
            param = model_weights[key]
            if len(param.shape) == 2:  # Check if it is a weight matrix (not a bias vector)
                _, singular_values, _ = svd(param, full_matrices=False)
                spectral_norm = np.max(singular_values)
                alpha_f *= spectral_norm

        print(alpha_f)

        # print(model_weights.keys())
        weights = [model_weights[f'network.{i * 2}.weight'] for i in range(network_depth + 1)]
        biases = [model_weights[f'network.{i * 2}.bias'] for i in range(network_depth + 1)]
        print(weights)
        V = partial(v_x, weights=weights, biases=biases)

        alpha = alpha_f * alpha_p
        print(alpha)
        def R(x):
            return np.average([V(P(x)[i]) for i in range(3)]) - V(x) + epsilon
        lipschitz_constant_statistical_test(R, alpha, domain_bounds)

        verify_start_time = time.process_time()
        is_sat, counter_example, its, avg_time, tree_depth, num_regions = mab_algorithm(
            initial_bounds=domain_bounds,
            dynamics=P,
            certificate=V,
            lipschitz_values=(alpha, alpha_f),
            tolerance=epsilon,
            confidence=confidence,
            kappa=kappa
        )

        return is_sat, counter_example, time.process_time() - verify_start_time, its, avg_time, tree_depth, num_regions, alpha

