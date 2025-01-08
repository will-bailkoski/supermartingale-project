import pandas as pd
import time
from functools import partial
import numpy as np
from numpy.linalg import svd
import pickle
import os

from param_generator import find_and_save_example, transition_kernel
from cegis import find_supermartingale
from verifier_reward_bound import find_reward_bound
from test_cases import param_sets
from property_tests import lipschitz_constant_statistical_test





def test_params(n, m, domain_bounds, network_width, network_depth, confidence):
    #asserts?

    filename = f"param_examples/n{n}_m{m}.npz"
    # Check if the file exists
    if os.path.exists(filename):
        print(f"Loading existing example")
        data = np.load(filename)
        C = data['C']
        B = data['B']
        r = data['r']
    else:
        print(f"Generating new example for n={n}, m={m}")
        C, D, p, B, v_threshold, r = find_and_save_example(n, m)

    P = partial(transition_kernel, C=C, B=B, r=r)
    print(f"Successfully found valid parameters")

    # Lipschitz constant calculations TODO: consider stochastic element of P
    max_eigenvalue = np.max(np.linalg.eigvals(np.dot(C, C)).real)
    beta_norm_squared = np.dot(np.diag(B), np.diag(B))
    lipschitz = np.sqrt(max_eigenvalue + beta_norm_squared)
    print(f"Lipschitz constant of P: {lipschitz}")
    # lipschitz_constant_statistical_test(P, lipschitz, 1000000, domain_bounds)

    reward_optimiser = partial(find_reward_bound, bounds=domain_bounds, input_size=n, layer_sizes=[n] + [network_width] * network_depth + [1], C=C, B=B, r=r)

    print("Looking for certificate")
    print(f"with parameters: n={n}, m={m}, domain_bounds={domain_bounds}, network_width={network_width}, network_depth={network_depth}, confidence={confidence}")
    start_time = time.process_time()
    success, network_run_times, network_it_nums, verifier_run_times, verifier_it_nums, verifier_avg_it_times, verifier_tree_depth, verifier_regions_nums, alpha_history, beta_history, loss_history, model_weights = find_supermartingale(domain_bounds, n, P, lipschitz, network_width, network_depth, confidence, reward_optimiser)
    end_time = time.process_time()
    print("validated supermartingale")

    size = 1  # size of state space
    for min_val, max_val in domain_bounds:
        size *= max_val - min_val
    with open(f"model_weights/{n}_{m}_{size}_{network_width}_{network_depth}_{confidence}.pkl", 'wb') as f:
        pickle.dump(model_weights, f)

    print(np.average(verifier_run_times), np.average(verifier_it_nums), verifier_avg_it_times, np.average(verifier_tree_depth), np.average(verifier_regions_nums))

    return {"successfully_found_and_verified?": success,
    "total_time_to_find_valid_supermartingale": (start_time - end_time),
    "network_run_times": network_run_times,
    "network_iteration_nums": network_it_nums,
    "verifier_run_times": verifier_run_times,
    "verifier_iteration_nums": verifier_it_nums,
    "verifier_average_iteration_lengths": verifier_avg_it_times,
    "verifier_tree_depth": verifier_tree_depth,
    "verifier_number_of_regions": verifier_regions_nums,
    "lipschitz_history": alpha_history,
    "reward_range_history": beta_history,
    "network_loss_history": loss_history}

    # params = {
    #     'bounds': domain_bounds,
    #     'epsilon': epsilon,
    #     'verified': is_sat,
    #     'C': C,
    #     'B': B,
    #     'r': r,
    #     'network weights': model_weights
    # }
    #
    #Save the dictionary to a file
    #


results = []
for params in param_sets:
    outcome = test_params(**params)
    results.append({**params, **outcome})
    df = pd.DataFrame(results)
    print(df)

df.to_csv("results.csv", index=False)
