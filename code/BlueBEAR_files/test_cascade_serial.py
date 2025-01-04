import pandas as pd
from functools import partial
import numpy as np
from numpy.linalg import svd

from generate_cascade_params import generate_params, transition_kernel
from cegis import find_supermartingale
from verifier_reward_bound import find_reward_bound


def test_params(n, m, domain_bounds, network_width, network_depth):
    #asserts?

    C, D, p, B, V_threshold, r = generate_params(n, m)
    norm_C = np.linalg.svd(C, compute_uv=False)[0]
    norm_B = np.linalg.svd(B, compute_uv=False)[0]
    lipschitz = max(norm_C, norm_B)  # lipschitz estimate of the system. doesnt include effect of indicator

    P = partial(transition_kernel, C=C, B=B, r=r)
    reward_optimiser = partial(find_reward_bound, bounds=domain_bounds, input_size=n, layer_sizes=[n] + [network_width] * network_depth + [1], C=C, B=B, r=r)

    find_supermartingale(domain_bounds, n, P, lipschitz, network_width, network_depth, confidence, reward_optimiser)



    params = {
        'bounds': domain_bounds,
        'epsilon': epsilon,
        'verified': is_sat,
        'C': C,
        'B': B,
        'r': r,
        'network weights': model_weights
    }

    # Save the dictionary to a file
    with open(f"cascade/{n}_players/params.pkl", 'wb') as f:
        pickle.dump(params, f)




param_sets = [
    {"param1": 1, "param2": 2},
    {"param1": 3, "param2": 4},
    {"param1": 5, "param2": 6}
]

results = []
for params in param_sets:
    outcome = test_params(**params)
    results.append({**params, **outcome})


df = pd.DataFrame(results)
print(df)
df.to_csv("results.csv", index=False)
