import pandas as pd
from functools import partial

from generate_cascade_params import generate_params, transition_kernel
from cegis import find_supermartingale
from verifier_reward_bound import find_reward_bound

def subroutine(n, m, domain_bounds, network_width, network_depth, ):
    #asserts?

    C, D, p, B, V_threshold, r = generate_params(n, m)
    P = partial(transition_kernel, C=C, B=B, r=r)
    reward_optimiser = partial(find_reward_bound, bounds=domain_bounds, n=n, h=network_depth, C=C, B=B, r=r)

    find_supermartingale()




param_sets = [
    {"param1": 1, "param2": 2},
    {"param1": 3, "param2": 4},
    {"param1": 5, "param2": 6}
]

results = []
for params in param_sets:
    outcome = subroutine(**params)
    results.append({**params, **outcome})


df = pd.DataFrame(results)
print(df)
df.to_csv("results.csv", index=False)
