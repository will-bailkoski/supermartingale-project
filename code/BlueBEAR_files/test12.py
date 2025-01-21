import numpy as np
import pickle
from verifier import verify_cycle
from param_generator import transition_kernel
from functools import partial

n = 2

with open(f"../cascade/{n}_players/True_73.pkl", 'rb') as f:
    params = pickle.load(f)
    alpha_f = params['lipschitz']
    model_weights = params['model_weights']
    params = np.load(f"../BlueBEAR_files/param_examples/n{n}_m2.npz")
    C = params['C']
    B = params['B']
    r = params['r']

model_weights = {'network.0.weight': model_weights['fc1.weight'],
                 'network.2.weight': model_weights['fc2.weight'],
                 'network.0.bias': model_weights['fc1.bias'],
                 'network.2.bias': model_weights['fc2.bias']}
P = partial(transition_kernel, C=C, B=B, r=r, kappa=0.1)


min_bound, max_bound = 10.0, 30.0
domain = [(min_bound, max_bound)] * n
epsilon = 0.05
alpha_p = 3.0924482627001937
print(alpha_f)


print(verify_cycle(model_weights, 1, domain, P, alpha_p, epsilon, 0.95, 0.1))
