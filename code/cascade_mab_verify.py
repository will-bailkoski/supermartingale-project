import pickle
import numpy as np
import torch
from cascade_functions import v_x, transition_kernel, e_r_x
from MAB_algorithm import mab_algorithm
from functools import partial
from autograd import grad
from cascade_ground_truth import verify_model_milp2
from cascade_reward_bound import find_reward_bound
from BlueBEAR_files.verifier_reward_bound import find_reward_bound as find_reward_bound2

import matplotlib.pyplot as plt

torch.set_printoptions(precision=20)
np.set_printoptions(threshold=20)

n = 4

with open(f"cascade/{n}_players/params.pkl", 'rb') as f:
    params = pickle.load(f)

model_weights = params['network weights']
C = params['C']
B = params['B']
r = params['r']
min_bound, max_bound = params['bounds']
domain = [(min_bound, max_bound)] * n
epsilon = params['epsilon']
verified = params['verified']

W1 = model_weights['fc1.weight']
W2 = model_weights['fc2.weight']
B1 = np.array([model_weights['fc1.bias']]).T
B2 = np.array([model_weights['fc2.bias']])
h = np.shape(W1)[0]

# import time
#
# start = time.time()
# print(verify_model_milp2(n, h, C, B, r, epsilon, model_weights['fc1.weight'], model_weights['fc2.weight'],
#                          model_weights['fc1.bias'], model_weights['fc2.bias'], (min_bound, max_bound)))
# print(start - time.time())


def lipschitz_constant_multivariate(f, domain, num_points=5000):
    dim = len(domain)

    # Generate random points within the domain
    points = np.array([np.random.uniform(domain[i][0], domain[i][1], num_points) for i in range(dim)]).T

    # Gradient function of f
    grad_f = grad(f)

    gradient_norms = []
    old = 0
    for point in points:
        current = np.linalg.norm(grad_f(np.array([point]).T))
        gradient_norms.append(current)
        if current > old:
            old = current
            print(current, point)

    plt.scatter(points[:, 0], points[:, 1], c=gradient_norms, cmap='viridis')
    plt.colorbar(label='Lipschitz Constant')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Color-mapped Scatter Plot of Lipschitz Constant')
    plt.show()

    # Compute gradient norm for each point
    gradient_norms = np.array(gradient_norms)

    # Return the maximum gradient norm (Lipschitz constant estimate)
    return np.max(gradient_norms)


V = partial(v_x, W1=W1, W2=W2, B1=B1, B2=B2)
P = partial(transition_kernel, C=C, B=B, r=r)
R = partial(e_r_x, C=C, B=B, r=r, W1=W1, W2=W2, B1=B1, B2=B2, epsilon=epsilon)
print(P(np.array([[30.0, -10.0, 30.0, 30.0]]).T))
exit(0)

# domain = [(0, 23), (0, 1)]

L = 22.52480933948132
# L = lipschitz_constant_multivariate(R, domain)
# print(f"Estimated Lipschitz constant: {L}")



Ub, upper_point = find_reward_bound(n, h, C, B, r,
                                    model_weights['fc1.weight'], model_weights['fc2.weight'],
                                    model_weights['fc1.bias'], model_weights['fc2.bias'], domain,
                                    upper=True)


print(find_reward_bound2(domain, n, [4, 16, 1], C, B, r, [model_weights['fc1.weight'], model_weights['fc2.weight']], [model_weights['fc1.bias'], model_weights['fc2.bias']], True))
print("now_here")
print(find_reward_bound2(domain, n, [4, 16, 1], C, B, r, [model_weights['fc1.weight'], model_weights['fc2.weight']], [model_weights['fc1.bias'], model_weights['fc2.bias']], False))


#(12.551405305543398, [-10.0, 12.02687502038417, 3.561307329436798, 23.071559298450737])

#(-206.32555632504835, [30.0, -10.0, 30.0, 30.0])



Lb, lower_point = find_reward_bound(n, h, C, B, r,
                                    model_weights['fc1.weight'], model_weights['fc2.weight'],
                                    model_weights['fc1.bias'], model_weights['fc2.bias'], domain,
                                    upper=False)

beta = abs(Ub - Lb)
print(Ub, upper_point)
print(Lb, lower_point)
print(beta)
# beta = 372.244154085987  # 55.88438556500181  # ub, lb: -0.5523203880765994 -559.3961760380944


# lipschitz=calculate_lipschitz_constant(n, 10, C, B, r,
#                                            model_weights['fc1.weight'], model_weights['fc2.weight'],
#                                            model_weights['fc1.bias'], model_weights['fc2.bias'],
#                                            (min_bound, max_bound), 0),
#
# print(lipschitz)

result = mab_algorithm(
    initial_bounds=domain,
    dynamics=P,
    certificate=V,
    lipschitz=L,
    reward_range=beta,
    max_iterations=100000,
    tolerance=0.5
)
print(result)
