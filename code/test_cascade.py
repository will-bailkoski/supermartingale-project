import numpy as np
import torch
from function_application import v_x, transition_kernel, e_r_x
from MAB_algorithm import mab_algorithm
from functools import partial
from model_params import B, min_bound, max_bound, n
from autograd import grad
from verify_cascade_supermartingale import verify_model_milp
from cascade_reward_bound import find_reward_bound

import matplotlib.pyplot as plt

torch.set_printoptions(precision=20)
np.set_printoptions(threshold=20)

model_weights = torch.load("cascade/model_weights_epsilon1.pth")
C = np.load('cascade/covariance_working.npy')

r = np.load("cascade/saved_r_working.npy")

epsilon = 1   # epsilon that model was trained on, not scaled = 1
scale_factor = 1

W1 = model_weights['fc1.weight']
W2 = model_weights['fc2.weight'] * scale_factor
B1 = np.array([model_weights['fc1.bias']]).T
B2 = np.array([model_weights['fc2.bias']]) * scale_factor

print(verify_model_milp(2, 16, C, B, r, epsilon, model_weights['fc1.weight'], model_weights['fc2.weight'] * scale_factor, model_weights['fc1.bias'], model_weights['fc2.bias'] * scale_factor))


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

domain = [(min_bound, max_bound)] * n  # Domain for x1 and x2
#
#domain = [(-0.4, 0.01), (-0.11, 0.25)]
# # # #
#L = lipschitz_constant_multivariate(R, domain)
# print(f"Estimated Lipschitz constant: {L}")

L = 14.792065347942774  # [-0.0058817 -0.0538078]

# Ub, upper_point = find_reward_bound(2, 16, C, B, r,
#                                       model_weights['fc1.weight'], model_weights['fc2.weight'] * scale_factor,
#                                       model_weights['fc1.bias'], model_weights['fc2.bias'] * scale_factor, epsilon, True)
#
# Lb, lower_point = find_reward_bound(2, 16, C, B, r,
#                                        model_weights['fc1.weight'], model_weights['fc2.weight'] * scale_factor,
#                                        model_weights['fc1.bias'], model_weights['fc2.bias'] * scale_factor, epsilon, False)
#
# beta = abs(Ub - Lb)
# print(beta)
beta = 372.244154085987  # 55.88438556500181  # ub, lb: -0.5523203880765994 -559.3961760380944


#lipschitz=calculate_lipschitz_constant(n, 10, C, B, r,
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
    beta=beta,
    max_iterations=5000000,
    epsilon=epsilon
)
print(result)
