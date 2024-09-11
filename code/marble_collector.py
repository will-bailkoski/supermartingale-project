from MAB_algorithm import mab_algorithm
from scipy.stats import bernoulli
import numpy as np

params = {
    'n_dims' : 2,
    'bounds' : (-10, 10),
    'epsilon': 0.5,
    'max_its': 10000000,
}



def ReLU(x):
    if x > 0:
        return x
    else:
        return 0


def V_not_supermartingale(xs):
    return sum([x[0] ** 2 for x in xs])  # should be invalid


def V_is_supermartingale(xs):
    return sum([ReLU(x[0]) for x in xs])  # should be invalid


def P(xs):
    p = bernoulli.rvs(0.01)
    one = xs[0][0]
    two = xs[1][0]
    if p == 1:
        one -= 1
    else:
        two -= 1
    new = [[one], [two]]
    return np.array([new, new])


print("Testing valid certificate")
result_2 = mab_algorithm(initial_bounds=[params['bounds']] * params['n_dims'],
                         dynamics=P,
                         certificate=V_is_supermartingale,
                         lipschitz=1,
                         beta=1,
                         max_iterations=params['max_its'],
                         epsilon=params['epsilon']
                         )

print("Testing invalid certificate")
result_1 = mab_algorithm(initial_bounds=[params['bounds']] * params['n_dims'],
                         dynamics=P,
                         certificate=V_not_supermartingale,
                         lipschitz=1,
                         beta=1,
                         max_iterations=params['max_its'],
                         epsilon=params['epsilon']
                         )


assert not result_1, "invalid supermartingale was validated by MAB algorithm"

assert result_2, "valid supermartingale was invalidated by MAB algorithm"

print("\n\n")
print("marble collector problem was successful")

# W1 = np.array([[-0.40369884828535151922, -0.51349175080265641036],
#         [ 0.67971181048501649880, -0.15243524296605237556],
#         [-0.55905744056457729041, -0.65526218343386899434],
#         [ 0.29559114953665854841,  0.43805472188591809690],
#         [-0.18126565307869768606,  0.62632325396041199639],
#         [ 0.24195706778399536652,  0.43944569022091728439],
#         [-0.22654341259628951732,  0.18717453706565784222]])
#
# b1 = np.array([[-0.22969296337774958161, -0.52858386952670555203,
#          0.41835231905633618599,  0.12779194018533107657,
#          0.04565641053426932466,  0.50197623701607585467,
#         -0.46429536846822128116]]).T
#
# W2 = np.array([[ 0.23900907519939920687,  0.07265609797110803014,
#           0.17770430631556272116,  0.16104838848234889759,
#           0.13104681440947504867, -0.21462923352033916324,
#           0.37907912351542294438]])
#
# b2 = np.array([
#     [0.19227620959281921387]
# ])
#
# # Example input
# x = np.array([[18.05375591065114804223], [14.80897300800913285457]])
#
# # tensor([13.67473958974708381220, 24.66485265891335387778], dtype=torch.float64) tensor([ 0.95210715260547029715, 18.36132468426156805208], dtype=torch.float64) tensor([18.05375591065114804223, 14.80897300800913285457], dtype=torch.float64)
# # tensor([1.82910928270255901396], dtype=torch.float64,
# #        grad_fn=<SelectBackward0>) tensor([1.83914045636807044204], dtype=torch.float64,
# #        grad_fn=<SelectBackward0>) tensor([1.83797686966330253711], dtype=torch.float64,
# #        grad_fn=<SelectBackward0>)
#
# import torch
#
# C, _, _, B, _, _, r = generate_model_params(2, 2)
#
# print(e_v_p_x(x, C, B, r,  W1, W2, b1, b2))
