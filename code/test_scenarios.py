import numpy as np
from cascade_functions import v_x, transition_kernel, e_v_p_x
from run_model import generate_model_params
from MAB_algorithm import mab_algorithm
#from old_mab import mab_algorithm

from cascade_params import n
from functools import partial
from find_lipschitz import calculate_lipschitz_constant

scenarios = {
    "Stable System": {
        "C": np.array([[0, 0.02], [0.01, 0]]),
        "D": np.array([[0.03, 0.03], [0.03, 0.03]]),
        "p": np.array([20, 20]),
        "B": np.diag([10, 10]),
        "V_threshold": np.array([1.5, 1.5]),
        "V_initial": np.array([5, 5])
    },
    "Single Failure Propagation": {
        "C": np.array([[0, 0.025], [0.005, 0]]),
        "D": np.array([[0.05, 0.05], [0.05, 0.05]]),
        "p": np.array([20, 20]),
        "B": np.diag([12, 12]),
        "V_threshold": np.array([1.5, 1.5]),
        "V_initial": np.array([1.6, 5])
    },
    "Multiple Failures": {
        "C": np.array([[0, 0.03], [0.03, 0]]),
        "D": np.array([[0.04, 0.04], [0.04, 0.04]]),
        "p": np.array([20, 20]),
        "B": np.diag([15, 15]),
        "V_threshold": np.array([2, 2]),
        "V_initial": np.array([2.1, 2.1])
    },
    "Recovery Scenario": {
        "C": np.array([[0, 0.03], [0.01, 0]]),
        "D": np.array([[0.04, 0.04], [0.04, 0.04]]),
        "p": np.array([20, 20]),
        "B": np.diag([12, 12]),
        "V_threshold": np.array([1.5, 1.5]),
        "V_initial": np.array([1.6, 1.6])
    },
    "Example 4: Countries": {
        "C": np.array([
            [0, 0.03, 0.01, 0.07, 0.01, 0.04, 0.04, 0.05, 0.04],  # FR
            [0.04, 0, 0.06, 0.03, 0.00, 0.05, 0.04, 0.09, 0.04],  # DE
            [0.00, 0.00, 0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # GR
            [0.01, 0.03, 0.00, 0, 0.00, 0.01, 0.02, 0.01, 0.00],  # IT
            [0.04, 0.02, 0.00, 0.02, 0, 0.01, 0.01, 0.06, 0.10],  # JP
            [0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00, 0.00],  # PT
            [0.01, 0.02, 0.01, 0.02, 0.00, 0.15, 0, 0.09, 0.02],  # ES
            [0.03, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0, 0.04],  # GB
            [0.04, 0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.09, 0]  # US
        ]),
        "D": np.eye(9),
        "p": np.array([[12.29, 16.81, 1.02, 9.30, 20.00, 1.00, 6.00, 12.99, 75.70]]).T,
        "B": np.diag(np.array([0.5] * 9)),
        "V_threshold": np.array([[10] * 9]).T,
        "V_initial": np.array([[15.2838, 19.9137, 0.9863, 9.0642, 28.3350, 0.7829, 8.8020, 12.1361, 59.8130]]).T
    },
    "Random Gen": {
        "C": np.array([np.random.uniform(0, 0.01, 2), np.random.uniform(0, 0.01, 2)]),  # fill diagonal
        "D": np.array([[0.05] * 2] * 2).T,
        "p": np.array([[10] * 2]).T,
        "B": np.diag(np.array([0.4] * 2)),
        "V_threshold": np.array([[5] * 2]).T,
        "V_initial": np.array([np.random.uniform(0, 30, 2)]).T
    }

}



# milp is wrong, z3 is right (available counterexample)
# W1 = np.array([[-0.27046862244606018066,  0.43653589487075805664],
#         [ 0.10967749357223510742, -0.50722730159759521484],
#         [ 0.61245816946029663086, -0.03348112106323242188],
#         [ 0.06456953287124633789, -0.70694494247436523438],
#         [ 0.03421127796173095703, -0.06865435838699340820],
#         [-0.56489008665084838867,  0.38552206754684448242],
#         [-0.68398714065551757812, -0.57208490371704101562]])
#
# W2 = np.array([[ 0.01577985286712646484, -0.14954538643360137939,
#          -0.30621963739395141602,  0.31595715880393981934,
#          -0.15766184031963348389, -0.12686502933502197266,
#          -0.00484248995780944824]])
#
# B1 = np.array([ 0.11334848403930664062, -0.09055352210998535156,
#         -0.30319076776504516602, -0.41028404235839843750,
#         -0.62621456384658813477,  0.58490008115768432617,
#         -0.02504062652587890625])
#
# B2 = np.array([-0.36402562260627746582])


# milp is wrong, z3 is right (no counter example)
# W1 = np.array([[-0.51114511489868164062,  0.57946199178695678711],
#         [ 0.32534515857696533203,  0.77040141820907592773],
#         [ 0.29491353034973144531, -0.23997142910957336426],
#         [ 0.13821397721767425537,  0.23935167491436004639],
#         [-0.18271064758300781250, -0.37764522433280944824],
#         [ 0.03944269940257072449,  0.05401853471994400024],
#         [-0.22435133159160614014, -0.02875019051134586334]])
#
# W2 = np.array([[ 0.65042066574096679688,  0.47615325450897216797,
#           0.11454802751541137695,  0.02800757251679897308,
#          -0.04927513748407363892, -0.13231205940246582031,
#           0.32181644439697265625]])
#
# B1 = np.array([-0.26514554023742675781,  0.23466676473617553711,
#          0.69748425483703613281, -0.19338539242744445801,
#         -0.22832578420639038086, -0.21010714769363403320,
#         -1.01455318927764892578])
#
# B2 = np.array([0.11887560784816741943])

#both correct (no counter example)
W1 = np.array([[-0.05612364783883094788,  0.22464643418788909912],
        [ 0.85973018407821655273, -0.60979819297790527344],
        [-0.29311794042587280273, -0.60186690092086791992],
        [-0.08638557791709899902,  0.38987949490547180176],
        [ 0.07084782421588897705, -0.07997295260429382324],
        [ 0.49275588989257812500,  0.50038772821426391602],
        [ 0.22580890357494354248,  0.31280735135078430176]])

W2 = np.array([[ 0.07325454056262969971,  0.33157727122306823730,
          0.09291243553161621094,  0.67297852039337158203,
         -0.01096701808273792267,  0.44333624839782714844,
          0.32278397679328918457]])

B1 = np.array([-0.75074553489685058594,  0.50885915756225585938,
         0.25227051973342895508,  1.08249890804290771484,
        -0.08847936987876892090,  1.22585427761077880859,
         0.78795963525772094727])

B2 = np.array([-0.22296056151390075684])


from cascade_reward_bound import find_reward_upper_bound
from find_lower import find_reward_lower_bound
C, _, _, B, _, _, r = generate_model_params(2, 2)
#
# z = verify_model(2, 7, A, C, B, r, 0.001, W1, W2, B1, B2)
# g = verify_model_gurobi(2, 7, A, C, B, r, 0.001, W1, W2, B1, B2)
# print("\n\n\n", z, g, z == g)

P = partial(transition_kernel, C=C, B=B, r=r)
V = partial(v_x, W1=W1, W2=W2, B1=np.array([B1]).T, B2=np.array([B2]))
EVP = partial(e_v_p_x, C=C, B=B, r=r, W1=W1, W2=W2, B1=np.array([B1]).T, B2=np.array([B2]))

up = find_reward_upper_bound(2, 7, C, B, r, W1, W2, B1, B2)
low = find_reward_lower_bound(2, 7, C, B, r, W1, W2, B1, B2)
bounds = (-10, 30)
L = calculate_lipschitz_constant(2, 7, C, B, r, W1, W2, B1, B2, bounds, 0)


# test = np.array([[1.6667707707454265],[0.5935334256882377]])
# print(EVP(test)-V(test))

print("\n\n\n\n")
print("L: ", L)
print("beta: ", abs(up-low))
print(up)

result = mab_algorithm(
    initial_bounds=[bounds] * n,
    dynamics=P,
    certificate=V,
    lipschitz=L,
    beta=abs(up-low),
    max_iterations=500000
)
#
# def P(x):
#     return -2 * x + 5
#
# def P(x):
#     return -np.sqrt(x[0][0]**2 + x[1][0]**2)
#
# def V(x):
#     return x
#
#
#
# print("\n\n\n\n")
#
# result = mab_algorithm(
#     initial_bounds=[(1, 3)],
#     dynamics=P,
#     certificate=V,
#     lipschitz=2,
#     beta=2,
#     max_iterations=2000
# )






print(f"Certificate is {'valid' if result else 'invalid'}")
