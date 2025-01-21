# # List of hyperparameters
import numpy as np
from invariant_set import InvariantBall, check_invariant_set, find_point_in_invariant_set, construct_invariant_set
from copy import deepcopy

n = 4  # number of players (companies/countries/agents/...)
m = 2  # number of assets

# domain bounds
max_bound = 30.0
min_bound = 10.0

assert min_bound < max_bound, "bounds are incorrectly ordered"


def negative_quad_invariant_check(C, Cinv, r, beta, V_threshold, D, p):  # checks to see if the entire negative quadrant is invariant

    check_1 = all(np.dot(Cinv, r - beta) < 0)  # Lemma 3.3, ensure an equilibrium point is in the negative quadrant
    check_2 = not all(np.dot(Cinv, r) < 0)  # Lemma 3.4, ensure the entire plane is not invariant
    check_3 = all(np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p) < beta)  # Theorem 3, ensure the negative quadrant is invariant

    return check_1 and check_2 and check_3


def is_in_invariant_set(x):
    return np.all(x < 0)


def generate_random_C(n):
    while True:
        # Initialize matrix with zeros
        C = np.zeros((n, n))

        # Populate off-diagonal elements with random non-negative values
        for i in range(n):
            for j in range(n):
                if i != j:
                    C[i, j] = np.random.random()

        # Scale columns if necessary to ensure column sum < 1
        column_sums = C.sum(axis=0)
        scaling_factors = np.minimum(1, 1 / column_sums)
        C = C * scaling_factors[np.newaxis, :]


        # Check for nonsingularity (determinant must be non-zero)
        cond_threshold = 1e9
        if np.linalg.det(C) != 0:
            I_minus_C = np.eye(n) - C
            if np.linalg.det(I_minus_C) != 0 and np.linalg.cond(I_minus_C) < cond_threshold:
                return C, np.linalg.inv(I_minus_C)


flag = True
i = 1
while not flag:

    print(f"attempt {i}")
    i += 1
    C, Cinv = generate_random_C(n)


    D = np.random.uniform(0, 0.1, (n, m))
    p = np.array([[10] * m]).T  # paper has the shape incorrect sometimes

    beta = np.array([[0.3] * n]).T  # definable
    B = np.diag(beta.T[0])  # Failure costs

    V_threshold = np.array([[5] * n]).T  # Failure thresholds

    r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)

    flag = negative_quad_invariant_check(C, Cinv, r, beta, V_threshold, D, p)

#print(C)








# Equilibrium check
# Cinv = np.linalg.inv(np.eye(len(C)) - C)
# check1 = np.dot(Cinv, r - beta)
# check2 = np.dot(Cinv, r)
# assert(check1.shape == (n, 1))
# assert(check2.shape == (n, 1))
# positivity_check = all([True if x[0] >= 0 else False for x in check2])
# negativity_check = all([True if x[0] < 0 else False for x in check1])
#
# if positivity_check:
#     print("positive equilibrium exists")
#     if all([True if x[0] >= 0 else False for x in check1]):
#         print("equilibrium is unique")
#     Phi = deepcopy(r)
#
# elif negativity_check:
#     print("negative equilibrium exists")
#     if all([True if x[0] < 0 else False for x in check2]):
#         print("equilibrium is unique")
#     Phi = r - beta
#
# else:
#     assert False, "no equilibrium found"
#
#
# A, b = construct_invariant_set(2, n, C, Phi)
#
# invariant_check = check_invariant_set(n, C, B, r, A, b, (min_bound, max_bound))
# print(invariant_check)
#
# print(find_point_in_invariant_set(n, A, b))
#
# from visualise import show_invariant_region
# show_invariant_region(A, b, (-10, 0), 100000)




# invariant set
# TODO: determine how to properly construct the invariant set
NN = 5  # Number of iterations





# center = -4
# A = InvariantBall([[center]] * n, 1)
#
# invalid_set, _ = A.check_invariant_ball(n, C, B, r, (min_bound, max_bound))
#
# assert invalid_set, "set is not invariant"



#
# C = []
# for i in range(n):
#     C.append(np.random.uniform(0, 0.01, n))  # cross-holdings matrix                # definable
# C = np.array(C)
# np.fill_diagonal(C, 0)
#
# D = np.array([[0.06] * n] * m).T  # Market price of assets                          # definable
# p = np.array([[10] * m]).T  # Initial market prices                                 # definable
#
# beta = np.array([0.4] * n)                                                          # definable
# B = np.diag(beta)  # Failure costs
# beta = np.array([beta]).T
#
# V_threshold = np.array([[5] * n]).T  # Failure thresholds                           # definable
# X_initial = np.array([np.random.uniform(0, 30, n)]).T                               # definable
# r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)
#
