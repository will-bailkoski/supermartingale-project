# # List of hyperparameters
# import numpy as np
from equilibrium_set import EquilibriumSet

n = 2      # number of players
m = 2     # number of assets
time_steps = 10

center = -4
# A = EquilibriumSet([[center]] * n, 0.75)
A = EquilibriumSet([[center]] * n, 2)

max_bound = 30.0
min_bound = -10.0

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

