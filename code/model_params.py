# # List of hyperparameters
import numpy as np
from equilibrium_set import EquilibriumSet

n = 2  # number of players (companies/countries/agents/...)
m = 2  # number of assets
time_steps = 10  # length of trajectory modelling

# domain bounds
max_bound = 30.0
min_bound = -10.0

C = np.array([[0., 0.0052315],
              [0.00117426, 0.]])

D = np.array([[0.06] * n] * m)
p = np.array([[10] * m]).T  # TODO: shape of p in original cascade paper is unclear

beta = np.array([[0.4] * n]).T  # definable
B = np.diag(beta.T[0])  # Failure costs

V_threshold = np.array([[5] * n]).T  # Failure thresholds

r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)

# Equilibrium check
Cinv = np.linalg.inv(np.eye(len(C)) - C)
check1 = np.dot(Cinv, r - beta)
check2 = np.dot(Cinv, r)
assert(check1.shape == (n, 1))
assert(check2.shape == (n, 1))
positivity_check = [True if x[0] >= 0 else False for x in check2]
negativity_check = [True if x[0] < 0 else False for x in check1]

if all(positivity_check):
    print("positive equilibrium exists")
    if all([True if x[0] >= 0 else False for x in check1]):
        print("equilibrium is unique")
if all(negativity_check):
    print("negative equilibrium exists")
    if all([True if x[0] < 0 else False for x in check2]):
        print("equilibrium is unique")

# invariant set
# TODO: determine how to properly construct the invariant set
NN = 10  # Number of iterations
Psip = 2 * np.random.rand(n, 1) - np.ones((n, 1)) + 0.2 * np.ones((n, 1))  # Replace with your Psip


# Compute x_0 (initial random vector)
x_0 = 2 * np.random.rand(n, 1) - np.ones((n, 1)) - np.random.rand(n, 1)

# Start with A as the identity matrix
A = np.eye(n)
b = np.zeros((n, 1))

# Iterate to construct A and b
for k in range(1, NN + 1):
    A = np.vstack((np.eye(n), np.dot(A, C)))
    b_add = np.dot((np.eye(n) - np.linalg.matrix_power(C, k)), np.linalg.inv(np.eye(n) - C)).dot(Psip)
    b = np.vstack((b, b_add))

# Adjust b with initial condition x_0
b = b + np.dot(A, x_0)

# Print or return A and b
print("Matrix A:")
print(A)
print("Vector b:")
print(b)

center = -4
A = EquilibriumSet([[center]] * n, 2)


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
