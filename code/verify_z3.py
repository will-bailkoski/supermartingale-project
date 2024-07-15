from z3 import *
import numpy as np
from model_params import n

solver = Solver()

### MODEL CREATION

# state
x = [Real(f"X_{i}") for i in range(n)]

# set A
center = [-4.2] * n
radius = 0.3

squared_distance = sum((p - c) ** 2 for p, c in zip(x, center))
# The point is contained if squared_distance <= radius^2
solver.add(squared_distance > radius ** 2)

# model parameters
from model_params import C, B, V_threshold, D, p

# Convert numpy arrays to Z3 expressions
def np_to_float_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


C = np_to_float_list(C)
B = np_to_float_list(B)
V_threshold = np_to_float_list(V_threshold)
D = np_to_float_list(D)
p = np_to_float_list(p)
r = [Sum([C[i][j] * V_threshold[j][0] for j in range(n)]) - V_threshold[i][0] + Sum([D[i][j] * p[j][0] for j in range(len(p))]) for i in range(n)]

# transition kernel
def P(x):
    Cx = [Sum([C[i][j] * x[j] for j in range(n)]) for i in range(n)]
    phi_x = [If(x[i] < 0, 1, 0) for i in range(n)]
    Bphi = [Sum([B[i][j] * phi_x[j] for j in range(n)]) for i in range(n)]
    return [Cx[i] + r[i] - Bphi[i] for i in range(n)]

### NEURAL NETWORK

h = 10

# weights
W1 = [[Real(f"W1_{i}_{j}") for j in range(h)] for i in range(n)]
W2 = [[Real(f"W2_{i}_{j}") for j in range(1)] for i in range(h)]

# relu function
def relu(x):
    return If(x > 0, x, 0)

# neural network
def V(x):
    layer1 = [Sum([W1[i][j] * x[i] for i in range(n)]) for j in range(h)]
    relu_layer = [relu(val) for val in layer1]
    layer2 = Sum([W2[i][0] * relu_layer[i] for i in range(h)])
    return relu(layer2)

### SUPERMARTINGALE PROPERTIES

V_x = V(x)

epsilon = Real("epsilon")
solver.add(epsilon > 0)

x_tplus1 = P(x)
x_tplus1_up = [i * RealVal('1.1') for i in x_tplus1]
x_tplus1_down = [i * RealVal('0.9') for i in x_tplus1]
E_V_X_tplus1 = 0.5 * V(x_tplus1_up) + 0.5 * V(x_tplus1_down)

solver.add(E_V_X_tplus1 > V_x - epsilon)

# Check satisfiability
if solver.check() == sat:
    m = solver.model()
    print("Satisfiable")
else:
    print("Unsatisfiable")
