from z3 import *
import numpy as np
from neural import input_size as n, hidden_size as h

solver = Solver()


### MODEL CREATION

# state
x = [Real(f"X_{i}") for i in range(n)]
solver.add()  # TODO: state constraints, including not being in set A

# model parameters
C = np.array([[0.9, 0.1], [0.1, 0.9]])
B = np.array([[0.1, 0], [0, 0.1]])
r = np.array([0.1, 0.1])  # TODO: implement

# transition kernel
def P(x):
    Cx = [Sum([C[i][j] * x[j] for j in range(n)]) for i in range(n)]
    phi_x = [If(x[i] < 0, 1, 0) for i in range(n)]
    Bphi = [Sum([B[i][j] * phi_x[j] for j in range(n)]) for i in range(n)]
    return [Cx[i] + r[i] - Bphi[i] for i in range(n)]


### NEURAL NETWORK

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
x_tplus1_up = [i * 1.1 for i in x_tplus1]
x_tplus1_down = [i * 0.9 for i in x_tplus1]
E_V_X_tplus1 = 0.5 * V(x_tplus1_up) + 0.5 * V(x_tplus1_down)

solver.add(E_V_X_tplus1 > V_x - epsilon)

# Check satisfiability
if solver.check() == sat:
    m = solver.model()
    print("Satisfiable")
else:
    print("Unsatisfiable")
