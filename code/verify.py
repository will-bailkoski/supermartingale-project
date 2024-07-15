import gurobipy as gp
from gurobipy import GRB
import numpy as np
from model_params import n, C, B, V_threshold, D, p

model = gp.Model("MILP_Encoding_Verification")

# State
x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="X")

# Set A
center = [-4.2] * n
radius = 0.3
squared_distance = gp.quicksum((x[i] - center[i])**2 for i in range(n))
model.addConstr(squared_distance >= radius**2, "not_in_set_A") # TODO: ask about constraints

# model
C = C.tolist() if isinstance(C, np.ndarray) else C
B = B.tolist() if isinstance(B, np.ndarray) else B
V_threshold = V_threshold.tolist() if isinstance(V_threshold, np.ndarray) else V_threshold
D = D.tolist() if isinstance(D, np.ndarray) else D
p = p.tolist() if isinstance(p, np.ndarray) else p

r = [sum(C[i][j] * V_threshold[j][0] for j in range(n)) - V_threshold[i][0] +
     sum(D[i][j] * p[j][0] for j in range(len(p))) for i in range(n)]

# Transition kernel (P(x))
Cx = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
phi_x = model.addVars(n, vtype=GRB.BINARY)
Bphi = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
P_x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

for i in range(n):
    model.addConstr(Cx[i] == gp.quicksum(C[i][j] * x[j] for j in range(n)))
    model.addGenConstrIndicator(phi_x[i], True, x[i] <= 0)
    model.addConstr(Bphi[i] == gp.quicksum(B[i][j] * phi_x[j] for j in range(n)))
    model.addConstr(P_x[i] == Cx[i] + r[i] - Bphi[i])

# neural network
h = 5
W1 = model.addVars(n, h, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W1")
W2 = model.addVars(h, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W2")

# neural output of V(x)

hidden_layer_x = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
relu_layer_x = model.addVars(h, lb=0, ub=GRB.INFINITY)
V_x = model.addVar(lb=0, ub=GRB.INFINITY, name="V_x")

for j in range(h):
    model.addConstr(hidden_layer_x[j] == gp.quicksum(W1[i, j] * x[i] for i in range(n)))
    model.addGenConstrMax(relu_layer_x[j], [hidden_layer_x[j], 0])

model.addConstr(V_x == gp.quicksum(W2[i, 0] * relu_layer_x[i] for i in range(h)))

# state noise

x_tplus1_up = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
x_tplus1_down = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

for i in range(n):
    model.addConstr(x_tplus1_up[i] == 1.1 * P_x[i])
    model.addConstr(x_tplus1_down[i] == 0.9 * P_x[i])

# expected neural output of V(x_{t+1})

hidden_layer_Px_up = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
relu_layer_Px_up = model.addVars(h, lb=0, ub=GRB.INFINITY)
V_Px_up = model.addVar(lb=0, ub=GRB.INFINITY)

hidden_layer_Px_down = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
relu_layer_Px_down = model.addVars(h, lb=0, ub=GRB.INFINITY)
V_Px_down = model.addVar(lb=0, ub=GRB.INFINITY)

for j in range(h):
    model.addConstr(hidden_layer_Px_up[j] == gp.quicksum(W1[i, j] * x[i] for i in range(n)))
    model.addGenConstrMax(relu_layer_Px_up[j], [hidden_layer_Px_up[j], 0])
    model.addConstr(hidden_layer_Px_down[j] == gp.quicksum(W1[i, j] * x[i] for i in range(n)))
    model.addGenConstrMax(relu_layer_Px_down[j], [hidden_layer_Px_down[j], 0])

model.addConstr(V_Px_up == gp.quicksum(W2[i, 0] * relu_layer_Px_up[i] for i in range(h)))
model.addConstr(V_Px_down == gp.quicksum(W2[i, 0] * relu_layer_Px_down[i] for i in range(h)))

# supermartingale property
epsilon = model.addVar(lb=0, ub=GRB.INFINITY, name="epsilon")

E_V_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="E_V_X_tplus1")
model.addConstr(E_V_X_tplus1 == 0.5 * V_Px_up + 0.5 * V_Px_down)

model.addConstr(E_V_X_tplus1 >= V_x - epsilon)  # TODO: ask about constraint

# Solve the model
model.optimize()

# Check results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
    for v in model.getVars():
        if v.varName:
            print(f"{v.varName} = {v.x}")
else:
    print("No optimal solution found")