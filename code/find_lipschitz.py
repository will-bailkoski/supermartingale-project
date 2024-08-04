import gurobipy as gp
from gurobipy import GRB
import numpy as np


def calculate_lipschitz_constant(W1, b1, W2, b2):

    model = gp.Model('LipschitzConstant')

    n = W1.shape[1]
    h = W2.shape[0]
    print(n, h)

    x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')

    # layers are calculated without ReLu. we are calculating worst-case gradient, which for ReLu is 1
    z1 = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z1')
    for i in range(h):
        model.addConstr(z1[i] == gp.quicksum(W1[i, j] * x[j] for j in range(n)) + b1[i])
    z2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z2')
    model.addConstr(z2 == gp.quicksum(W2[0, j] * z1[j] for j in range(h)) + b2[0])

    grad_x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='grad_x')

    # gradient of the output with respect to the input
    for j in range(n):
        grad_x[j] = gp.quicksum(W2[i, k] * W1[k, j] for i in range(1) for k in range(h))

    # norm of the gradient
    norm_grad = gp.quicksum(grad_x[j] * grad_x[j] for j in range(n))

    # objective to maximize the norm of the gradient
    model.setObjective(norm_grad, GRB.MAXIMIZE)

    # vector must lie within the unit norm ball
    model.addQConstr(gp.quicksum(x[j] * x[j] for j in range(n)) <= 1, name="unit_norm")

    model.optimize()
    return np.sqrt(model.objVal)
