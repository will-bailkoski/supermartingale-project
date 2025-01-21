"""Functions for validating a CEGIS supermartingale via various solvers"""
from itertools import product

import gurobipy as gp
from gurobipy import GRB

from z3 import *

def verify_model_milp(n, h, C, B, r, epsilon, W1, W2, B1, B2, domain):
    """A function for validating a CEGIS supermartingale via Gurobi's MILP solver"""
    model = gp.Model("verify_model")

    # State variables
    x = model.addVars(n, lb=domain[0], ub=domain[1], name="X")
    model.setParam(GRB.Param.OutputFlag, 0)

    z = model.addVar(vtype=GRB.BINARY)  # Binary variable
    M = 1e6 * abs(domain[1])
    model.addConstr(x[0] >= -M * z)
    model.addConstr(x[1] >= -M * (1 - z))


    # Transition kernel P(x)
    C = C.tolist()
    B = B.tolist()
    r = r.T.tolist()[0]

    Cx = [None] * n
    phi_x = model.addVars(n, vtype=GRB.BINARY)
    Bphi = [None] * n
    P_x = [None] * n

    for i in range(n):
        Cx[i] = gp.quicksum(C[i][j] * x[j] for j in range(n))
        model.addConstr(x[i] + M * phi_x[i] >= 0)
        model.addConstr(x[i] <= M * (1 - phi_x[i]))
        Bphi[i] = gp.quicksum(B[i][j] * phi_x[j] for j in range(n))
        P_x[i] = Cx[i] + r[i] - Bphi[i]


    # State noise

    scaling_factors = list(product([1.1, 0.9], repeat=n))

    # Create variables for each possible outcome
    x_tplus1 = {}
    for index, factors in enumerate(scaling_factors):
        x_tplus1[index] = [None] * n
        for i in range(n):
            x_tplus1[index][i] = factors[i] * P_x[i]


    # Neural network V(x)

    W1 = W1.tolist()
    W2 = W2.tolist()
    B1 = B1.tolist()
    B2 = B2.tolist()

    hidden_layer1_x = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_x = model.addVars(h, lb=0, ub=GRB.INFINITY)
    hidden_layer2_x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    V_x = model.addVar(lb=0, ub=GRB.INFINITY, name="V_x")

    for j in range(h):
        model.addConstr(hidden_layer1_x[j] == gp.quicksum(W1[j][i] * x[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_x[j] == gp.max_(hidden_layer1_x[j], constant=0))

    model.addConstr(hidden_layer2_x == gp.quicksum(W2[0][i] * relu_layer_x[i] for i in range(h)) + B2[0])
    model.addConstr(V_x == gp.max_(hidden_layer2_x, constant=0))

    # expected neural output of V(P(x))

    hidden_layer_x_tplus1 = {}
    hidden_layer_x_tplus1_relu = {}
    V_x_tplus1 = {}
    for index in x_tplus1.keys():
        hidden_layer_x_tplus1[index] = [None] * h
        for j in range(h):
            hidden_layer_x_tplus1[index][j] = gp.quicksum(W1[j][i] * x_tplus1[index][i] for i in range(n)) + B1[j]

        hidden_layer_x_tplus1_relu[index] = model.addVars(h, lb=0, ub=GRB.INFINITY)
        model.addConstr(hidden_layer_x_tplus1_relu[index] == [gp.max_(hidden_layer_x_tplus1[index][j], constant=0) for j in range(h)])

        V_x_tplus1[index] = gp.max_(gp.quicksum(W2[0][i] * hidden_layer_x_tplus1_relu[index][i] for i in range(h)) + B2[0], constant=0)


    # supermartingale property
    # Compute the expected value as the average of all V_x_tplus1 values
    E_V_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    values = [V_x_tplus1[index] for index in V_x_tplus1.keys()]
    model.addConstr(
        E_V_X_tplus1 == gp.quicksum(values) / (2 ** n)
    )
    model.addConstr(E_V_X_tplus1 - V_x >= - epsilon)

    # Solve the model
    model.optimize()

    # Check if a solution was found
    if model.status == GRB.OPTIMAL:
        counterexample = [x[i].X for i in range(n)]
        return True, counterexample
    else:

        return False, None


def verify_model_milp2(n, h, C, B, r, epsilon, W1, W2, B1, B2, domain):
    """A function for validating a CEGIS supermartingale via Gurobi's MILP solver"""
    model = gp.Model("verify_model")

    # State variables
    x = model.addVars(n, lb=domain[0], ub=domain[1], name="X")
    model.setParam(GRB.Param.OutputFlag, 0)

    z = model.addVar(vtype=GRB.BINARY)  # Binary variable
    M = 1e6 * abs(domain[1])
    model.addConstr(x[0] >= -M * z)
    model.addConstr(x[1] >= -M * (1 - z))

    # Transition kernel P(x)
    C = C.tolist()
    B = B.tolist()
    r = r.T.tolist()[0]

    Cx = [None] * n
    phi_x = model.addVars(n, vtype=GRB.BINARY)
    Bphi = [None] * n
    P_x = [None] * n

    for i in range(n):
        Cx[i] = gp.quicksum(C[i][j] * x[j] for j in range(n))
        model.addConstr(x[i] + M * phi_x[i] >= 0)
        model.addConstr(x[i] <= M * (1 - phi_x[i]))
        Bphi[i] = gp.quicksum(B[i][j] * phi_x[j] for j in range(n))
        P_x[i] = Cx[i] + r[i] - Bphi[i]

    # State noise
    scaling_factors = list(product([0.1, -0.1], repeat=n))

    # Create variables for each possible outcome
    x_tplus1 = {}
    for index, factors in enumerate(scaling_factors):
        x_tplus1[index] = [None] * n
        for i in range(n):
            x_tplus1[index][i] = factors[i] * x[i] + P_x[i]

    # Neural network V(x)
    W1 = W1.tolist()
    W2 = W2.tolist()
    B1 = B1.tolist()
    B2 = B2.tolist()

    hidden_layer1_x = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_x = model.addVars(h, lb=0, ub=GRB.INFINITY)
    hidden_layer2_x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    V_x = model.addVar(lb=0, ub=GRB.INFINITY, name="V_x")

    for j in range(h):
        model.addConstr(hidden_layer1_x[j] == gp.quicksum(W1[j][i] * x[i] for i in range(n)) + B1[j])

        # Replace relu_layer_x[j] == max(hidden_layer1_x[j], 0)
        delta_relu = model.addVar(vtype=GRB.BINARY)  # Binary variable
        model.addConstr(relu_layer_x[j] >= hidden_layer1_x[j])
        model.addConstr(relu_layer_x[j] >= 0)
        model.addConstr(relu_layer_x[j] <= hidden_layer1_x[j] + M * (1 - delta_relu))
        model.addConstr(relu_layer_x[j] <= 0 + M * delta_relu)

    # Replace hidden_layer2_x == max(hidden_layer2_x, 0)
    model.addConstr(hidden_layer2_x == gp.quicksum(W2[0][i] * relu_layer_x[i] for i in range(h)) + B2[0])
    delta_V = model.addVar(vtype=GRB.BINARY)  # Binary variable for V_x
    model.addConstr(V_x >= hidden_layer2_x)
    model.addConstr(V_x >= 0)
    model.addConstr(V_x <= hidden_layer2_x + M * (1 - delta_V))
    model.addConstr(V_x <= 0 + M * delta_V)

    # Expected neural output of V(P(x))
    hidden_layer_x_tplus1 = {}
    hidden_layer_x_tplus1_relu = {}
    V_x_tplus1 = {}
    for index in x_tplus1.keys():
        hidden_layer_x_tplus1[index] = [None] * h
        for j in range(h):
            hidden_layer_x_tplus1[index][j] = gp.quicksum(W1[j][i] * x_tplus1[index][i] for i in range(n)) + B1[j]

        hidden_layer_x_tplus1_relu[index] = model.addVars(h, lb=0, ub=GRB.INFINITY)
        for j in range(h):
            # Replace relu_layer_x_tplus1[j] == max(hidden_layer_x_tplus1[index][j], 0)
            delta_relu_tplus1 = model.addVar(vtype=GRB.BINARY)
            model.addConstr(hidden_layer_x_tplus1_relu[index][j] >= hidden_layer_x_tplus1[index][j])
            model.addConstr(hidden_layer_x_tplus1_relu[index][j] >= 0)
            model.addConstr(hidden_layer_x_tplus1_relu[index][j] <= hidden_layer_x_tplus1[index][j] + M * (1 - delta_relu_tplus1))
            model.addConstr(hidden_layer_x_tplus1_relu[index][j] <= 0 + M * delta_relu_tplus1)

        V_x_tplus1[index] = model.addVar(lb=0, ub=GRB.INFINITY)  # Create a variable for V_x_tplus1[index]
        # Replace V_x_tplus1[index] == max(sum(...), 0)
        delta_V_tplus1 = model.addVar(vtype=GRB.BINARY)
        model.addConstr(V_x_tplus1[index] >= gp.quicksum(W2[0][i] * hidden_layer_x_tplus1_relu[index][i] for i in range(h)) + B2[0])
        model.addConstr(V_x_tplus1[index] >= 0)
        model.addConstr(V_x_tplus1[index] <= gp.quicksum(W2[0][i] * hidden_layer_x_tplus1_relu[index][i] for i in range(h)) + B2[0] + M * (1 - delta_V_tplus1))
        model.addConstr(V_x_tplus1[index] <= 0 + M * delta_V_tplus1)

    # Supermartingale property
    # Compute the expected value as the average of all V_x_tplus1 values
    E_V_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    values = [V_x_tplus1[index] for index in V_x_tplus1.keys()]
    model.addConstr(
        E_V_X_tplus1 == gp.quicksum(values) / (2 ** n)
    )
    model.addConstr(E_V_X_tplus1 >= V_x - epsilon + 1e-6)

    # Solve the model
    model.optimize()

    # Check if a solution was found
    if model.status == GRB.OPTIMAL:
        counterexample = [x[i].X for i in range(n)]
        return True, counterexample
    else:
        return False, None


# A function for validating a CEGIS supermartingale via Z3's SAT solver

def z3_value_to_double(val):
    if is_real(val):
        return val.numerator().as_long() / val.denominator().as_long()
    else:
        try:
            return float(str(val))
        except ValueError:
            raise ValueError(f"Cannot convert {val} of type {type(val)} to double")


def verify_model_sat(n, h, C, B, r, epsilon, W1, W2, B1, B2, domain):
    """A function for validating a CEGIS supermartingale via Z3's SAT solver"""
    solver = Solver()

    # state
    x = [Real(f"X_{i}") for i in range(n)]

    # Add constraints to the solver to bound each value in x
    for xi in x:
        solver.add(xi >= domain[0])
        solver.add(xi <= domain[1])

    # set A
    # squared_distance = sum((p_ - c[0]) ** 2 for p_, c in zip(x, equil_set.center))
    # solver.add(squared_distance > equil_set.radius ** 2)

    solver.add(z3.Or(x[0] >= 0, x[1] >= 0))

    # model parameters
    C = C.tolist()
    B = B.tolist()
    r = r.T[0].tolist()

    # transition kernel
    def P(x):
        Cx = [Sum([C[i][j] * x[j] for j in range(n)]) for i in range(n)]
        phi_x = [If(x[i] < 0, 1, 0) for i in range(n)]
        Bphi = [Sum([B[i][j] * phi_x[j] for j in range(n)]) for i in range(n)]
        return [Cx[i] + r[i] - Bphi[i] for i in range(n)]

    # neural network weights and biases
    w1 = W1.tolist()
    w2 = W2.tolist()
    b1 = B1.tolist()
    b2 = B2.tolist()

    # relu function
    def relu(x):
        return If(x > 0, x, 0)

    # neural network
    def V(x):
        layer1 = [Sum([w1[j][i] * x[i] for i in range(n)]) + b1[j] for j in range(h)]
        relu_layer = [relu(val) for val in layer1]
        layer2 = Sum([w2[0][i] * relu_layer[i] for i in range(h)]) + b2[0]
        return relu(layer2)

    # supermartingale properties
    V_x = V(x)

    solver.add(epsilon > 0)

    x_tplus1 = P(x)
    x_tplus1_up = [i * RealVal('1.1') for i in x_tplus1]
    x_tplus1_down = [i * RealVal('0.9') for i in x_tplus1]
    E_V_X_tplus1 = 0.25 * V(x_tplus1_up) + 0.25 * V(x_tplus1_down) + 0.25 * V([x_tplus1_up[0], x_tplus1_down[1]]) + 0.25 * V([x_tplus1_down[0], x_tplus1_up[1]])

    solver.add(E_V_X_tplus1 > V_x - epsilon)

    # Check satisfiability
    result = solver.check()



    if result == sat:
        model = solver.model()
        # print(z3_value_to_double(model.eval(E_V_X_tplus1)))
        counterexample_dict = {str(d): model[d] for d in model.decls() if d.name().startswith('X_')}
        counterexample = [z3_value_to_double(counterexample_dict[key]) for key in sorted(counterexample_dict.keys())]
        return True, counterexample
    else:
        return False, None
