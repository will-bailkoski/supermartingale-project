"""This function seeks to find the upper-bound for the reward function E[R(x)] = E[V(X')] - V(x) where X' ~ xP"""
import gurobipy as gp
from gurobipy import GRB
from itertools import product


def find_reward_bound(bounds, input_size, layer_sizes, C, B, r, weights, biases, upper):
    model = gp.Model()
    model.setParam(GRB.Param.OutputFlag, 0)

    # State variables
    x = model.addVars(
        input_size,
        lb={i: bounds[i][0] for i in range(input_size)},
        ub={i: bounds[i][1] for i in range(input_size)},
        name="X"
    )

    # Transition kernel P(x)
    C = C.tolist()
    B = B.tolist()
    r = r.T.tolist()[0]

    Cx = model.addVars(input_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    phi_x = model.addVars(input_size, vtype=GRB.BINARY)
    Bphi = model.addVars(input_size, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    P_x = model.addVars(input_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Px")
    M = 1e6

    for i in range(input_size):
        model.addConstr(Cx[i] == gp.quicksum(C[i][j] * x[j] for j in range(input_size)))
        model.addConstr(x[i] + M * phi_x[i] >= 0)
        model.addConstr(x[i] <= M * (1 - phi_x[i]))
        model.addConstr(Bphi[i] == gp.quicksum(B[i][j] * phi_x[j] for j in range(input_size)))
        model.addConstr(P_x[i] == Cx[i] + r[i] - Bphi[i])

    # Neural network V(x)
    activations = [x]
    for layer_idx in range(len(layer_sizes) - 1):
        W = weights[layer_idx]
        b = biases[layer_idx]

        hidden_layer = model.addVars(
            layer_sizes[layer_idx + 1], lb=-GRB.INFINITY, ub=GRB.INFINITY)
        relu_layer = model.addVars(
            layer_sizes[layer_idx + 1], lb=0, ub=GRB.INFINITY)

        for j in range(layer_sizes[layer_idx + 1]):
            model.addConstr(
                hidden_layer[j] == gp.quicksum(W[j][i] * activations[-1][i] for i in range(layer_sizes[layer_idx])) + b[j]
            )
            model.addConstr(relu_layer[j] == gp.max_(hidden_layer[j], constant=0))

        activations.append(relu_layer)

    V_x = model.addVar(lb=0, ub=GRB.INFINITY, name="V_x")
    model.addConstr(V_x == gp.max_(activations[-1][0], constant=0))


    # State noise
    perturbation_cases = list(product([1.1, 0.9], repeat=input_size))  # All combinations of +/-10% for n dimensions
    x_tplus1_cases = [
        model.addVars(input_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_tplus1_case_{case_idx}")
        for case_idx in range(len(perturbation_cases))
    ]

    # Add constraints for each case
    for case_idx, perturbations in enumerate(perturbation_cases):
        for i in range(input_size):
            model.addConstr(x_tplus1_cases[case_idx][i] == perturbations[i] * P_x[i])

    # Expected neural output of V(P(x))
    V_Px_cases = []
    for case_idx, x_tplus1 in enumerate(x_tplus1_cases):
        activations = [x_tplus1]
        for layer_idx in range(len(weights)):
            W = weights[layer_idx]
            b = biases[layer_idx]
            layer_size = layer_sizes[layer_idx + 1]

            # Hidden layer and ReLU layer variables
            hidden_layer = model.addVars(layer_size, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                         name=f"hidden_layer_case_{case_idx}_layer_{layer_idx}")
            relu_layer = model.addVars(layer_size, lb=0, ub=GRB.INFINITY,
                                       name=f"relu_layer_case_{case_idx}_layer_{layer_idx}")

            for j in range(layer_size):
                # Hidden layer computation
                model.addConstr(hidden_layer[j] == gp.quicksum(
                    W[j][i] * activations[-1][i] for i in range(layer_sizes[layer_idx])) + b[j])
                # ReLU activation
                model.addConstr(relu_layer[j] == gp.max_(hidden_layer[j], constant=0))

            # Add ReLU layer as the next layer of activations
            activations.append(relu_layer)

        # Output layer value for this case
        V_Px = model.addVar(lb=0, ub=GRB.INFINITY, name=f"V_Px_case_{case_idx}")
        model.addConstr(V_Px == gp.max_(activations[-1][0], constant=0))  # Assuming a single output neuron
        V_Px_cases.append(V_Px)

    maxV_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="maxV_X_tplus1")
    minV_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="minV_X_tplus1")

    # Max and Min constraints
    model.addConstr(maxV_X_tplus1 == gp.max_(*V_Px_cases))
    model.addConstr(minV_X_tplus1 == gp.min_(*V_Px_cases))

    if upper:
        model.setObjective(maxV_X_tplus1 - V_x, GRB.MAXIMIZE)
    else:
        model.setObjective(minV_X_tplus1 - V_x, GRB.MINIMIZE)

    # Solve the model
    model.optimize()

    import numpy as np

    return model.objVal, [x[i].X for i in range(input_size)]
