"""This function seeks to find the upper-bound for the reward function E[R(x)] = E[V(X')] - V(x) where X' ~ xP"""
import gurobipy as gp
from gurobipy import GRB
from model_params import min_bound, max_bound


def find_reward_upper_bound(n, h, C, B, r, W1, W2, B1, B2):
    model = gp.Model()

    # State variables
    x = model.addVars(n, lb=min_bound, ub=max_bound, name="X")


    # Transition kernel P(x)
    C = C.tolist()
    B = B.tolist()
    r = r.T.tolist()[0]

    Cx = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    phi_x = model.addVars(n, vtype=GRB.BINARY)
    Bphi = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    P_x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Px")
    M = 1e6

    for i in range(n):
        model.addConstr(Cx[i] == gp.quicksum(C[i][j] * x[j] for j in range(n)))
        model.addConstr(x[i] + M * phi_x[i] >= 0)
        model.addConstr(x[i] <= M * (1 - phi_x[i]))
        model.addConstr(Bphi[i] == gp.quicksum(B[i][j] * phi_x[j] for j in range(n)))
        model.addConstr(P_x[i] == Cx[i] + r[i] - Bphi[i])


    # State noise

    x_tplus1_up_up = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x_tplus1_down_up = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x_tplus1_up_down = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x_tplus1_down_down = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    for i in range(n):
        model.addConstr(x_tplus1_up_up[i] == 1.1 * P_x[i])
        model.addConstr(x_tplus1_down_down[i] == 0.9 * P_x[i])

    model.addConstr(x_tplus1_up_down[0] == 1.1 * P_x[0])
    model.addConstr(x_tplus1_up_down[1] == 0.9 * P_x[1])
    model.addConstr(x_tplus1_down_up[0] == 0.9 * P_x[0])
    model.addConstr(x_tplus1_down_up[1] == 1.1 * P_x[1])


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

    model.addConstr(hidden_layer2_x == gp.quicksum(W2[0][i] * relu_layer_x[i] for i in range(h))+ B2[0])
    model.addConstr(V_x == gp.max_(hidden_layer2_x, constant=0))


    # expected neural output of V(P(x))

    hidden_layer_Px_up_up = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Px_up_up2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_up_up = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_up_up = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Px_down_down = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Px_down_down2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_down_down = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_down_down = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Px_up_down = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Px_up_down2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_up_down = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_up_down = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Px_down_up = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Px_down_up2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_down_up = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_down_up = model.addVar(lb=0, ub=GRB.INFINITY)

    for j in range(h):
        model.addConstr(hidden_layer_Px_up_up[j] == gp.quicksum(W1[j][i] * x_tplus1_up_up[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Px_up_up[j] == gp.max_(hidden_layer_Px_up_up[j], constant=0))
        model.addConstr(hidden_layer_Px_down_up[j] == gp.quicksum(W1[j][i] * x_tplus1_down_up[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Px_down_up[j] == gp.max_(hidden_layer_Px_down_up[j], constant=0))
        model.addConstr(hidden_layer_Px_up_down[j] == gp.quicksum(W1[j][i] * x_tplus1_up_down[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Px_up_down[j] == gp.max_(hidden_layer_Px_up_down[j], constant=0))
        model.addConstr(hidden_layer_Px_down_down[j] == gp.quicksum(W1[j][i] * x_tplus1_down_down[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Px_down_down[j] == gp.max_(hidden_layer_Px_down_down[j], constant=0))

    model.addConstr(hidden_layer_Px_up_up2 == gp.quicksum(W2[0][i] * relu_layer_Px_up_up[i] for i in range(h)) + B2[0])
    model.addConstr(hidden_layer_Px_down_up2 == gp.quicksum(W2[0][i] * relu_layer_Px_down_up[i] for i in range(h)) + B2[0])
    model.addConstr(hidden_layer_Px_up_down2 == gp.quicksum(W2[0][i] * relu_layer_Px_up_down[i] for i in range(h)) + B2[0])
    model.addConstr(hidden_layer_Px_down_down2 == gp.quicksum(W2[0][i] * relu_layer_Px_down_down[i] for i in range(h)) + B2[0])

    model.addConstr(V_Px_down_up == gp.max_(hidden_layer_Px_down_up2, constant=0))
    model.addConstr(V_Px_up_down == gp.max_(hidden_layer_Px_up_down2, constant=0))
    model.addConstr(V_Px_up_up == gp.max_(hidden_layer_Px_up_up2, constant=0))
    model.addConstr(V_Px_down_down == gp.max_(hidden_layer_Px_down_down2, constant=0))

    # supermartingale property

    E_V_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="E_V_X_tplus1")
    model.addConstr(E_V_X_tplus1 == 0.25 * V_Px_up_up + 0.25 * V_Px_down_down + 0.25 * V_Px_up_down + 0.25 * V_Px_down_up)

    model.setObjective(E_V_X_tplus1 - V_x, GRB.MAXIMIZE)

    # Solve the model
    model.optimize()

    # Check if a solution was found
    if model.status == GRB.OPTIMAL:
        print(x[0].X)
        print(x[1].X)
        return model.objVal
    else:

        return False, None
