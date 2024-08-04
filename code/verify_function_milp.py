import gurobipy as gp
from gurobipy import GRB
from model_params import min_bound, max_bound


def verify_model_gurobi(n, h, equil_set, C, B, r, epsilon, W1, W2, B1, B2):
    model = gp.Model("verify_model")

    # State variables
    x = model.addVars(n, lb=min_bound, ub=max_bound, name="X")

    # Set A constraint (outside the equilibrium set)
    squared_distance = gp.quicksum((x[i] - equil_set.center[i])**2 for i in range(n))
    model.addConstr(squared_distance >= equil_set.radius**2 + 1e-6)  # ensure strict inequality

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
        model.addConstr(hidden_layer1_x[j] == gp.quicksum(W1[j][i] * x[i] + B1[j] for i in range(n)))
        model.addGenConstrMax(relu_layer_x[j], [hidden_layer1_x[j], 0])

    model.addConstr(hidden_layer2_x == gp.quicksum(W2[0][i] * relu_layer_x[i] + B2[0] for i in range(h)))
    model.addGenConstrMax(V_x, [hidden_layer2_x, 0])


    # expected neural output of V(P(x))

    hidden_layer_Px_up_up = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_up_up = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_up_up = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Px_down_down = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_down_down = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_down_down = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Px_up_down = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_up_down = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_up_down = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Px_down_up = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Px_down_up = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Px_down_up = model.addVar(lb=0, ub=GRB.INFINITY)

    for j in range(h):
        model.addConstr(hidden_layer_Px_up_up[j] == gp.quicksum(W1[j][i] * x_tplus1_up_up[i] + B1[j] for i in range(n)))
        model.addGenConstrMax(relu_layer_Px_up_up[j], [hidden_layer_Px_up_up[j], 0])
        model.addConstr(hidden_layer_Px_down_up[j] == gp.quicksum(W1[j][i] * x_tplus1_down_up[i] + B1[j] for i in range(n)))
        model.addGenConstrMax(relu_layer_Px_down_up[j], [hidden_layer_Px_down_up[j], 0])
        model.addConstr(hidden_layer_Px_up_down[j] == gp.quicksum(W1[j][i] * x_tplus1_up_down[i] + B1[j] for i in range(n)))
        model.addGenConstrMax(relu_layer_Px_up_down[j], [hidden_layer_Px_up_down[j], 0])
        model.addConstr(hidden_layer_Px_down_down[j] == gp.quicksum(W1[j][i] * x_tplus1_down_down[i] + B1[j] for i in range(n)))
        model.addGenConstrMax(relu_layer_Px_down_down[j], [hidden_layer_Px_down_down[j], 0])

    model.addConstr(V_Px_up_up == gp.quicksum(W2[0][i] * relu_layer_Px_up_up[i] + B2[0] for i in range(h)))
    model.addConstr(V_Px_down_up == gp.quicksum(W2[0][i] * relu_layer_Px_down_up[i] + B2[0] for i in range(h)))
    model.addConstr(V_Px_up_down == gp.quicksum(W2[0][i] * relu_layer_Px_up_down[i] + B2[0] for i in range(h)))
    model.addConstr(V_Px_down_down == gp.quicksum(W2[0][i] * relu_layer_Px_down_down[i] + B2[0] for i in range(h)))

    # supermartingale property

    E_V_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="E_V_X_tplus1")
    model.addConstr(E_V_X_tplus1 == 0.25 * V_Px_up_up + 0.25 * V_Px_down_down + 0.25 * V_Px_up_down + 0.25 * V_Px_down_up)

    model.addConstr(E_V_X_tplus1 >= V_x - epsilon + 1e-6)

    # Solve the model
    model.optimize()

    # Check if a solution was found
    if model.status == GRB.OPTIMAL:
        counterexample = [x[i].X for i in range(n)]
        #print(counterexample)
        # print("Cx: " + str([Cx[i].X for i in range(n)]))
        # #print([phi_x[i].X for i in range(n)])
        # print("Bphi: " + str([Bphi[i].X for i in range(n)]))
        # print("r: " + str(r))
        ## print("Px: " + str([P_x[i].X for i in range(n)]))

        print(str(V_x.X))
        print(str([x_tplus1_up_up[i].X for i in range(n)]) + "  ->  " + str(V_Px_up_up.X))
        print(str([x_tplus1_down_up[i].X for i in range(n)]) + "  ->  " + str(V_Px_down_up.X))
        print(str([x_tplus1_up_down[i].X for i in range(n)]) + "  ->  " + str(V_Px_up_down.X))
        print(str([x_tplus1_down_down[i].X for i in range(n)]) + "  ->  " + str(V_Px_down_down.X))
        print(E_V_X_tplus1.X)
        print("milp ^^^")
        return True, counterexample
    else:
        return False, None
