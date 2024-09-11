"""This function seeks to find the Lipschitz constant for the estimated reward, E[R(x)] = E[V(X')] - V(x) where X' ~ xP
Use num_points parameter to perform a check on a random pair of points. To skip this step, set num_points to 0"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from functools import partial
from function_application import e_r_x

def calculate_lipschitz_constant(n, h, C, B, r, W1, W2, B1, B2, domain_bounds, num_points):

    model = gp.Model("Lipschitz Constant Calculation")

    x = model.addVars(n, lb=domain_bounds[0], ub=domain_bounds[1], name="X")
    y = model.addVars(n, lb=domain_bounds[0], ub=domain_bounds[1], name="Y")

    # Transition kernel P(x)
    C = C.tolist()
    B = B.tolist()
    r = r.T.tolist()[0]

    Cx = [None] * n
    phi_x = model.addVars(n, vtype=GRB.BINARY)
    Bphi = [None] * n
    P_x = [None] * n
    M = 1e6

    for i in range(n):
        Cx[i] = gp.quicksum(C[i][j] * x[j] for j in range(n))
        model.addConstr(x[i] + M * phi_x[i] >= 0)
        model.addConstr(x[i] <= M * (1 - phi_x[i]))
        Bphi[i] = gp.quicksum(B[i][j] * phi_x[j] for j in range(n))
        P_x[i] = Cx[i] + r[i] - Bphi[i]

    # State noise

    x_tplus1_up_up = [None] * n
    x_tplus1_down_up = [None] * n
    x_tplus1_up_down = [None] * n
    x_tplus1_down_down = [None] * n

    for i in range(n):
        x_tplus1_up_up[i] = 1.1 * P_x[i]
        x_tplus1_down_down[i] = 0.9 * P_x[i]

    x_tplus1_up_down[0] = 1.1 * P_x[0]
    x_tplus1_up_down[1] = 0.9 * P_x[1]
    x_tplus1_down_up[0] = 0.9 * P_x[0]
    x_tplus1_down_up[1] = 1.1 * P_x[1]

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
        model.addConstr(
            hidden_layer_Px_down_up[j] == gp.quicksum(W1[j][i] * x_tplus1_down_up[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Px_down_up[j] == gp.max_(hidden_layer_Px_down_up[j], constant=0))
        model.addConstr(
            hidden_layer_Px_up_down[j] == gp.quicksum(W1[j][i] * x_tplus1_up_down[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Px_up_down[j] == gp.max_(hidden_layer_Px_up_down[j], constant=0))
        model.addConstr(
            hidden_layer_Px_down_down[j] == gp.quicksum(W1[j][i] * x_tplus1_down_down[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Px_down_down[j] == gp.max_(hidden_layer_Px_down_down[j], constant=0))

    model.addConstr(hidden_layer_Px_up_up2 == gp.quicksum(W2[0][i] * relu_layer_Px_up_up[i] for i in range(h)) + B2[0])
    model.addConstr(
        hidden_layer_Px_down_up2 == gp.quicksum(W2[0][i] * relu_layer_Px_down_up[i] for i in range(h)) + B2[0])
    model.addConstr(
        hidden_layer_Px_up_down2 == gp.quicksum(W2[0][i] * relu_layer_Px_up_down[i] for i in range(h)) + B2[0])
    model.addConstr(
        hidden_layer_Px_down_down2 == gp.quicksum(W2[0][i] * relu_layer_Px_down_down[i] for i in range(h)) + B2[0])

    model.addConstr(V_Px_down_up == gp.max_(hidden_layer_Px_down_up2, constant=0))
    model.addConstr(V_Px_up_down == gp.max_(hidden_layer_Px_up_down2, constant=0))
    model.addConstr(V_Px_up_up == gp.max_(hidden_layer_Px_up_up2, constant=0))
    model.addConstr(V_Px_down_down == gp.max_(hidden_layer_Px_down_down2, constant=0))

    # supermartingale property

    E_V_X_tplus1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="E_V_X_tplus1")
    model.addConstr(
        E_V_X_tplus1 == 0.25 * V_Px_up_up + 0.25 * V_Px_down_down + 0.25 * V_Px_up_down + 0.25 * V_Px_down_up)


#### Y

    Cy = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    phi_y = model.addVars(n, vtype=GRB.BINARY)
    Bphi_y = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    P_y = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Px")


    for i in range(n):
        model.addConstr(Cy[i] == gp.quicksum(C[i][j] * y[j] for j in range(n)))
        model.addConstr(y[i] + M * phi_y[i] >= 0)
        model.addConstr(y[i] <= M * (1 - phi_y[i]))
        model.addConstr(Bphi_y[i] == gp.quicksum(B[i][j] * phi_y[j] for j in range(n)))
        model.addConstr(P_y[i] == Cy[i] + r[i] - Bphi_y[i])

    # State noise

    y_tplus1_up_up = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y_tplus1_down_up = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y_tplus1_up_down = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y_tplus1_down_down = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    for i in range(n):
        model.addConstr(y_tplus1_up_up[i] == 1.1 * P_y[i])
        model.addConstr(y_tplus1_down_down[i] == 0.9 * P_y[i])

    model.addConstr(y_tplus1_up_down[0] == 1.1 * P_y[0])
    model.addConstr(y_tplus1_up_down[1] == 0.9 * P_y[1])
    model.addConstr(y_tplus1_down_up[0] == 0.9 * P_y[0])
    model.addConstr(y_tplus1_down_up[1] == 1.1 * P_y[1])

    hidden_layer1_y = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_y = model.addVars(h, lb=0, ub=GRB.INFINITY)
    hidden_layer2_y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    V_y = model.addVar(lb=0, ub=GRB.INFINITY, name="V_y")

    for j in range(h):
        model.addConstr(hidden_layer1_y[j] == gp.quicksum(W1[j][i] * y[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_y[j] == gp.max_(hidden_layer1_y[j], constant=0))

    model.addConstr(hidden_layer2_y == gp.quicksum(W2[0][i] * relu_layer_y[i] for i in range(h)) + B2[0])
    model.addConstr(V_y == gp.max_(hidden_layer2_y, constant=0))

    # expected neural output of V(P(y))

    hidden_layer_Py_up_up = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Py_up_up2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Py_up_up = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Py_up_up = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Py_down_down = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Py_down_down2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Py_down_down = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Py_down_down = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Py_up_down = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Py_up_down2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Py_up_down = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Py_up_down = model.addVar(lb=0, ub=GRB.INFINITY)

    hidden_layer_Py_down_up = model.addVars(h, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    hidden_layer_Py_down_up2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    relu_layer_Py_down_up = model.addVars(h, lb=0, ub=GRB.INFINITY)
    V_Py_down_up = model.addVar(lb=0, ub=GRB.INFINITY)

    for j in range(h):
        model.addConstr(hidden_layer_Py_up_up[j] == gp.quicksum(W1[j][i] * y_tplus1_up_up[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Py_up_up[j] == gp.max_(hidden_layer_Py_up_up[j], constant=0))
        model.addConstr(
            hidden_layer_Py_down_up[j] == gp.quicksum(W1[j][i] * y_tplus1_down_up[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Py_down_up[j] == gp.max_(hidden_layer_Py_down_up[j], constant=0))
        model.addConstr(
            hidden_layer_Py_up_down[j] == gp.quicksum(W1[j][i] * y_tplus1_up_down[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Py_up_down[j] == gp.max_(hidden_layer_Py_up_down[j], constant=0))
        model.addConstr(
            hidden_layer_Py_down_down[j] == gp.quicksum(W1[j][i] * y_tplus1_down_down[i] for i in range(n)) + B1[j])
        model.addConstr(relu_layer_Py_down_down[j] == gp.max_(hidden_layer_Py_down_down[j], constant=0))

    model.addConstr(hidden_layer_Py_up_up2 == gp.quicksum(W2[0][i] * relu_layer_Py_up_up[i] for i in range(h)) + B2[0])
    model.addConstr(
        hidden_layer_Py_down_up2 == gp.quicksum(W2[0][i] * relu_layer_Py_down_up[i] for i in range(h)) + B2[0])
    model.addConstr(
        hidden_layer_Py_up_down2 == gp.quicksum(W2[0][i] * relu_layer_Py_up_down[i] for i in range(h)) + B2[0])
    model.addConstr(
        hidden_layer_Py_down_down2 == gp.quicksum(W2[0][i] * relu_layer_Py_down_down[i] for i in range(h)) + B2[0])

    model.addConstr(V_Py_down_up == gp.max_(hidden_layer_Py_down_up2, constant=0))
    model.addConstr(V_Py_up_down == gp.max_(hidden_layer_Py_up_down2, constant=0))
    model.addConstr(V_Py_up_up == gp.max_(hidden_layer_Py_up_up2, constant=0))
    model.addConstr(V_Py_down_down == gp.max_(hidden_layer_Py_down_down2, constant=0))

    # supermartingale property

    E_V_Y_tplus1 = 0.25 * V_Py_up_up + 0.25 * V_Py_down_down + 0.25 * V_Py_up_down + 0.25 * V_Py_down_up

    model.addConstr((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) == 1)

    Rx = E_V_X_tplus1 - V_x
    Ry = E_V_Y_tplus1 - V_y

    numer = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    model.addConstr(numer <= Ry - Rx)
    model.addConstr(numer >= Rx - Ry)

    model.setObjective(numer, GRB.MAXIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:

        L = model.objVal

        if bool(num_points):
            EVP = partial(e_r_x, C=np.array(C), B=np.array(B), r=np.array([r]).T, W1=np.array(W1), W2=np.array(W2), B1=np.array([B1]).T, B2=np.array([B2]).T)
            is_lipschitz = lipschitz_check(L, EVP, num_points, n, domain_bounds)
        else:
            is_lipschitz = True

        if is_lipschitz:
            return model.objVal
        else:
            assert is_lipschitz, "lipschitz test failed"
            return
    else:
        return None


def lipschitz_check(L, function, num_points, n, domain_bounds):
    # Generate random points within 2D bounds
    x_bounds = domain_bounds
    y_bounds = domain_bounds

    # Generate random points within the specified bounds
    points = np.random.rand(num_points, n)
    points[:, 0] = points[:, 0] * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    points[:, 1] = points[:, 1] * (y_bounds[1] - y_bounds[0]) + y_bounds[0]

    # Verify the Lipschitz constant
    is_lipschitz = True

    for i in range(num_points):
        if i % 100 == 0:
            print(i)
        for j in range(i + 1, num_points):
            x1, y1 = points[i]
            x2, y2 = points[j]

            # Calculate the Euclidean distance between points
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if distance == 0:
                continue

            # Calculate the difference in function values
            diff_f = np.abs(
                function(np.array([[x1], [y1]])) - function(np.array([[x2], [y2]])))

            # Check if the Lipschitz condition is violated
            if diff_f > L * distance:
                is_lipschitz = False
                break

    return is_lipschitz
