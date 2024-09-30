"""Functions for validating a CEGIS supermartingale via various solvers"""
import gurobipy as gp
from gurobipy import GRB

from z3 import *

from model_params import min_bound, max_bound


def verify_model_milp(n, h, C, B, r, epsilon, W1, W2, B1, B2):
    """A function for validating a CEGIS supermartingale via Gurobi's MILP solver"""
    model = gp.Model("verify_model")

    # State variables
    x = model.addVars(n, lb=min_bound, ub=max_bound, name="X")
    model.setParam(GRB.Param.OutputFlag, 0)

    # Set A constraint (outside the equilibrium set)
    # squared_distance = gp.quicksum((x[i] - equil_set.center[i])**2 for i in range(n))
    # model.addConstr(squared_distance >= equil_set.radius**2 + 1e-6)  # ensure strict inequality

    z = model.addVar(vtype=GRB.BINARY, name="z")  # Binary variable

    # Big M value (should be sufficiently large based on your problem context)
    M = 1e6 * abs(max_bound)

    # Add constraints to ensure the vector (x, y) is not in the negative quadrant
    model.addConstr(x[0] >= -M * z)  # When z = 1, x[0] can be negative (bounded by -M)
    model.addConstr(x[1] >= -M * (1 - z))  # When z = 0, x[1] can be negative (bounded by -M)
    # model.addConstr(x[0] >= 0 - M * (1 - z))  # Ensure x[0] is non-negative when z = 0
    # model.addConstr(x[1] >= 0 - M * (1 - z))  # Ensure x[1] is non-negative when z = 1

    #model.addConstr((x[0] >= 0) | (x[1] >= 0 ))

    # Transition kernel P(x)
    C = C.tolist()
    B = B.tolist()
    r = r.T.tolist()[0]

    Cx = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    phi_x = model.addVars(n, vtype=GRB.BINARY)
    Bphi = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    P_x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Px")

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

        # print(str(V_x.X))
        # print(str([x_tplus1_up_up[i].X for i in range(n)]) + "  ->  " + str(V_Px_up_up.X))
        # print(str([x_tplus1_down_up[i].X for i in range(n)]) + "  ->  " + str(V_Px_down_up.X))
        # print(str([x_tplus1_up_down[i].X for i in range(n)]) + "  ->  " + str(V_Px_up_down.X))
        # print(str([x_tplus1_down_down[i].X for i in range(n)]) + "  ->  " + str(V_Px_down_down.X))
        # print(E_V_X_tplus1.X)
        # print("milp ^^^")
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


def verify_model_sat(n, h, C, B, r, epsilon, W1, W2, B1, B2):
    """A function for validating a CEGIS supermartingale via Z3's SAT solver"""
    solver = Solver()

    # state
    x = [Real(f"X_{i}") for i in range(n)]

    # Add constraints to the solver to bound each value in x
    for xi in x:
        solver.add(xi >= min_bound)
        solver.add(xi <= max_bound)

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
