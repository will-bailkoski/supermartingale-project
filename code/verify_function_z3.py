from z3 import *
from model_params import min_bound, max_bound


def z3_value_to_double(val):
    if is_real(val):
        return val.numerator().as_long() / val.denominator().as_long()
    else:
        try:
            return float(str(val))
        except ValueError:
            raise ValueError(f"Cannot convert {val} of type {type(val)} to double")


def verify_model(n, h, equil_set, C, B, r, epsilon, W1, W2, B1, B2):
    solver = Solver()

    # state
    x = [Real(f"X_{i}") for i in range(n)]

    # Add constraints to the solver to bound each value in x
    for xi in x:
        solver.add(xi >= min_bound)
        solver.add(xi <= max_bound)

    # set A
    squared_distance = sum((p_ - c[0]) ** 2 for p_, c in zip(x, equil_set.center))
    solver.add(squared_distance > equil_set.radius ** 2)

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
    E_V_X_tplus1 = 0.5 * V(x_tplus1_up) + 0.5 * V(x_tplus1_down)

    solver.add(E_V_X_tplus1 > V_x - epsilon)

    # Check satisfiability
    result = solver.check()
    if result == sat:
        model = solver.model()
        counterexample_dict = {str(d): model[d] for d in model.decls() if d.name().startswith('X_')}
        counterexample = [z3_value_to_double(counterexample_dict[key]) for key in sorted(counterexample_dict.keys())]
        return True, counterexample
    else:
        return False, None
