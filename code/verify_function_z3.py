from z3 import *
import numpy as np


def z3_value_to_double(val):
    if is_real(val):
        # For RatNumRef (rational numbers)
        return val.numerator().as_long() / val.denominator().as_long()
    elif is_int(val):
        # For IntNumRef (integers)
        return float(val.as_long())
    elif is_true(val):
        return 1.0
    elif is_false(val):
        return 0.0
    elif is_algebraic_value(val):
        # For algebraic numbers
        return val.approx(20)  # 20 digits of precision, adjust if needed
    else:
        # For other types, attempt to get a string representation and convert
        try:
            return float(str(val))
        except ValueError:
            raise ValueError(f"Cannot convert {val} of type {type(val)} to double")


def verify_model(n, h, equil_set, C, B, V_threshold, D, p, r, epsilon, W1, W2, B1, B2):
    solver = Solver()

    # state
    x = [Real(f"X_{i}") for i in range(n)]

    # set A
    center = equil_set.center
    radius = equil_set.radius

    squared_distance = sum((p_ - c[0]) ** 2 for p_, c in zip(x, center))
    solver.add(squared_distance > radius ** 2)

    # model parameters
    C_ = C.tolist() if isinstance(C, np.ndarray) else C
    B_ = B.tolist() if isinstance(B, np.ndarray) else B
    V_threshold_ = V_threshold.tolist() if isinstance(V_threshold, np.ndarray) else V_threshold
    D_ = D.tolist() if isinstance(D, np.ndarray) else D
    p_ = p.tolist() if isinstance(p, np.ndarray) else p
    r_ = r.T[0].tolist()
    #r_ = [Sum([C[i][j] * V_threshold[j][0] for j in range(n)]) - V_threshold[i][0] + Sum(
    #    [D[i][j] * p[j][0] for j in range(len(p))]) for i in range(n)]

    # transition kernel
    def P(x):
        Cx = [Sum([C_[i][j] * x[j] for j in range(n)]) for i in range(n)]
        phi_x = [If(x[i] < 0, 1, 0) for i in range(n)]
        Bphi = [Sum([B_[i][j] * phi_x[j] for j in range(n)]) for i in range(n)]
        return [Cx[i] + r_[i] - Bphi[i] for i in range(n)]

    # neural network weights and biases
    w1 = W1.tolist() if isinstance(W1, np.ndarray) else W1
    w2 = W2.tolist() if isinstance(W2, np.ndarray) else W2
    b1 = B1.tolist() if isinstance(B1, np.ndarray) else B1
    b2 = B2.tolist() if isinstance(B2, np.ndarray) else B2

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
