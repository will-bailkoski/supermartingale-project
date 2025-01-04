import numpy as np
from itertools import product


def phi(v_vbar):
    indicator = []
    for i in range(0, len(v_vbar)):
        if v_vbar[i][0] < 0:
            indicator.append([1])
        else:
            indicator.append([0])
    return indicator


# Stochasticity
def state_noise(state):
    noise_dict = {}
    for k in range(len(state)):
        noise_dict[str(k)] = (state[k] * 0.9, state[k] * 1.1)

    return list(product(*noise_dict.values()))

def transition_kernel(previous_state, C, B, r):
    # returns list of possible states

    n = len(previous_state)
    assert previous_state.shape == (n, 1), \
        f"Input x must have shape ({n}, 1), but has shape {previous_state.shape}"
    assert C.shape == (n, n), \
        f"Input x must have shape ({n}, {n}), but has shape {C.shape}"
    assert B.shape == (n, n), \
        f"Input x must have shape (2, 1), but has shape {B.shape}"
    assert r.shape == (n, 1), \
        f"Input x must have shape ({n}, 1), but has shape {r.shape}"

    Cx = np.dot(C, previous_state)
    Bphi = np.dot(B, phi(previous_state))
    return np.array(state_noise(Cx + r - Bphi))


def negative_quad_invariant_check(C, Cinv, r, beta, V_threshold, D, p):  # checks to see if the entire negative quadrant is invariant

    check_1 = all(np.dot(Cinv, r - beta) < 0)  # Lemma 3.3, ensure an equilibrium point is in the negative quadrant
    check_2 = not all(np.dot(Cinv, r) < 0)  # Lemma 3.4, ensure the entire plane is not invariant
    check_3 = all(np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p) < beta)  # Theorem 3, ensure the negative quadrant is invariant

    return check_1 and check_2 and check_3


def is_in_invariant_set(x):
    return np.all(x < 0)


def generate_random_C(n):
    while True:
        # Initialize matrix with zeros
        C = np.zeros((n, n))

        # Populate off-diagonal elements with random non-negative values
        for i in range(n):
            for j in range(n):
                if i != j:
                    C[i, j] = np.random.random()

        # Scale columns if necessary to ensure column sum < 1
        column_sums = C.sum(axis=0)
        scaling_factors = np.minimum(1, 1 / column_sums)
        C = C * scaling_factors[np.newaxis, :]

        # Check for nonsingularity (determinant must be non-zero)
        cond_threshold = 1e9
        if np.linalg.det(C) != 0:
            I_minus_C = np.eye(n) - C
            if np.linalg.det(I_minus_C) != 0 and np.linalg.cond(I_minus_C) < cond_threshold:
                return C, np.linalg.inv(I_minus_C)


def generate_params(n, m):

    print("Attempting to generate parameters...")
    i = 1
    while True:
        print(f"attempt {i}")
        i += 1
        C, Cinv = generate_random_C(n)

        D = np.random.uniform(0, 0.1, (n, m))
        p = np.array([[10] * m]).T  # paper has the shape incorrect sometimes

        beta = np.array([[0.3] * n]).T  # definable
        B = np.diag(beta.T[0])  # Failure costs

        V_threshold = np.array([[5] * n]).T  # Failure thresholds

        r = np.dot(C - np.eye(len(C)), V_threshold) + np.dot(D, p)

        if negative_quad_invariant_check(C, Cinv, r, beta, V_threshold, D, p):
            return C, D, p, B, V_threshold, r
