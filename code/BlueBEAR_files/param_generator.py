import numpy as np
from itertools import product
import time


def generate_stable_nonneg_C(n, max_col_sum=0.99):
    """Generate a stable non-negative matrix C with:
    - zeros on diagonal
    - sub-stochastic (column sums < 1)
    """
    # Start with a non-negative random matrix
    A = np.abs(np.random.rand(n, n))

    # Set diagonal elements to zero
    np.fill_diagonal(A, 0)

    # Scale columns to ensure sub-stochastic property
    col_sums = np.sum(A, axis=0)
    scaling_factors = max_col_sum / (col_sums + 1e-10)

    # Apply scaling column-wise
    C = A * scaling_factors[np.newaxis, :]

    return C


def generate_parameters(n, m):
    """Generate initial parameters with all constraints"""
    # Generate non-negative, sub-stochastic C with zeros on diagonal
    C = generate_stable_nonneg_C(n)

    # Generate non-negative D with normalized columns
    D = np.abs(np.random.rand(n, m))
    D = D / np.max(D)  # Normalize while keeping non-negative

    # Generate non-negative, non-null p
    p = np.random.rand(m) * 0.8 + 0.2  # Values between 0.2 and 1.0
    p = p / np.max(p)  # Normalize while keeping non-null

    # Generate positive beta vector
    b = np.random.rand() + 0.1  # Adding 0.1 to avoid zero
    beta = np.array([b] * n)
    # Generate v_threshold with positive components
    v_threshold = np.abs(np.random.rand(n)) + 0.1

    return C, D, p, beta, v_threshold


def check_conditions(C, D, p, beta, v_threshold):
    """Check all conditions including sub-stochastic constraint"""
    n = C.shape[0]
    I = np.eye(n)

    # Check non-negativity constraints
    if (np.any(C < 0) or np.any(D < 0) or np.any(p < 0) or np.any(beta < 0)):
        return False

    # Check diagonal of C is zero
    if not np.allclose(np.diag(C), 0):
        return False

    # Check p is non-null (no zero elements)
    if np.any(p == 0):
        return False

    # Check C is sub-stochastic (column sums < 1)
    if np.any(np.sum(C, axis=0) >= 1):
        return False

    # Calculate r
    r = (C - I) @ v_threshold + D @ p

    try:
        inv_IC = np.linalg.inv(I - C)

        # Check condition 1: (C − I)v_threshold + Dp < β (element-wise)
        cond1 = np.all(r < beta)

        # Check condition 2: (I − C)^−1(r −β) < 0
        cond2 = np.all(inv_IC @ (r - beta) < 0)

        # Check condition 3: NOT (I − C)^−1 r < 0
        cond3 = not np.all(inv_IC @ r < 0)

        return cond1 and cond2 and cond3
    except np.linalg.LinAlgError:
        return False


def smart_search(n, m, max_attempts=100000000):
    """Search for valid parameters with intelligent constraints"""
    for attempt in range(max_attempts):
        # Generate parameters
        C, D, p, beta, v_threshold = generate_parameters(n, m)

        # Check conditions
        if check_conditions(C, D, p, beta, v_threshold):
            return C, D, p, beta, v_threshold, attempt + 1

    return None, None, None, None, None, max_attempts


def find_example(n, m):
    """Find and return a valid example"""
    start_time = time.time()
    C, D, p, beta, v_threshold, attempts = smart_search(n, m)
    end_time = time.time()

    if C is not None:
        print(f"Found valid parameters for n={n}, m={m} in {attempts} attempts")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return C, D, p, beta, v_threshold
    else:
        print(f"Failed to find valid parameters after {attempts} attempts")
        return None


def verify_solution(C, D, p, beta, v_threshold):
    """Verify and print detailed information about a solution"""
    n = C.shape[0]
    I = np.eye(n)
    r = (C - I) @ v_threshold + D @ p
    inv_IC = np.linalg.inv(I - C)

    print("\nVerification Results:")
    print("1. Matrix C properties:")
    print(f"   C >= 0: {np.all(C >= 0)}")
    print(f"   Diagonal elements = 0: {np.allclose(np.diag(C), 0)}")
    print(f"   Sub-stochastic (column sums < 1): {np.all(np.sum(C, axis=0) < 1)}")
    print(f"   Maximum column sum: {np.max(np.sum(C, axis=0)):.4f}")

    print("\n2. Other non-negativity constraints:")
    print(f"   D >= 0: {np.all(D >= 0)}")
    print(f"   p >= 0: {np.all(p >= 0)}")
    print(f"   p non-null: {np.all(p > 0)}")
    print(f"   beta >= 0: {np.all(beta >= 0)}")

    print("\n3. System conditions:")
    print(f"   Condition 1 ((C-I)v + Dp < β): {np.all(r < beta)}")
    print(f"   Condition 2 ((I-C)^(-1)(r-β) < 0): {np.all(inv_IC @ (r - beta) < 0)}")
    print(f"   Condition 3 (NOT (I-C)^(-1)r < 0): {not np.all(inv_IC @ r < 0)}")


def print_parameters(C, D, p, beta, v_threshold):
    """Print all parameters and r in a readable format"""
    n, m = C.shape[0], D.shape[1]
    I = np.eye(n)
    r = (C - I) @ v_threshold + D @ p

    print("\nParameter Values:")
    print("\nC matrix:")
    print(np.array2string(C, precision=4, suppress_small=True))

    print("\nD matrix:")
    print(np.array2string(D, precision=4, suppress_small=True))

    print("\np vector:")
    print(np.array2string(p, precision=4, suppress_small=True))

    print("\nbeta vector:")
    print(np.array2string(beta, precision=4, suppress_small=True))

    print("\nv_threshold vector:")
    print(np.array2string(v_threshold, precision=4, suppress_small=True))

    print("\nr vector ((C-I)v_threshold + Dp):")
    print(np.array2string(r, precision=4, suppress_small=True))


def find_and_save_example(n, m, directory="param_examples"):
    """Convenience function to find, print, and save an example with a categorized filename"""
    result = find_example(n, m)

    if result is not None:
        C, D, p, beta, v_threshold = result
        verify_solution(C, D, p, beta, v_threshold)
        print_parameters(C, D, p, beta, v_threshold)
        p = np.array([p]).T
        v_threshold = np.array([v_threshold]).T
        B = np.diag(beta)
        r = np.dot(C - np.eye(len(C)), v_threshold) + np.dot(D, p)

        # Generate the filename based on n and m
        filename = f"{directory}/n{n}_m{m}.npz"

        # Save the numpy arrays to the file
        np.savez(filename, C=C, D=D, p=p, beta=beta, v_threshold=v_threshold, B=B, r=r)
        print(f"Arrays saved to {filename}")

        return C, D, p, B, v_threshold, r
    return None


def phi(v_vbar):
    indicator = []
    for i in range(0, len(v_vbar)):
        if v_vbar[i][0] < 0:
            indicator.append([1])
        else:
            indicator.append([0])
    return indicator

# Stochasticity
def state_noise(state, kappa):
    noise_dict = {}
    for k in range(len(state)):
        noise_dict[str(k)] = (state[k] * -kappa, state[k] * kappa)

    return list(product(*noise_dict.values()))

def transition_kernel(previous_state, C, B, r, kappa):
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

    deterministic = np.dot(C, previous_state) + r - np.dot(B, phi(previous_state))
    noise = state_noise(previous_state, kappa)
    successors = np.array([i + deterministic for i in noise])
    return successors
