import numpy as np
import matplotlib.pyplot as plt


def is_in_invariant_set(x):
    return np.all(x < 0)


def lipschitz_constant_statistical_test(P, calculated_val, num_samples, bounds):

    max_ratio = 0
    points = []
    gradient_norms = []

    for _ in range(num_samples):
        # Sample two random points within the bounds
        x1 = np.array([[np.random.uniform(low, high) for (low, high) in bounds]]).T
        x2 = np.array([[np.random.uniform(low, high) for (low, high) in bounds]]).T

        # Compute the outputs and the ratio
        f1, f2 = P(x1), P(x2)
        norm_diff_outputs = np.linalg.norm(f1 - f2)
        norm_diff_inputs = np.linalg.norm(x1 - x2)

        # Avoid division by zero
        if norm_diff_inputs > 1e-10:
            ratio = norm_diff_outputs / norm_diff_inputs
            max_ratio = max(max_ratio, ratio)
            # Collect data for plotting
            points.append((x1[0], x1[1]))  # Use the first point for scatter
            gradient_norms.append(ratio)

    # Convert points and gradient_norms to arrays for plotting
    points = np.array(points)
    gradient_norms = np.array(gradient_norms)

    # Plotting
    plt.scatter(points[:, 0], points[:, 1], c=gradient_norms, cmap='viridis')
    plt.colorbar(label='Lipschitz Constant')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Color-mapped Scatter Plot of Lipschitz Constant')
    plt.show()

    assert max_ratio < calculated_val, "incorrect lipschitz constant calculated"
