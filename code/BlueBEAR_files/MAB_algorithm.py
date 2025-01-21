import numpy as np
import random
from typing import Callable, Tuple, List, Any

from numpy import ndarray
from scipy.spatial.distance import euclidean
from collections import deque
from sortedcontainers import SortedList
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
import time

sample_history = []
ucbs = []
lcbs = []
rewards = []


class TreeNode:
    def __init__(self, bounds: List[Tuple[float, float]], parent=None, integer_domain=False):
        self.count = 0
        self.reward_sum = 0
        self.bounds = bounds
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.integer_domain = integer_domain
        self.diameter = self.compute_diameter()
        self.total_variance = 0
        if parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        for (sample, reward) in sample_history:
            if all(lower <= x <= upper for (x, (lower, upper)) in zip(sample, bounds)):
                self.reward_sum += reward
                self.total_variance += np.linalg.norm(sample, ord=2) ** 2
                self.count += 1


    def compute_diameter(self) -> float:
        if self.integer_domain:
            return euclidean(
                [int(b[0]) for b in self.bounds],
                [int(b[1]) for b in self.bounds]
            )
        else:
            return euclidean(
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds]
            )

    def update(self, reward: float, sample: Any):
        self.count += 1
        self.reward_sum += reward
        self.total_variance += np.linalg.norm(sample)

    @property
    def estimated_reward(self):
        if self.count > 0:
            return self.reward_sum / self.count
        else:
            return 0  # TODO

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


def compute_confidence_interval(node: TreeNode, kappa: float, alpha_rho: float, alpha_f: float, confidence: float) -> float:
    return compute_statistical_error(node, alpha_f, kappa, confidence) + compute_discretisation_error(node, alpha_rho)


def compute_statistical_error_old(node: TreeNode, beta: float, confidence: float) -> float:
    return np.sqrt(3.3 * beta * (2 * np.log(np.log(max(1, node.count))) + np.log(2/confidence)) / max(1, node.count))


def compute_statistical_error(node: TreeNode, alpha: float, kappa: float, confidence: float) -> float:
    V = (2 * alpha * kappa) ** 2 * node.total_variance
    if node.total_variance == 0:
        return float('inf')
    else:
        return np.sqrt(3.3 * max(1.0, V) * (2 * np.log(np.log(max(np.e, V))) + np.log(2/confidence)) / max(1, node.count) ** 2)

def compute_discretisation_error(node: TreeNode, alpha: float) -> float:
    return alpha * node.diameter


def split_node(node: TreeNode, n_dims: int) -> tuple[TreeNode, TreeNode]:
    max_dim = max(range(n_dims), key=lambda i: node.bounds[i][1] - node.bounds[i][0])

    mid = (node.bounds[max_dim][0] + node.bounds[max_dim][1]) / 2

    new_bounds1 = node.bounds.copy()
    new_bounds2 = node.bounds.copy()

    new_bounds1[max_dim] = (node.bounds[max_dim][0], mid)
    new_bounds2[max_dim] = (mid, node.bounds[max_dim][1])

    node.left_child = TreeNode(new_bounds1, parent=node)
    node.right_child = TreeNode(new_bounds2, parent=node)

    return node.left_child, node.right_child


def state_to_column(state):
    return np.array([state]).T


def column_to_state(state):
    return state.T[0]


class Graphs:
    def __init__(self):
        self.num_nodes = [0]
        self.max_ucb = []
        self.min_lcb = []
        self.confidence = []
        self.diameter = []
        self.avg_reward = []
        self.heatmap_data = []
        self.count_no = []

    def update(self, current: TreeNode, t: int, beta: float, alpha_rho: float, alpha_f: float, confidence: float):
        ucb = current.estimated_reward + compute_confidence_interval(current, beta, alpha_rho, alpha_f, confidence)
        lcb = current.estimated_reward - compute_confidence_interval(current, beta, alpha_rho, alpha_f, confidence)
        self.max_ucb.append(ucb)
        self.min_lcb.append(lcb)

    def get_leaf_nodes(self, root: TreeNode) -> List[TreeNode]:
        leaf_nodes = []
        stack = deque([root])
        while stack:
            node = stack.pop()
            if node.is_leaf():
                leaf_nodes.append(node)
            else:
                if node.right_child:
                    stack.append(node.right_child)
                if node.left_child:
                    stack.append(node.left_child)
        return leaf_nodes

    def depth(self, root: TreeNode) -> int:
        if root is None:
            return 0

        # Use a stack to perform DFS
        stack = deque([(root, 1)])  # The stack stores tuples of (node, depth)
        max_depth = 0

        while stack:
            node, current_depth = stack.pop()
            max_depth = max(max_depth, current_depth)

            if node.left_child:
                stack.append((node.left_child, current_depth + 1))
            if node.right_child:
                stack.append((node.right_child, current_depth + 1))

        return max_depth

    def plot_metrics(self, root: TreeNode, nodes: SortedList[TreeNode]):
        print("Plotting graph...")
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        bounds = root.bounds
        if len(bounds) != 2:
            return

        # Scatter plot of samples
        scatter = axs[0, 0].scatter(
            [x[0][0] for x in sample_history],
            [x[0][1] for x in sample_history],
            c=[x[1] for x in sample_history], cmap='viridis', marker='.'
        )
        axs[0, 0].set_title('Scatterplot of Samples')
        fig.colorbar(scatter, ax=axs[0, 0], label='Reward')
        axs[0, 0].set_xlim(root.bounds[0][0], root.bounds[0][1])
        axs[0, 0].set_ylim(root.bounds[1][0], bounds[1][1])

        leaf_nodes = [(node, tuple(node.bounds), ucb) for node, ucb in nodes if node.is_leaf()]

        # Loop through the list of rectangles and plot each one
        for rect in leaf_nodes:
            # Unpack bounds and labels
            node, ((x_min, x_max), (y_min, y_max)), val = rect
            width = x_max - x_min
            height = y_max - y_min

            lcb = 2 * node.estimated_reward - val

            if val <= 0:  # Verified (green)
                color = "yellow"
            elif lcb > 0:  # Invalid (red)
                color = "red"
            else:  # Unverified (yellow)
                color = "yellow"

            axs[0, 1].add_patch(
                patches.Rectangle(
                    (x_min, y_min),  # (x, y) position of the bottom left corner
                    width,  # width of the rectangle
                    height,  # height of the rectangle
                    edgecolor='black',  # outline color
                    facecolor=color,   # fill color
                    linewidth=.1
                )
            )

        # Set the limits of the plot to cover all rectangles
        axs[0, 1].set_xlim(root.bounds[0][0], root.bounds[0][1])
        axs[0, 1].set_ylim(root.bounds[1][0], root.bounds[1][1])

        # Set aspect of the plot to be equal to maintain the aspect ratio of rectangles
        axs[0, 1].set_aspect('equal', adjustable='box')
        axs[0, 1].set_title('Gridding of state space')

        # UCB and LCB plot
        axs[1, 0].plot(ucbs, label='Max UCB')
        axs[1, 0].plot(lcbs, label='Min LCB')
        axs[1, 0].axhline(y=0, color='r', linestyle='--', label='Threshold')
        axs[1, 0].set_title('UCB and LCB Over Iterations')
        axs[1, 0].legend()

        plt.tight_layout()
        plt.show()


def mab_algorithm(
        initial_bounds: List[Tuple[float, float]],
        dynamics: Callable[[np.ndarray], np.ndarray],
        certificate: Callable[[np.ndarray], float],
        lipschitz_values: Tuple[float, float],
        tolerance: float,
        confidence: float,
        kappa: float,
        max_iterations=10000,
) -> tuple[(bool | None), None, int, Any, int, int] | tuple[bool, ndarray, int, Any, int, int]:

    root = TreeNode(initial_bounds)
    n_dims = len(initial_bounds)
    t = 0
    run_times = []
    monitor = Graphs()
    lipschitz_rho, lipschitz_f = lipschitz_values

    # Initial grid
    nodes = SortedList([(root, root.estimated_reward + compute_confidence_interval(root, kappa, lipschitz_rho, lipschitz_f, confidence))],
                       key=lambda x: x[1])

    # Initial griding to avoid edge cases
    nodes.pop(-1)
    for leaf in split_node(root, n_dims):
        sample = np.array([np.random.uniform(*b) for b in leaf.bounds])

        x = state_to_column(sample)
        x_next = random.choice(dynamics(x))

        reward = certificate(x_next) - certificate(x) - tolerance

        leaf.update(reward, sample)
        sample_history.append((sample, reward))
        rewards.append(reward)
        t += 1

        confidence_interval = compute_confidence_interval(leaf, kappa, lipschitz_rho, lipschitz_f, confidence)
        ucb = leaf.estimated_reward + confidence_interval
        lcb = leaf.estimated_reward - confidence_interval
        ucbs.append(ucb)
        lcbs.append(lcb)

        nodes.add((leaf, ucb))

    while t < max_iterations:

        if t % 1000 == 0:
            print("iteration: ", t)
        if t % 1000 == 1:
            print(confidence_interval, reward)

        # if t % 1000000 == 1:
        #     monitor.plot_metrics(50, root, epsilon, beta, lipschitz, initial_bounds[0])
        if t % 100000 == 1:
            print(f"Max UCB: {ucb}      Corresponding LCB: {lcb}")
            monitor.plot_metrics(root, nodes)

        start_time_verify = time.process_time()
        # Select node with highest UCB
        selected_node, _ = nodes.pop(-1)

        # Sample and compute reward
        for _ in range(0, 1):
            sample = np.array([np.random.uniform(*b) for b in selected_node.bounds])

            x = state_to_column(sample)
            x_next = random.choice(dynamics(x))

            reward = certificate(x_next) - certificate(x) + tolerance

            selected_node.update(reward, sample)
            sample_history.append((sample, reward))
            rewards.append(reward)
        t += 1

        confidence_interval = compute_confidence_interval(selected_node, kappa, lipschitz_rho, lipschitz_f, confidence)
        ucb = selected_node.estimated_reward + confidence_interval
        lcb = selected_node.estimated_reward - confidence_interval
        ucbs.append(ucb)
        lcbs.append(lcb)

        # monitor.update(root, selected_node, len(sample_history), beta, lipschitz)

        # termination conditions
        if ucb <= 0:
            result = "Certificate is VALID"
            nodes.add((selected_node, ucb))
            monitor.plot_metrics(root, nodes)
            print(f"{result}, iterations: {t}, tree depth: {monitor.depth(root)}")
            sample_history.clear()
            rewards.clear()
            ucbs.clear()
            lcbs.clear()
            return True, None, t, np.average(run_times), monitor.depth(root), len(nodes)  # Certificate is valid
        if lcb > 0:
            result = "Certificate is INVALID"
            nodes.add((selected_node, ucb))
            monitor.plot_metrics(root, nodes)
            print(f"{result}, iterations: {t}, tree depth: {monitor.depth(root)}")
            sample_history.clear()
            rewards.clear()
            ucbs.clear()
            lcbs.clear()
            return False, np.array([np.random.uniform(*b) for b in selected_node.bounds]), t, np.average(run_times), monitor.depth(root), len(nodes)  # Certificate is invalid

        # Split node
        if selected_node.depth < 10000 and compute_statistical_error(selected_node, lipschitz_f, kappa, confidence) < compute_discretisation_error(selected_node, lipschitz_rho):
            left, right = split_node(selected_node, n_dims)
            nodes.add((left, left.estimated_reward + compute_confidence_interval(left, kappa, lipschitz_rho, lipschitz_f, confidence)))
            nodes.add((right, right.estimated_reward + compute_confidence_interval(right, kappa, lipschitz_rho, lipschitz_f,confidence)))
            # for child in [left, right]:
            #     if integer_domain:
            #         sample = np.array([np.random.randint(int(b[0]), int(b[1]) + 1) for b in child.bounds])
            #     else:
            #         sample = np.array([np.random.uniform(*b) for b in child.bounds])
            #
            #     x = state_to_column(sample)
            #     x_next = random.choice(dynamics(x))
            #
            #     reward = certificate(x_next) - certificate(x) - 10
            #     #reward = x[1][0] ** 2 + x[0][0] ** 2 + random.random() - 0.5
            #     selected_node.update(reward)
            #
            #     sample_history.append((sample, reward))

            # monitor.update(root, child, t, beta, lipschitz)
        else:
            nodes.add((selected_node, ucb))
            run_times.append(time.process_time() - start_time_verify)

    # Display the plot
    plt.show()
    monitor.plot_metrics(root, nodes)

    print(ucbs[-1], lcbs[-1])
    print(max(rewards))
    sample_history.clear()
    rewards.clear()
    ucbs.clear()
    lcbs.clear()
    return None, None, t, np.average(run_times), monitor.depth(root), len(nodes)

    raise ValueError("Maximum iterations reached without conclusion")
