import numpy as np
import random
from typing import Callable, Tuple, List
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, bounds: List[Tuple[float, float]], parent=None):
        self.bounds = bounds
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.count = 0
        self.reward_sum = 0
        self.diameter = self.compute_diameter()

    def compute_diameter(self) -> float:
        return euclidean(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds]
        )

    def update(self, reward: float):
        self.count += 1
        self.reward_sum += reward

    @property
    def estimated_reward(self):
        return self.reward_sum / max(1, self.count)

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

def compute_confidence_interval(node: TreeNode, t: int, beta: float, alpha: float) -> float:
    return (beta * np.sqrt(4 * np.log(t + 1) / max(1, node.count)) +
            alpha * node.diameter)

def split_node(node: TreeNode) -> None:
    n_dims = len(node.bounds)
    max_dim = max(range(n_dims), key=lambda i: node.bounds[i][1] - node.bounds[i][0])
    mid = (node.bounds[max_dim][0] + node.bounds[max_dim][1]) / 2

    new_bounds1 = node.bounds.copy()
    new_bounds1[max_dim] = (node.bounds[max_dim][0], mid)

    new_bounds2 = node.bounds.copy()
    new_bounds2[max_dim] = (mid, node.bounds[max_dim][1])

    node.left_child = TreeNode(new_bounds1, parent=node)
    node.right_child = TreeNode(new_bounds2, parent=node)

def sample_from_node(node: TreeNode) -> np.ndarray:
    return np.array([np.random.uniform(*b) for b in node.bounds])

def state_to_column(state):
    return np.array([state]).T

def column_to_state(state):
    return state.T[0]

class Graphs:
    def __init__(self):
        self.num_nodes = []
        self.max_ucb = []
        self.min_lcb = []
        self.avg_reward = []
        self.beta_values = []

    def update(self, root: TreeNode, t: int, beta: float):
        nodes = self.get_leaf_nodes(root)
        self.num_nodes.append(len(nodes))
        ucbs = [n.estimated_reward + compute_confidence_interval(n, t, beta, 1.0) for n in nodes]
        lcbs = [n.estimated_reward - compute_confidence_interval(n, t, beta, 1.0) for n in nodes]
        self.max_ucb.append(max(ucbs))
        self.min_lcb.append(min(lcbs))
        self.avg_reward.append(np.mean([n.estimated_reward for n in nodes]))
        self.beta_values.append(beta)

    def get_leaf_nodes(self, node: TreeNode) -> List[TreeNode]:
        if node.is_leaf():
            return [node]
        return self.get_leaf_nodes(node.left_child) + self.get_leaf_nodes(node.right_child)

    def plot_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0, 0].plot(self.num_nodes)
        axs[0, 0].set_title('Number of Leaf Nodes')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Count')

        axs[0, 1].plot(self.max_ucb, label='Max UCB')
        axs[0, 1].plot(self.min_lcb, label='Min LCB')
        axs[0, 1].set_title('UCB and LCB')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].legend()

        axs[1, 0].plot(self.avg_reward)
        axs[1, 0].set_title('Average Estimated Reward')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Value')

        axs[1, 1].plot(self.beta_values)
        axs[1, 1].set_title('Beta Values')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

def mab_algorithm(
        initial_bounds: List[Tuple[float, float]],
        dynamics: Callable[[np.ndarray], np.ndarray],
        certificate: Callable[[np.ndarray], float],
        lipschitz: float,
        beta: float,
        max_iterations: int = 1000
) -> bool:
    root = TreeNode(initial_bounds)
    t = 0

    monitor = Graphs()

    while t < max_iterations:
        t += 1

        # Select node with highest UCB
        leaf_nodes = monitor.get_leaf_nodes(root)
        ucbs = [node.estimated_reward + compute_confidence_interval(node, t, beta, lipschitz)
                for node in leaf_nodes]
        selected_node = leaf_nodes[np.argmax(ucbs)]

        # Sample and compute reward
        x = sample_from_node(selected_node)
        x_next = column_to_state(random.choice(dynamics(state_to_column(x))))

        reward = certificate(x_next) - certificate(x)

        # Update node
        selected_node.update(reward)

        monitor.update(root, t, beta)

        # Check termination conditions
        if max(ucbs) <= 0:
            monitor.plot_metrics()
            return True  # Certificate is valid
        if selected_node.estimated_reward - compute_confidence_interval(selected_node, t, beta, lipschitz) > 0:
            monitor.plot_metrics()
            return False  # Certificate is invalid

        # Split node
        split_node(selected_node)

    monitor.plot_metrics()
    raise ValueError("Maximum iterations reached without conclusion")
