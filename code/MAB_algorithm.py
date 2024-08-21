import numpy as np
import random
from typing import Callable, Tuple, List
from scipy.spatial.distance import euclidean
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from collections import deque




sample_history = []


class TreeNode:
    def __init__(self, bounds: List[Tuple[float, float]], parent=None):
        self.count = 0
        self.reward_sum = 0
        self.bounds = bounds
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.diameter = self.compute_diameter()

        # Iterate over the sample history and check if samples are within bounds
        for (sample, reward) in sample_history:
            if all(lower <= x <= upper for (x, (lower, upper)) in zip(sample, bounds)):
                self.reward_sum += reward
                self.count += 1

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
        if self.count > 0:
            return self.reward_sum / self.count
        else:
            return 1

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

def compute_confidence_interval(node: TreeNode, t: int, beta: float, alpha: float) -> float:
    return (beta * np.sqrt(4 * np.log(t + 1) / max(1, node.count)) +
            alpha * node.diameter)

def split_node(node: TreeNode, n_dims: int) -> tuple[TreeNode, TreeNode]:

    max_dim = max(range(n_dims), key=lambda i: node.bounds[i][1] - node.bounds[i][0])
    mid = (node.bounds[max_dim][0] + node.bounds[max_dim][1]) / 2

    new_bounds1 = node.bounds.copy()
    new_bounds1[max_dim] = (node.bounds[max_dim][0], mid)

    new_bounds2 = node.bounds.copy()
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

    def update(self, root: TreeNode, current: TreeNode, t: int, beta: float, l: float):
        self.num_nodes.append(self.num_nodes[-1])
        ucb = current.estimated_reward + compute_confidence_interval(current, t, beta, l)
        lcb = current.estimated_reward - compute_confidence_interval(current, t, beta, l)
        self.confidence.append(beta * np.sqrt(4 * np.log(t + 1) / max(1, current.count)))
        self.count_no.append(current.count)
        self.diameter.append(l * current.diameter)
        self.max_ucb.append(ucb)
        self.min_lcb.append(lcb)
        self.avg_reward.append(current.estimated_reward)

        self.get_leaf_nodes(root)




        # heatmap data
        for node in self.get_leaf_nodes(root):
            center = [(b[0] + b[1]) / 2 for b in node.bounds]
            self.heatmap_data.append((center, node.count))

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

    def plot_metrics(self, resolution):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0, 0].scatter([x[0][0] for x in sample_history], [x[0][1] for x in sample_history], marker='.')
        axs[0, 0].set_title('scatterplot')
        # axs[0, 0].set_xlabel('Iteration')
        # axs[0, 0].set_ylabel('Count')

        axs[0, 1].plot(self.max_ucb, label='Max UCB')
        axs[0, 1].plot(self.min_lcb, label='Min LCB')
        axs[0, 1].plot([0] * len(self.max_ucb), dashes=[4, 4])
        #axs[0, 1].plot(self.diameter, label='diameter')
        #axs[0, 1].plot(self.confidence, label='confidence')
        # axs[0, 1].plot(self.count_no, label='count')
        axs[0, 1].set_title('UCB and LCB')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].legend()

        axs[1, 0].plot(self.avg_reward)
        axs[1, 0].set_title('Average Estimated Reward')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Value')

        # Extract data for heatmap

        # #print(self.heatmap_data)
        # x_coords = [p[0][0] for p in self.heatmap_data]
        # y_coords = [p[0][1] for p in self.heatmap_data]
        # rewards = [p[1] for p in self.heatmap_data]
        #
        # # Create grid
        # xi = np.linspace(min(x_coords), max(x_coords), resolution)
        # yi = np.linspace(min(y_coords), max(y_coords), resolution)
        # xi, yi = np.meshgrid(xi, yi)
        #
        # # Interpolate the reward data onto the grid
        # zi = griddata((x_coords, y_coords), rewards, (xi, yi), method='linear')
        #
        #
        # heatmap = axs[1, 1].contourf(xi, yi, zi, levels=20, cmap="viridis")
        # #axs[1, 1].set_colorbar(heatmap, label='Estimated Reward')
        # axs[1, 1].set_title('Heatmap of the Explored Domain')
        # axs[1, 1].set_xlabel('X coordinate')
        # axs[1, 1].set_ylabel('Y coordinate')


        plt.tight_layout()
        plt.show()

def mab_algorithm(
        initial_bounds: List[Tuple[float, float]],
        dynamics: Callable[[np.ndarray], np.ndarray],
        certificate: Callable[[np.ndarray], float],
        lipschitz: float,
        beta: float,
        max_iterations: int
) -> bool:
    root = TreeNode(initial_bounds)
    n_dims = len(initial_bounds)
    t = 0
    too_deep = False

    split_node(root, n_dims)

    monitor = Graphs()

    while t < max_iterations:
        t += 1

        leaf_nodes = monitor.get_leaf_nodes(root)

        if t % 1000 == 0:
            print("iteration: ", t)

        # Select node with highest UCB
        ucbs = [node.estimated_reward + compute_confidence_interval(node, t, beta, lipschitz)
                for node in leaf_nodes]
        selected_node = leaf_nodes[np.argmax(ucbs)]

        # Sample and compute reward
        sample = np.array([np.random.uniform(*b) for b in selected_node.bounds])

        x = state_to_column(sample)
        x_next = random.choice(dynamics(x))

        reward = certificate(x_next) - certificate(x)
        selected_node.update(reward)

        sample_history.append((sample, reward))

        monitor.update(root, selected_node, t, beta, lipschitz)

        # termination conditions
        if max(ucbs) <= 0:
            monitor.plot_metrics(50)
            return True  # Certificate is valid
        if selected_node.estimated_reward - compute_confidence_interval(selected_node, t, beta, lipschitz) > 0:
            monitor.plot_metrics(50)
            return False  # Certificate is invalid

        # Split node
        if not too_deep:
            left, right = split_node(selected_node, n_dims)
            for child in [left, right]:
                sample = np.array([np.random.uniform(*b) for b in child.bounds])

                x = state_to_column(sample)
                x_next = random.choice(dynamics(x))

                reward = certificate(x_next) - certificate(x)
                selected_node.update(reward)

                sample_history.append((sample, reward))

                monitor.update(root, selected_node, t, beta, lipschitz)

            if monitor.depth(root) > 50:
                too_deep = True

    print(sample_history)
    monitor.plot_metrics(50)
    compute_confidence_interval(selected_node, t, beta, lipschitz)
    raise ValueError("Maximum iterations reached without conclusion")
