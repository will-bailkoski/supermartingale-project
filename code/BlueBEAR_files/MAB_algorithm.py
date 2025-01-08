import numpy as np
import random
from typing import Callable, Tuple, List, Any

from numpy import ndarray
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from collections import deque
from sortedcontainers import SortedList
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

sample_history = []
ucbs = []
lcbs = []
rewards = []


class TreeNode:
    def __init__(self, bounds: List[Tuple[float, float]], kernel: Any, parent=None, integer_domain=False):
        self.count = 0
        self.reward_sum = 0
        self.bounds = bounds
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.integer_domain = integer_domain
        self.diameter = self.compute_diameter()
        self.varience_proxy_total = 0
        if parent == None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        for (sample, reward) in sample_history:
            if all(lower <= x <= upper for (x, (lower, upper)) in zip(sample, bounds)):
                self.reward_sum += reward
                #self.varience_proxy_total +=
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

    def update(self, reward: float):
        self.count += 1
        self.reward_sum += reward

    @property
    def estimated_reward(self):
        if self.count > 0:
            return self.reward_sum / self.count
        else:
            return 0  # TODO

    def is_leaf(self):
        return self.left_child is None and self.right_child is None


def compute_confidence_interval(node: TreeNode, beta: float, alpha: float, confidence: float) -> float:
    return compute_statistical_error(node, beta, confidence) + compute_discretisation_error(node, alpha)


def compute_statistical_error_old(node: TreeNode, beta: float, confidence: float) -> float:
    return np.sqrt(3.3 * beta * (2 * np.log(np.log(max(1, node.count))) + np.log(2/confidence)) / max(1, node.count))


def compute_statistical_error(node: TreeNode, alpha: float, beta: float, confidence: float) -> float:
    return np.sqrt(13.2 * len(node.bounds) * (alpha**2) * beta * (2 * np.log(np.log(max(1, node.count))) + np.log(2/confidence)) / max(1, node.count**2))


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

    def update(self, root: TreeNode, current: TreeNode, t: int, beta: float, l: float):

        # bottom right graph
        ucb = current.estimated_reward + compute_confidence_interval(current, t, beta, l)
        lcb = current.estimated_reward - compute_confidence_interval(current, t, beta, l)
        self.max_ucb.append(ucb)
        self.min_lcb.append(lcb)

        # bottom left graph
        self.avg_reward.append(current.estimated_reward)

        # self.num_nodes.append(self.num_nodes[-1])

        # self.confidence.append(beta * np.sqrt(4 * np.log(t + 1) / max(1, current.count)))
        # self.count_no.append(current.count)
        # self.diameter.append(l * current.diameter)

        # self.get_leaf_nodes(root)
        #
        # # heatmap data
        # for node in self.get_leaf_nodes(root):
        #     center = [(b[0] + b[1]) / 2 for b in node.bounds]
        #     self.heatmap_data.append((center, node.count))

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

    def plot_metrics(self, title, root, epsilon, beta, lipschitz, bounds):

        print("Plotting graph...")

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        fig.suptitle(title, fontsize=20)

        # scatter = axs[0, 0].scatter([x[0][0] for x in sample_history], [x[0][1] for x in sample_history], marker='.',
        #                             c=[x[1] for x in sample_history], cmap='viridis')
        # axs[0, 0].set_title('Scatterplot of samples over state space')
        # fig.colorbar(scatter, ax=axs[0, 0], label='MAB reward')
        #
        # leaf_nodes = self.get_leaf_nodes(root)
        # rectangles = [(x.bounds, (
        #     x.count, x.estimated_reward + compute_confidence_interval(x, len(sample_history), beta, lipschitz),
        #     lipschitz * x.diameter, (beta * np.sqrt(4 * np.log(len(sample_history) + 1) / max(1, x.count))),
        #     x.estimated_reward)) for x in leaf_nodes]

        # states = np.array([x[0] for x in sample_history])  # High-dimensional states
        # rewards = [x[1] for x in sample_history]  # Rewards
        #
        # # Apply PCA to reduce the state space to 2D
        # pca = PCA(n_components=2)
        # reduced_states = pca.fit_transform(states)
        #
        # # Create the plot
        # fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        # fig.suptitle(title, fontsize=20)
        #
        # # Scatter plot for PCA-reduced states
        # scatter = axs[0, 0].scatter(
        #     reduced_states[:, 0], reduced_states[:, 1],
        #     marker='.', c=rewards, cmap='viridis'
        # )
        # axs[0, 0].set_title('Scatterplot of samples over state space (PCA Reduced)')
        # axs[0, 0].set_xlabel('Principal Component 1')
        # axs[0, 0].set_ylabel('Principal Component 2')
        # fig.colorbar(scatter, ax=axs[0, 0], label='MAB reward')

        # # Visualization of leaf nodes (as in your original code)
        # leaf_nodes = self.get_leaf_nodes(root)
        # rectangles = [(x.bounds, (
        #     x.count, x.estimated_reward + compute_confidence_interval(x, len(sample_history), beta, lipschitz),
        #     lipschitz * x.diameter, (beta * np.sqrt(4 * np.log(len(sample_history) + 1) / max(1, x.count))),
        #     x.estimated_reward)) for x in leaf_nodes]
        #
        # # Loop through the list of rectangles and plot each one
        # i = 0
        # for rect in rectangles:
        #     i += 1
        #     ((x_min, x_max), (y_min, y_max)), labels = rect
        #     width = x_max - x_min
        #     height = y_max - y_min
        #     axs[0, 1].add_patch(
        #         patches.Rectangle(
        #             (x_min, y_min),  # (x, y) position of the bottom left corner
        #             width,  # width of the rectangle
        #             height,  # height of the rectangle
        #             edgecolor='blue',  # outline color
        #             facecolor='none',  # no fill color
        #             linewidth=.5
        #         )
        #     )
        # # Set the limits of the plot to cover all rectangles
        # axs[0, 1].set_xlim(bounds[0], bounds[1])
        # axs[0, 1].set_ylim(bounds[0], bounds[1])
        # # Set aspect of the plot to be equal to maintain the aspect ratio of rectangles
        # axs[0, 1].set_aspect('equal', adjustable='box')
        # axs[0, 1].set_title('Griding of state space')

        # rectangle_bounds = [(x.bounds, x) for x in leaf_nodes]  # Store bounds and node info
        #
        # # Transform bounds into the PCA-reduced space
        # projected_rectangles = []
        # for ((x_min, x_max), (y_min, y_max)), node in rectangle_bounds:
        #     corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        #     projected_corners = pca.transform(corners)
        #     x_coords, y_coords = zip(*projected_corners)
        #     projected_rectangles.append(((min(x_coords), max(x_coords)), (min(y_coords), max(y_coords)), node))
        #
        # # Plot projected rectangles
        # for ((x_min, x_max), (y_min, y_max), node) in projected_rectangles:
        #     axs[0, 1].add_patch(
        #         patches.Rectangle(
        #             (x_min, y_min),
        #             x_max - x_min,
        #             y_max - y_min,
        #             edgecolor='blue',
        #             facecolor='none',
        #             linewidth=0.5
        #         )
        #     )

        axs[1, 0].plot(rewards)
        axs[1, 0].set_title('Estimated Reward of Current Leaf')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Reward')

        axs[1, 1].plot(ucbs, label='Max UCB')
        axs[1, 1].plot(lcbs, label='Corresponding LCB')
        axs[1, 1].plot([epsilon] * len(sample_history), dashes=[4, 4])
        axs[1, 1].plot([-epsilon] * len(sample_history), dashes=[4, 4])
        axs[1, 1].set_title('UCB and LCB')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Reward')
        axs[1, 1].legend()
        # print(self.diameter[45])

        plt.tight_layout()
        plt.show()


def mab_algorithm(
        initial_bounds: List[Tuple[float, float]],
        dynamics: Callable[[np.ndarray], np.ndarray],
        certificate: Callable[[np.ndarray], float],
        lipschitz: float,
        reward_range: float,
        max_iterations: int,
        tolerance: float,
        confidence: float,
) -> tuple[bool, None, int, Any, int, int] | tuple[bool, ndarray, int, Any, int, int]:
    root = TreeNode(initial_bounds)
    n_dims = len(initial_bounds)
    t = 0
    run_times = []
    monitor = Graphs()

    # Initial grid
    nodes = SortedList([(root, root.estimated_reward + compute_confidence_interval(root, reward_range, lipschitz, confidence))],
                       key=lambda x: x[1])

    # Initial griding to avoid edge cases
    nodes.pop(-1)
    for leaf in split_node(root, n_dims):
        sample = np.array([np.random.uniform(*b) for b in leaf.bounds])

        x = state_to_column(sample)
        x_next = random.choice(dynamics(x))

        reward = certificate(x_next) - certificate(x) - tolerance

        leaf.update(reward)
        sample_history.append((sample, reward))
        rewards.append(reward)
        t += 1

        confidence_interval = compute_confidence_interval(leaf, reward_range, lipschitz, confidence)
        ucb = leaf.estimated_reward + confidence_interval
        ucbs.append(ucb)

        nodes.add((leaf, ucb))

    while t < max_iterations:

        if t % 1000 == 0:
            print("iteration: ", t)
        # if t % 1000000 == 1:
        #     monitor.plot_metrics(50, root, epsilon, beta, lipschitz, initial_bounds[0])
        if t % 100000 == 1:
            print(f"Max UCB: {ucb}      Corresponding LCB: {lcb}")

        start_time_verify = time.process_time()
        # Select node with highest UCB
        selected_node, ucb = nodes.pop(-1)
        ucbs.append(ucb)

        # Sample and compute reward

        sample = np.array([np.random.uniform(*b) for b in selected_node.bounds])

        x = state_to_column(sample)
        x_next = random.choice(dynamics(x))

        reward = certificate(x_next) - certificate(x)

        selected_node.update(reward)
        sample_history.append((sample, reward))
        rewards.append(reward)
        t += 1

        confidence_interval = compute_confidence_interval(selected_node, reward_range, lipschitz, confidence)

        lcb = selected_node.estimated_reward - confidence_interval
        lcbs.append(lcb)

        # monitor.update(root, selected_node, len(sample_history), beta, lipschitz)

        # termination conditions
        if ucb <= tolerance:
            result = "Certificate is VALID"
            #monitor.plot_metrics(result, root, tolerance, reward_range, lipschitz, initial_bounds[0])
            print(f"{result}, iterations: {t}, tree depth: {monitor.depth(root)}")
            sample_history.clear()
            rewards.clear()
            ucbs.clear()
            lcbs.clear()
            return True, None, t, np.average(run_times), monitor.depth(root), len(nodes)  # Certificate is valid
        if lcb > -tolerance:
            result = "Certificate is INVALID"
            #monitor.plot_metrics(result, root, tolerance, reward_range, lipschitz, initial_bounds[0])
            print(f"{result}, iterations: {t}, tree depth: {monitor.depth(root)}")
            sample_history.clear()
            rewards.clear()
            ucbs.clear()
            lcbs.clear()
            return False, np.array([np.random.uniform(*b) for b in selected_node.bounds]), t, np.average(run_times), monitor.depth(root), len(nodes)  # Certificate is invalid

        # Split node
        if selected_node.depth < 10000 and compute_statistical_error(selected_node,
                                                                    reward_range, confidence) < compute_discretisation_error(selected_node,
                                                                                                                 lipschitz):
            left, right = split_node(selected_node, n_dims)
            nodes.add((left, left.estimated_reward + compute_confidence_interval(left, reward_range, lipschitz, confidence)))
            nodes.add((right, right.estimated_reward + compute_confidence_interval(right, reward_range, lipschitz, confidence)))
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
            nodes.add((selected_node,
                       selected_node.estimated_reward + compute_confidence_interval(selected_node, reward_range, lipschitz, confidence)))
        run_times.append(start_time_verify - time.process_time())

    # Display the plot
    plt.show()
    monitor.plot_metrics("Did not terminate", root, tolerance, reward_range, lipschitz, initial_bounds[0])

    print(ucbs[-1], lcbs[-1])
    print(max(rewards))
    sample_history.clear()
    rewards.clear()
    ucbs.clear()
    lcbs.clear()
    raise ValueError("Maximum iterations reached without conclusion")
