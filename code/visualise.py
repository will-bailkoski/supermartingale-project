import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch


def show_invariant_region(A, b, x_bounds, num_points):

    if len(x_bounds) != 2:
        raise ValueError("Can only display invariant regions for two dimensions.")

    xs = np.random.uniform(x_bounds[0], x_bounds[1], num_points)
    ys = np.random.uniform(x_bounds[0], x_bounds[1], num_points)
    plot_x = []
    plot_y = []

    for x, y in zip(xs, ys):
        if all(A @ np.array([[x],[y]]) <= b):
            plot_x.append(x)
            plot_y.append(y)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the feasible region
    ax.scatter(plot_x, plot_y, marker='.')

    # Labeling and customization
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(x_bounds[0], x_bounds[1])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Feasible Region for $A\mathbf{x} \leq \mathbf{b}$')

    # Show plot
    plt.show()



def plot_weight_distribution(model, epoch):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(model.fc1.weight.detach().numpy().flatten(), bins=50)
    plt.title(f'FC1 Weight Distribution - Epoch {epoch}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(model.fc2.weight.detach().numpy().flatten(), bins=50)
    plt.title(f'FC2 Weight Distribution - Epoch {epoch}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_weight_changes(weight_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    fc1_weights = np.array(weight_history['fc1'])
    for i in range(fc1_weights.shape[1]):
        plt.plot(fc1_weights[:, i], label=f'Neuron {i + 1}')
    plt.title('FC1 Weight Changes Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')


    plt.subplot(1, 2, 2)
    fc2_weights = np.array(weight_history['fc2'])
    for i in range(fc2_weights.shape[1]):
        plt.plot(fc2_weights[:, i], label=f'Neuron {i + 1}')
    plt.title('FC2 Weight Changes Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')


    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history, sat_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    zero_loss_indices = []
    for i, loss in enumerate(loss_history):
        if loss == 0 and (i == 0 or loss_history[i - 1] != 0):
            zero_loss_indices.append(i)
    plt.scatter(zero_loss_indices, [loss_history[i] for i in zero_loss_indices], color='red', marker='x')

    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(sat_history)
    full_sat_indices = []
    for i, sat in enumerate(sat_history):
        if sat == 1 and (i == 0 or sat_history[i - 1] != 1):
            full_sat_indices.append(i)
    plt.scatter(full_sat_indices, [sat_history[i] for i in full_sat_indices], color='red', marker='x')

    plt.title('Satisfaction Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Sat Rate')

    plt.tight_layout()
    plt.show()


def plot_output_distribution(model, X, epoch):
    with torch.no_grad():
        outputs = model(X).numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.hist(outputs, bins=50)
    plt.title(f'Output Distribution - Epoch {epoch}')
    plt.xlabel('Output Value')
    plt.ylabel('Frequency')
    plt.show()


def plot_decision_boundary(model, X, epoch):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy()
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=model(X).detach().numpy().flatten(), cmap=plt.cm.RdYlBu)
    plt.title(f'Decision Boundary - Epoch {epoch}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()