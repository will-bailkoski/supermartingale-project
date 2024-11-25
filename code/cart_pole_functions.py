import random

#import gymnasium as gym
#from stable_baselines3 import DQN
from functools import partial
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

# # Function to step the environment forward given a state and an action
# def transition_kernel(state, env, agent):
#     env.env.state = state
#     agent_action, _ = agent.predict(state)
#
#     next_state, _, _, _, _ = env.step(agent_action)
#
#     # Stochasticity
#     probabilities = np.random.uniform(size=4)
#     brownian = []
#     for p in probabilities:
#         if p < 0.5:
#             brownian.append(0.9)
#         else:
#             brownian.append(1.1)
#
#     brownian_state = [p * s for p, s in zip(brownian, next_state)]
#
#     return np.array(brownian_state)
#
#
# env = gym.make('CartPole-v1')
# loaded_model = DQN.load("maximal_cart_pole_agent")
# env.reset(seed=100)
#
# P = partial(transition_kernel, env=env, agent=loaded_model)
#
#
# def V(state):
#     return state[2] ** 2
#
# def E_V_P(state):
#     new_angle = P(state)[2]
#     return ((new_angle * 0.9) ** 2 + (new_angle * 1.1) ** 2) / 2
#
# def R(state):
#     return E_V_P(state) - V(state)
#
# env.close()

#from cart_pole_params import A, B, K


def state_noise(state):
    noise_dict = {}
    for k in range(len(state)):
        noise_dict[str(k)] = (state[k] * 0.9, state[k] * 1.1)

    return list(product(*noise_dict.values()))
def transition_kernel(x, K, A, B):
    u = -K @ x
    # TODO: Stochastic element?
    xprime = A @ x + B @ u
    return np.array(state_noise(xprime))

def V(x):
    return sum([xs[0] ** 2 for xs in x])

 # # Initial state and parameters for simulation
# x = np.array([[0.5], [0.5], [0.5], [0.5]])  # Example initial state (adjust as needed)
# Ts = 0.01  # Sampling time
# t = np.arange(0, 5, Ts)
# states = [x]
# P_values = [V(x)]
#
# # Simulate system over time using transition kernel
# from cart_pole_params import K,A,B
# for _ in t[1:]:
#     x = transition_kernel(x, K, A, B)
#     states.append(x)
#     P_values.append(V(x))
#
# # Convert results to numpy arrays for easier plotting
# states = np.array(states)
# P_values = np.array(P_values)
#
# # Plot the first graph (Cart Position and Pendulum Angle)
# fig, ax1 = plt.subplots()
#
# # Plot cart position on left y-axis
# ax1.plot(t, states[:, 0], 'b-', label='Cart Position (m)')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Cart Position (m)', color='b')
# ax1.tick_params(axis='y', labelcolor='b')
# ax1.grid()
#
# # Plot pendulum angle on right y-axis
# ax2 = ax1.twinx()
# ax2.plot(t, states[:, 2], 'r-', label='Pendulum Angle (radians)')
# ax2.set_ylabel('Pendulum Angle (radians)', color='r')
# ax2.tick_params(axis='y', labelcolor='r')
#
# # Title and layout
# fig.suptitle('Step Response with Digital LQR Control')
# fig.tight_layout()
#
# # Plot the second graph (Evolution of P for each visited state)
# plt.figure()
# plt.plot(t, P_values, 'g-', label='V(x)')
# plt.xlabel('Time (s)')
# plt.ylabel('Value Function V(x)')
# plt.title('Evolution of V for each visited state')
# plt.grid()
# plt.legend()
#
# # Show plots
# plt.show()