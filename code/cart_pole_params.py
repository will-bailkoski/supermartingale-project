"""Taken from https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import dlsim, dlti
import matplotlib.pyplot as plt

A = np.array([[1,    0.009991,   0.0001336,   4.453e-07],
              [0,      0.9982,     0.02672,   0.0001336],
              [0,  -2.272e-05,       1.002,     0.01001],
              [0,   -0.004544,      0.3119,       1.002]])

B = np.array([[9.086e-05],
              [0.01817],
              [0.0002272],
              [0.04544]])

C = np.array([[1,0,0,0],
              [0,0,1,0]])

D = np.array([[0],
              [0]])

Q = np.array([[1,0,0,0],
              [0,0,0,0],
              [0,0,1,0],
              [0,0,0,0]])

K = np.array([[-0.9384, -1.5656, 18.0351, 3.3368]])

R = 1



# Solve the discrete-time Algebraic Riccati equation
# P = solve_discrete_are(A, B, Q, R)
# K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
# print(K)
#
#
# # Define the closed-loop state-space system
# Ac = A - B @ K
# Bc = B
# Cc = C
# Dc = D
# Ts = 0.01  # Sampling time, equivalent to MATLAB's `Ts`
#
# # Create the state-space system for closed-loop
# sys_cl = dlti(Ac, Bc, Cc, Dc, dt=Ts)
#
# # Define the time vector and input (reference step)
# t = np.arange(0, 0.01, Ts)
# r = 0.2 * np.ones_like(t)  # Step input of 0.2
#
# # Simulate the closed-loop response
# _, y, x = dlsim(sys_cl, r, t)
#
# # Simulate the response
# _, y, x = dlsim(sys_cl, r, t)
#
# print(x[0])
#
# # Plot with two y-axes
# fig, ax1 = plt.subplots()
#
# # Plot cart position on the left y-axis
# ax1.plot(t, y[:, 0], 'b-', label='Cart Position (m)')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Cart Position (m)', color='b')
# ax1.tick_params(axis='y', labelcolor='b')
# ax1.grid()
#
# # Create a second y-axis for the pendulum angle
# ax2 = ax1.twinx()
# ax2.plot(t, y[:, 1], 'r-', label='Pendulum Angle (radians)')
# ax2.set_ylabel('Pendulum Angle (radians)', color='r')
# ax2.tick_params(axis='y', labelcolor='r')
#
# # Title and layout
# fig.suptitle('Step Response with Digital LQR Control')
# fig.tight_layout()
# plt.show()
