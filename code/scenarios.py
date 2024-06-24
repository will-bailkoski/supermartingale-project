import numpy as np
import random

scenarios = {
    "Stable System": {
        "C": np.array([[0, 0.02], [0.01, 0]]),
        "D": np.array([[0.03, 0.03], [0.03, 0.03]]),
        "p": np.array([20, 20]),
        "B": np.diag([10, 10]),
        "V_threshold": np.array([1.5, 1.5]),
        "V_initial": np.array([5, 5])
    },
    "Single Failure Propagation": {
        "C": np.array([[0, 0.025], [0.005, 0]]),
        "D": np.array([[0.05, 0.05], [0.05, 0.05]]),
        "p": np.array([20, 20]),
        "B": np.diag([12, 12]),
        "V_threshold": np.array([1.5, 1.5]),
        "V_initial": np.array([1.6, 5])
    },
    "Multiple Failures": {
        "C": np.array([[0, 0.03], [0.03, 0]]),
        "D": np.array([[0.04, 0.04], [0.04, 0.04]]),
        "p": np.array([20, 20]),
        "B": np.diag([15, 15]),
        "V_threshold": np.array([2, 2]),
        "V_initial": np.array([2.1, 2.1])
    },
    "Recovery Scenario": {
        "C": np.array([[0, 0.03], [0.01, 0]]),
        "D": np.array([[0.04, 0.04], [0.04, 0.04]]),
        "p": np.array([20, 20]),
        "B": np.diag([12, 12]),
        "V_threshold": np.array([1.5, 1.5]),
        "V_initial": np.array([1.6, 1.6])
    },
    "Example 4: Countries": {
        "C": np.array([
            [0, 0.03, 0.01, 0.07, 0.01, 0.04, 0.04, 0.05, 0.04],  # FR
            [0.04, 0, 0.06, 0.03, 0.00, 0.05, 0.04, 0.09, 0.04],  # DE
            [0.00, 0.00, 0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # GR
            [0.01, 0.03, 0.00, 0, 0.00, 0.01, 0.02, 0.01, 0.00],  # IT
            [0.04, 0.02, 0.00, 0.02, 0, 0.01, 0.01, 0.06, 0.10],  # JP
            [0.00, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00, 0.00],  # PT
            [0.01, 0.02, 0.01, 0.02, 0.00, 0.15, 0, 0.09, 0.02],  # ES
            [0.03, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0, 0.04],  # GB
            [0.04, 0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.09, 0]  # US
        ]),
        "D": np.eye(9),
        "p": np.array([[12.29, 16.81, 1.02, 9.30, 20.00, 1.00, 6.00, 12.99, 75.70]]).T,
        "B": np.diag(np.array([0.5] * 9)),
        "V_threshold": np.array([[10] * 9]).T,
        "V_initial": np.array([[15.2838, 19.9137, 0.9863, 9.0642, 28.3350, 0.7829, 8.8020, 12.1361, 59.8130]]).T
    },
    "Random Gen": {
        "C": np.array([np.random.uniform(0, 0.01, 2), np.random.uniform(0, 0.01, 2)]),  # fill diagonal
        "D": np.array([[0.05] * 2] * 2).T,
        "p": np.array([[10] * 2]).T,
        "B": np.diag(np.array([0.4] * 2)),
        "V_threshold": np.array([[5] * 2]).T,
        "V_initial": np.array([np.random.uniform(0, 30, 2)]).T
    }

}
