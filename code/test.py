import numpy as np
n =2
C = np.random.uniform(0, 0.01, (n, n))
np.fill_diagonal(C, 0)
print(C)