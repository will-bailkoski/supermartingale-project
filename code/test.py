lst = [1, 2, 3, 4, 5, 6]

# List comprehension to create a list of adjacent pairs
adjacent_pairs = [(lst[i], lst[i+1]) for i in range(len(lst) - 1)]

print(adjacent_pairs)


from model import C, V_threshold
import numpy as np
# print(np.fill_diagonal np.array([np.random.uniform(0, 0.01, 2), np.random.uniform(0, 0.01, 2)]))

print(np.multiply(C, V_threshold))
print(np.dot(C, V_threshold))