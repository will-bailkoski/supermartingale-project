
from itertools import product

list1 = ['0.9x', '1.1x']
list2 = ['0.9y', '1.1y']
list3 = ['0.9z', '1.1z']
print(list(product(list1, list2, list3)))