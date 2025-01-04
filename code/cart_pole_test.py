from cart_pole_functions import transition_kernel, V
from cart_pole_params import A,B,K

from functools import partial

from MAB_algorithm import mab_algorithm

#x = [[0.5],[0.5],[0.5],[0.5]]

P = partial(transition_kernel, A=A, B=B, K=K)
#print(len(P(x)))
#print(sum([V(state) for state in P(x)])/len(P(x)) - V(x))

#U = ((A- B@K) @ [[1],[1],[1],[1]]) ** 2 - [[1],[1],[1],[1]]
#print(U)

#print(sum([ i * (a[0] **2) for i, a in zip(U, x)]))

print(mab_algorithm(
    initial_bounds=[(-0.5, 0.5)] * 4,
    dynamics=P,
    certificate=V,
    lipschitz=10,
    reward_range=10,
    max_iterations=10000,
    tolerance=0.1
))
