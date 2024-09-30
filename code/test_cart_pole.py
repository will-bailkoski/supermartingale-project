from cart_pole_functions import P

next_state = [ 0.0009979,   0.22532752, -0.0033824,  -0.2808728 ]
value = next_state[2] ** 2
for i in range(30):
    next_state, reward, done, truncated, info = P(next_state)
    new_value = next_state[2] ** 2
    if new_value < value:
        print("WINNER!")
    else:
        print("loser")
    value = new_value
