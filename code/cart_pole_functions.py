import gymnasium as gym
from stable_baselines3 import DQN
from functools import partial

# Function to step the environment forward given a state and an action
def transition_kernel(state, env, agent):
    env.reset()
    env.env.state = state  # Assuming the environment supports manually setting state (CartPole does)
    agent_action, _ = agent.predict(state)

    next_state, reward, done, truncated, info = env.step(agent_action)

    return next_state, reward, done, truncated, info


env = gym.make('CartPole-v1')
loaded_model = DQN.load("dqn_cartpole_agent1")

P = partial(transition_kernel, env=env, agent=loaded_model)


def V(state):
    return state[2] ** 2

def E_V_P(state):
    return

def R(state):
    return E_V_P(state) - V(state)

env.close()
