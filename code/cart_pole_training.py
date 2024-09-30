import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Create the DQN agent
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02, target_update_interval=500, train_freq=1, gradient_steps=1)

# Train the agent
model.learn(total_timesteps=100000)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

model.save("dqn_cartpole_agent1")

# Close the environment
env.close()