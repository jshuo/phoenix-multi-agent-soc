# Install first (if needed):
# pip install stable-baselines3 gymnasium

import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment (CartPole is the "hello world" of RL)
env = gym.make("CartPole-v1")

# 2. Initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the agent
model.learn(total_timesteps=20_000)

# 4. Test the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
