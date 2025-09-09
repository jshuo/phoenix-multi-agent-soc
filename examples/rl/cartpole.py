import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment
env = gym.make("CartPole-v1")

# 2. Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the agent
model.learn(total_timesteps=200_000)

# 4. Save the trained model
model.save("ppo_cartpole")

# 5. Evaluate: run a few episodes with the trained agent
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

env.close()
