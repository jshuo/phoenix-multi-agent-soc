# pip install gymnasium stable-baselines3
import gymnasium as gym
from stable_baselines3 import PPO
from fab_env import FabDispatchEnv

env = FabDispatchEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Evaluate
obs, _ = env.reset()
ret = 0
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    ret += reward
    if term or trunc:
        break

print("Return:", ret, "| Completed:", env.total_completed, "| Defects:", env.total_defects)
