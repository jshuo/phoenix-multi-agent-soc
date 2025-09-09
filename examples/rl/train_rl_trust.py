# train_rl_trust.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from rl_trust_env import RLTrustEnv

SEED = 7

def make_env():
    return RLTrustEnv(seq_len=400, base_false_rate=0.15, seed=SEED)

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=512,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=SEED,
        verbose=1,
        device="auto",
    )

    callback = EvalCallback(eval_env, best_model_save_path="./trust_ckpt", log_path="./trust_logs", eval_freq=10_000)
    model.learn(total_timesteps=600_000, callback=callback)
    model.save("ppo_rl_trust_agent")

    # quick smoke test
    test_env = RLTrustEnv(seq_len=200, base_false_rate=0.2, seed=SEED+1)
    obs, _ = test_env.reset()
    ret = 0.0
    flags = accs = 0
    for _ in range(200):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = test_env.step(act)
        ret += r
        if act == 3:  # flagged
            flags += 1
        if act == 0:  # accepted
            accs += 1
        if done: break
    print(f"Return {ret:.2f}, accepts={accs}, flags={flags}")
