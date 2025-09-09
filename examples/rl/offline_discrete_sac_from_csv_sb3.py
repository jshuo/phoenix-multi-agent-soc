"""
Offline DQN on logged device history (pure off-policy)
------------------------------------------------------
Updated version using DQN instead of DiscreteSAC for discrete action spaces.
DQN is well-suited for offline RL and discrete actions.

python offline_discrete_sac_from_csv_sb3.py \
  --csv offpolicy_device_history.csv \    
  --total-steps 8000 --batch-size 1024 --gamma 0.99 --lr 3e-4

"""


import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN  # Changed from DiscreteSAC to DQN

FEATURES = [
    "nis99_rate","nis95_rate","temp_sla_violation","temp_jump_rate",
    "press_residual_proxy","pressure_jump_rate","route_corridor_dev_km",
    "speed_spike_rate","accel_spike_rate","ts_jitter_sec","non_monotonic_ts_rate",
    "missing_frac","battery_pct","cal_age_hours","gnss_hiacc_mode","trust_score"
]
N_ACTIONS = 5  # 0 monitor, 1 escalate, 2 calibrate, 3 peer_check, 4 flag

@dataclass
class Args:
    csv: str = "offpolicy_device_history.csv"
    total_steps: int = 800_000
    batch_size: int = 1024
    buffer_mult: float = 5.0  # buffer_size = buffer_mult * dataset_size
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 1.0  # DQN uses hard updates by default
    seed: int = 0
    train_frac: float = 0.9
    logdir: str = "sb3_offline_dqn"  # Updated directory name

# ----------------------------
# Dataset utilities (unchanged)
# ----------------------------

def build_episodes(df: pd.DataFrame) -> List[pd.DataFrame]:
    eps = []
    for dev, g in df.groupby("device_id"):
        g = g.sort_values("t").reset_index(drop=True)
        eps.append(g)
    return eps

def train_val_split(episodes: List[pd.DataFrame], frac=0.9, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    k = int(len(idx) * frac)
    train_idx, val_idx = idx[:k], idx[k:]
    return [episodes[i] for i in train_idx], [episodes[i] for i in val_idx]

class OfflineDeviceEnv(gym.Env):
    """
    Offline env that replays transitions from a concatenated set of episodes.
    The agent's action is ignored; we emit the logged (s, r, s', done) deterministically.
    """
    metadata = {"render_modes": []}
    def __init__(self, obs_arr, act_arr, rew_arr, next_obs_arr, done_arr):
        super().__init__()
        self.obs_arr = obs_arr.astype(np.float32)
        self.act_arr = act_arr.astype(np.int64)
        self.rew_arr = rew_arr.astype(np.float32)
        self.next_obs_arr = next_obs_arr.astype(np.float32)
        self.done_arr = done_arr.astype(bool)
        self.n = self.obs_arr.shape[0]
        self.idx = 0
        obs_dim = self.obs_arr.shape[1]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a random transition to decorrelate batches
        self.idx = np.random.randint(0, self.n)
        return self.obs_arr[self.idx], {}

    def step(self, action):
        # Ignore action; return logged transition at idx
        o = self.obs_arr[self.idx]
        a = self.act_arr[self.idx]
        r = self.rew_arr[self.idx]
        op = self.next_obs_arr[self.idx]
        d = bool(self.done_arr[self.idx])
        info = {"behavior_action": int(a)}
        # advance with wrap-around to keep sampling
        self.idx = (self.idx + 1) % self.n
        return op, float(r), d, False, info

# ----------------------------
# Preprocessing (unchanged)
# ----------------------------

def stack_transitions(episodes: List[pd.DataFrame], mu=None, sig=None):
    obs_list, act_list, rew_list, done_list, next_obs_list = [], [], [], [], []
    # for feature stats
    feat_stack = []
    for ep in episodes:
        feat_stack.append(ep[FEATURES].to_numpy(dtype=np.float32))
    feat_all = np.concatenate(feat_stack, axis=0)
    if mu is None:
        mu = feat_all.mean(axis=0)
        sig = feat_all.std(axis=0) + 1e-6
    # build transitions for each episode
    for ep in episodes:
        X = ep[FEATURES].to_numpy(dtype=np.float32)
        X = (X - mu) / sig
        A = ep["action"].to_numpy(dtype=np.int64)
        R = ep["reward"].to_numpy(dtype=np.float32)
        D = ep["done"].to_numpy(dtype=bool)
        Xp = np.roll(X, -1, axis=0)
        Xp[-1] = X[-1]
        obs_list.append(X)
        act_list.append(A)
        rew_list.append(R)
        done_list.append(D)
        next_obs_list.append(Xp)
    obs = np.concatenate(obs_list, axis=0)
    acts = np.concatenate(act_list, axis=0)
    rews = np.concatenate(rew_list, axis=0)
    dones = np.concatenate(done_list, axis=0)
    next_obs = np.concatenate(next_obs_list, axis=0)
    return obs, acts, rews, dones, next_obs, mu, sig

# ----------------------------
# Training (Updated for DQN)
# ----------------------------

def main(cli: Args):
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    df = pd.read_csv(cli.csv)
    episodes = build_episodes(df)
    train_eps, val_eps = train_val_split(episodes, frac=cli.train_frac, seed=cli.seed)

    # transitions + normalization
    obs, acts, rews, dones, next_obs, mu, sig = stack_transitions(train_eps)

    # Env wraps the dataset (ignored actions)
    env = OfflineDeviceEnv(obs, acts, rews, next_obs, dones)
    vec_env = DummyVecEnv([lambda: env])

    dataset_size = obs.shape[0]
    buffer_size = int(cli.buffer_mult * dataset_size)

    # Updated DQN configuration
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=cli.lr,
        batch_size=cli.batch_size,
        gamma=cli.gamma,
        buffer_size=buffer_size,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=int(cli.tau * cli.total_steps) if cli.tau < 1.0 else int(cli.tau),  # Handle tau parameter
        exploration_fraction=0.0,  # No exploration since we're doing offline RL
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        verbose=1,
        seed=cli.seed,
    )

    # Preload replay buffer with logged transitions
    # DQN expects actions as integers (unlike SAC which expects float arrays)
    obs_dim = obs.shape[1]
    
    # For DQN, we need to add transitions one by one
    for i in range(dataset_size):
        model.replay_buffer.add(
            obs=obs[i],
            next_obs=next_obs[i],
            action=acts[i],
            reward=rews[i],
            done=dones[i],
            infos=[{}],
        )

    print(f"Preloaded {dataset_size} transitions into replay buffer")

    # Learn purely from buffer; env rollouts just advance indices (ignored actions)
    model.learn(total_timesteps=cli.total_steps, progress_bar=True)

    # Save artifacts
    out = Path(cli.logdir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "checkpoint.zip")
    np.savez(out / "scaler.npz", mu=mu, sig=sig, features=np.array(FEATURES))

    # Simple offline eval: avg reward under behavior data (not on-policy return)
    report = {
        "dataset_size": int(dataset_size),
        "mean_reward_in_logs": float(rews.mean()),
        "std_reward_in_logs": float(rews.std()),
    }
    import json
    with open(out / "train_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved:", out / "checkpoint.zip", out / "scaler.npz", out / "train_report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=Args.csv)
    p.add_argument('--total-steps', type=int, default=Args.total_steps)
    p.add_argument('--batch-size', type=int, default=Args.batch_size)
    p.add_argument('--buffer-mult', type=float, default=Args.buffer_mult)
    p.add_argument('--gamma', type=float, default=Args.gamma)
    p.add_argument('--lr', type=float, default=Args.lr)
    p.add_argument('--tau', type=float, default=Args.tau)
    p.add_argument('--seed', type=int, default=Args.seed)
    p.add_argument('--train-frac', type=float, default=Args.train_frac)
    p.add_argument('--logdir', type=str, default=Args.logdir)
    args = p.parse_args()
    main(Args(**vars(args)))