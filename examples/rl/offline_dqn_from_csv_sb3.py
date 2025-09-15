"""
Offline DQN on logged device history (pure off-policy)
------------------------------------------------------
Train DQN from a fixed CSV dataset by:
  1) Normalizing features
  2) Wrapping the dataset in a read-only Gymnasium env (agent action ignored)
  3) Preloading the SB3 replay buffer with logged (s, a, r, s', done)
  4) Learning purely from the buffer

Run:
  python offline_dqn_from_csv_sb3.py \
    --csv offpolicy_device_history.csv \
    --total-steps 800000 --batch-size 1024 --gamma 0.99 --lr 3e-4
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN

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
    seed: int = 0
    train_frac: float = 0.9
    logdir: str = "sb3_offline_dqn"
    target_update_interval: int = 1000  # proper DQN target update control

# ----------------------------
# Dataset utilities
# ----------------------------

def build_episodes(df: pd.DataFrame) -> List[pd.DataFrame]:
    eps = []
    for _, g in df.groupby("device_id"):
        g = g.sort_values("t").reset_index(drop=True)
        eps.append(g)
    return eps

def train_val_split(episodes: List[pd.DataFrame], frac=0.9, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    k = int(len(idx) * frac)
    return [episodes[i] for i in idx[:k]], [episodes[i] for i in idx[k:]]

def reward_fn(action, *, label=None, nis99_rate=0.0, temp_jump_rate=0.0, press_residual=0.0):
    # 1) action costs
    r = {0:0.0, 1:-0.2, 2:-0.1, 3:-0.05, 4:0.0}[action]
    # 2) outcome scoring (if label known at this step)
    if label in (0, 1):
        if action == 4 and label == 1:   r += 1.0          # correct flag
        elif action == 4 and label == 0: r -= 3.0          # false positive
        elif action != 4 and label == 1: r -= 10.0         # false negative
        elif action != 4 and label == 0: r += 1.0          # correct keep
    # 3) shaping penalties
    r -= 0.5 * nis99_rate
    r -= 0.2 * temp_jump_rate
    r -= 0.2 * (press_residual / 20.0)
    return float(r)

# ----------------------------
# Replay-only environment
# ----------------------------

class OfflineDeviceEnv(gym.Env):
    """
    Offline env that replays states from a dataset,
    but computes reward using reward_fn() instead of a precomputed column.
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, mu, sig):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.mu, self.sig = mu, sig
        self.n = len(self.df)
        self.idx = 0

        obs_dim = len(FEATURES)
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = np.random.randint(0, self.n)
        return self._get_obs(self.idx), {}

    def _get_obs(self, i):
        # normalize feature vector
        x = self.df.loc[i, FEATURES].to_numpy(dtype=np.float32)
        return ((x - self.mu) / self.sig).astype(np.float32)

    def step(self, action: int):
        row = self.df.iloc[self.idx]

        # Compute reward dynamically using reward_fn
        r = reward_fn(
            action,
            label=row.get("hard_label", None),
            nis99_rate=row["nis99_rate"],
            temp_jump_rate=row["temp_jump_rate"],
            press_residual=row["press_residual_proxy"],
        )

        # Advance to next
        next_idx = (self.idx + 1) % self.n
        obs_next = self._get_obs(next_idx)

        done = bool(row.get("done", False))
        info = {"device_id": row["device_id"], "t": row["t"]}

        self.idx = next_idx
        return obs_next, r, done, False, info

# ----------------------------
# Preprocessing
# ----------------------------

def stack_transitions(episodes: List[pd.DataFrame], mu=None, sig=None):
    # Validate required columns exist
    required_cols = set(FEATURES + ["action"])
    missing = required_cols - set(episodes[0].columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    obs_list, act_list, rew_list, done_list, next_obs_list = [], [], [], [], []

    # Flatten features to compute scaler if needed
    feat_stack = [ep[FEATURES].to_numpy(dtype=np.float32) for ep in episodes]
    feat_all = np.concatenate(feat_stack, axis=0)

    if mu is None:
        mu = feat_all.mean(axis=0)
        sig = feat_all.std(axis=0) + 1e-6

    for ep in episodes:
        # Normalize observations with provided scaler
        X_raw = ep[FEATURES].to_numpy(dtype=np.float32)
        X = (X_raw - mu) / sig

        # Logged actions (ensure in range)
        A = ep["action"].to_numpy(dtype=np.int64)
        if (A < 0).any() or (A >= N_ACTIONS).any():
            raise ValueError("Found action outside [0, N_ACTIONS). Check your CSV.")

        # Compute rewards from reward_fn using per-row signals
        # Optional columns default gracefully via .get
        hard_label = ep["hard_label"] if "hard_label" in ep.columns else pd.Series([None] * len(ep))
        nis99 = ep["nis99_rate"]
        tjump = ep["temp_jump_rate"]
        pres  = ep["press_residual_proxy"]

        R = np.array(
            [
                reward_fn(
                    int(a),
                    label=(None if pd.isna(hl) else int(hl)),
                    nis99_rate=float(nr),
                    temp_jump_rate=float(tj),
                    press_residual=float(pr),
                )
                for a, hl, nr, tj, pr in zip(A, hard_label, nis99, tjump, pres)
            ],
            dtype=np.float32,
        )

        # Dones (optional; default False)
        D = ep["done"].to_numpy(dtype=bool) if "done" in ep.columns else np.zeros(len(ep), dtype=bool)

        # Next-obs: shift by one within episode
        Xp = np.roll(X, -1, axis=0)
        Xp[-1] = X[-1]

        obs_list.append(X)
        act_list.append(A.astype(np.int64))
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
# Training
# ----------------------------

def main(cli: Args):
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    df = pd.read_csv(cli.csv)
    episodes = build_episodes(df)
    train_eps, _ = train_val_split(episodes, frac=cli.train_frac, seed=cli.seed)

    # Concatenate training episodes back into a single DataFrame
    train_df = pd.concat(train_eps, ignore_index=True)

    # Compute normalization stats
    feat_all = train_df[FEATURES].to_numpy(dtype=np.float32)
    mu, sig = feat_all.mean(axis=0), feat_all.std(axis=0) + 1e-6

    # Create environment with dynamic reward
    env = OfflineDeviceEnv(train_df, mu, sig)
    vec_env = DummyVecEnv([lambda: env])


    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=cli.lr,
        batch_size=cli.batch_size,
        gamma=cli.gamma,
        buffer_size=200000,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=cli.target_update_interval,
        exploration_fraction=0.0,     # no exploration in strict offline
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        verbose=1,
        seed=cli.seed,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # After creating the model, preload the buffer
    obs, acts, rews, dones, next_obs, mu, sig = stack_transitions(train_eps)

    # Add transitions to replay buffer
    for i in range(len(obs)):
        model.replay_buffer.add(
            obs[i],
            next_obs[i],
            acts[i],
            rews[i],
            dones[i],
            infos=[{"TimeLimit.truncated": False}]  # length == n_envs (1)
    )


    # Learn purely from the preloaded buffer
    model.learn(total_timesteps=cli.total_steps, progress_bar=True)

    # Save artifacts
    out = Path(cli.logdir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "checkpoint.zip")
    np.savez(out / "scaler.npz", mu=mu, sig=sig, features=np.array(FEATURES))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=Args.csv)
    p.add_argument('--total-steps', type=int, default=Args.total_steps)
    p.add_argument('--batch-size', type=int, default=Args.batch_size)
    p.add_argument('--buffer-mult', type=float, default=Args.buffer_mult)
    p.add_argument('--gamma', type=float, default=Args.gamma)
    p.add_argument('--lr', type=float, default=Args.lr)
    p.add_argument('--seed', type=int, default=Args.seed)
    p.add_argument('--train-frac', type=float, default=Args.train_frac)
    p.add_argument('--logdir', type=str, default=Args.logdir)
    p.add_argument('--target-update-interval', type=int, default=Args.target_update_interval)
    args = p.parse_args()
    main(Args(**vars(args)))
