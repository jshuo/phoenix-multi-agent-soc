# rl_trust_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RLTrustEnv(gym.Env):
    """
    Per-fix triage for tracker measurements.
    Simulates a stream where some fixes are "true" and some are "false" (spoof/noisy).
    Agent chooses: accept / request_more / escalate_accuracy / flag.
    Rewards balance correctness vs. cost/latency.

    You can later replace the simulator with your real per-fix feature stream and labels.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 seq_len=300,            # steps per episode
                 base_false_rate=0.12,   # prior prob a fix is false
                 seed=None):
        super().__init__()
        self.seq_len = seq_len
        self.base_false_rate = base_false_rate
        self.rng = np.random.default_rng(seed)

        # Observation features (all normalized-ish):
        # [NIS, innov_norm, route_dev_m, peer_agree(0..1), speed_z, accel_z,
        #  ts_jitter_s, temp_bad(0/1), press_bad(0/1), recent_rule_rate(0..1),
        #  trust(0..1), battery(0..1), hiacc_on(0/1)]
        self.obs_dim = 13
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32)

        # Actions: 0=accept, 1=request_more, 2=escalate_accuracy, 3=flag
        self.action_space = spaces.Discrete(4)

        # Costs / rewards
        self.R_CORRECT = 1.0
        self.C_FALSE_NEG = 5.0
        self.C_FALSE_POS = 3.0
        self.C_WAIT = 0.15
        self.C_HIACC = 0.3

        # Internals
        self.t = 0
        self.trust = 0.7
        self.battery = 1.0
        self.hiacc = 0  # high-accuracy mode (0/1)
        self.recent_rule_rate = 0.0

        # buffers
        self.false_curr = False
        self._gen_fix()  # initialize first fix

    # ---------- simulator for a single fix ----------
    def _gen_fix(self):
        # mixture: with prob base_false_rate (adjusted by trust), generate a false fix
        p_false = np.clip(self.base_false_rate + (0.25 - 0.5*self.trust), 0.01, 0.7)
        self.false_curr = self.rng.random() < p_false

        # Generate features:
        # If false → inflate NIS, route deviation, lower peer agreement, weird phys flags
        # If hiacc → reduce NIS/innov/route dev a bit but costs later
        base = 1.0 if not self.false_curr else 3.0
        hiacc_scale = 0.6 if self.hiacc else 1.0

        nis = np.abs(self.rng.normal(base, 0.8)) * hiacc_scale
        innov = np.abs(self.rng.normal(base*5, 2.0)) * hiacc_scale
        route_dev = np.abs(self.rng.normal(8 if self.false_curr else 2, 3.0)) * hiacc_scale
        peer_agree = np.clip(self.rng.normal(0.8, 0.1), 0, 1) if not self.false_curr else np.clip(self.rng.normal(0.35, 0.2), 0, 1)
        speed_z = np.clip(self.rng.normal(0 if not self.false_curr else 1.5, 0.7), -3, 3)
        accel_z = np.clip(self.rng.normal(0 if not self.false_curr else 1.2, 0.7), -3, 3)
        ts_jitter = np.abs(self.rng.normal(0.15 if not self.false_curr else 0.6, 0.1))
        temp_bad = int(self.rng.random() < (0.05 if not self.false_curr else 0.20))
        press_bad = int(self.rng.random() < (0.03 if not self.false_curr else 0.15))

        # recent rule rate drifts towards more violations if many bad flags appear
        viol = (temp_bad or press_bad or (abs(speed_z) > 2.5) or (route_dev > 20))
        self.recent_rule_rate = 0.9*self.recent_rule_rate + 0.1*float(viol)
        
        # pack obs
        self.obs = np.array([
            nis, innov/10.0, route_dev/20.0, peer_agree, speed_z, accel_z,
            ts_jitter, float(temp_bad), float(press_bad), self.recent_rule_rate,
            self.trust, self.battery, float(self.hiacc)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.trust = 0.7
        self.battery = 1.0
        self.hiacc = 0
        self.recent_rule_rate = 0.0
        self._gen_fix()
        return self.obs.copy(), {}

    def step(self, action):
        reward = 0.0
        done = False
        info = {"false_fix": self.false_curr}

        # Apply action effects
        if action == 0:  # accept
            if self.false_curr:
                reward -= self.C_FALSE_NEG
                self.trust = max(0.0, self.trust - 0.15)
            else:
                reward += self.R_CORRECT
                self.trust = min(1.0, self.trust + 0.02)

        elif action == 1:  # request more evidence (latency cost)
            reward -= self.C_WAIT
            # increase accuracy of the *next* observation marginally
            self.trust = max(0.0, self.trust - 0.005)

        elif action == 2:  # escalate accuracy (turn on hi-acc mode for a few steps)
            reward -= self.C_HIACC
            self.hiacc = 1
            self.battery = max(0.0, self.battery - 0.01)

        elif action == 3:  # flag as false
            if self.false_curr:
                reward += self.R_CORRECT
                self.trust = max(0.0, self.trust - 0.02)  # conservative after a problem
            else:
                reward -= self.C_FALSE_POS
                self.trust = max(0.0, self.trust - 0.08)

        # High-accuracy auto-timeout
        if self.hiacc and self.rng.random() < 0.2:
            self.hiacc = 0

        # Small shaping to encourage consistency-based reasoning
        nis = float(self.obs[0])
        route_dev_scaled = float(self.obs[2])
        reward -= 0.02*nis - 0.01*(1.0 - route_dev_scaled)

        # Battery drain per step (radio on etc.)
        self.battery = max(0.0, self.battery - 0.001)

        self.t += 1
        if self.t >= self.seq_len or self.battery <= 1e-4:
            done = True

        # Next fix
        self._gen_fix()

        return self.obs.copy(), float(reward), done, False, info
