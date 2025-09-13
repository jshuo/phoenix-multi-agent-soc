# streaming_kf_if_rl_metrics.py
# KF -> IsolationForest -> RL (tabular Q) with rich metrics payload

import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest

ACTIONS = ["monitor", "escalate", "calibrate", "peer_check", "flag"]

# ---------------- Kalman filter (1D, per-sample) ----------------
class SimpleKF1D:
    def __init__(self, x0, P0=1.0, Q=1e-4, R=0.25):
        self.x, self.P, self.Q, self.R = float(x0), float(P0), float(Q), float(R)
    def step(self, z):
        # predict
        Pp = self.P + self.Q
        # innovation
        v = z - self.x
        S = Pp + self.R
        K = Pp / S
        # update
        self.x = self.x + K * v
        self.P = (1 - K) * Pp
        return self.x, v, S  # return state, innovation, innovation variance

# ---------------- Tiny tabular Q-policy (3 states x 5 actions) ----------------
def bucket_anomaly(score):
    return 0 if score < 0.1 else (1 if score < 0.3 else 2)

def reward_fn(state, action):
    #                   0:monitor 1:escalate 2:calibrate 3:peer_check 4:flag
    cost = {0:0.0,      1:-0.4,   2:-0.2,     3:-0.05,     4:-0.5}[action]  # cheaper to peer_check

    if state == 0:  # low anomaly
        gain = {0:1.0,  3:0.4,     2:0.2,      1:-0.6,      4:-1.0}[action]
    elif state == 1:  # medium anomaly → prefer peer_check
        gain = {3:1.2,  0:0.3,     2:0.2,      1:-0.1,      4:-0.6}[action]
    else:  # state == 2 high anomaly
        gain = {4:1.1,  1:0.8,     3:0.5,      2:0.3,       0:-1.0}[action]

    return gain + cost

def train_q_policy(episodes=800, gamma=0.9, alpha=0.3, eps_start=0.9, eps_end=0.05):
    Q = np.zeros((3,5))
    eps = eps_start; decay = (eps_start-eps_end)/max(1,episodes-1)
    for _ in range(episodes):
        s = np.random.choice([0,1,2], p=[0.6,0.25,0.15])
        for _ in range(5):
            a = np.random.randint(5) if np.random.rand()<eps else int(np.argmax(Q[s]))
            r = reward_fn(s,a)
            s_next = (0 if s==0 and np.random.rand()<0.85 else
                      2 if s==2 and np.random.rand()<0.75 else 1)
            """
            Q[s,a] → the current estimate of the Q-value.
            r → the immediate reward, given by reward_fn(state, action).
            gamma*np.max(Q[s_next]) → the discounted future reward, i.e. the best expected value of the next state.
            (1-alpha)*Q[s,a] + alpha*(...) → the learning rate update rule, blending old estimate with new Bellman target.
            """
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*(r + gamma*np.max(Q[s_next]))
            s = s_next
        eps = max(eps_end, eps-decay)
    return Q

# ---------------- Streaming pipeline with rich metrics ----------------
class StreamingPipeline:
    def __init__(self, w=60, step=15, refit_every=10, contam=0.07, dt_nominal=60.0):
        self.w, self.step, self.refit_every = w, step, refit_every
        self.dt_nominal = dt_nominal

        # rolling windows
        self.temp_z = deque(maxlen=w);   self.press_z = deque(maxlen=w)
        self.temp_x = deque(maxlen=w);   self.press_x = deque(maxlen=w)
        self.temp_v = deque(maxlen=w);   self.press_v = deque(maxlen=w)  # innovation
        self.temp_S = deque(maxlen=w);   self.press_S = deque(maxlen=w)  # innovation var
        self.ts     = deque(maxlen=w)    # timestamps (seconds)
        self.speed  = deque(maxlen=w);   self.accel = deque(maxlen=w)
        self.route_dev_km = deque(maxlen=w)
        self.batt   = deque(maxlen=w)
        self.gnss_mode = deque(maxlen=w)

        # filters
        self.kf_t = None; self.kf_p = None

        # model bits
        self.iforest = None; self.contam = contam
        self.Q = train_q_policy()

        self.window_i = 0
        self.feature_buffer = []  # recent features for (re)fitting IF

    def _ensure_kf(self, t0, p0):
        if self.kf_t is None: self.kf_t = SimpleKF1D(t0, R=0.1)
        if self.kf_p is None: self.kf_p = SimpleKF1D(p0, R=0.15)

    def _features_from_window(self):
        # numpy views
        tm = np.array(self.temp_z); pm = np.array(self.press_z)
        tx = np.array(self.temp_x); px = np.array(self.press_x)
        tv = np.array(self.temp_v); pv = np.array(self.press_v)
        tS = np.array(self.temp_S); pS = np.array(self.press_S)
        tvec = np.array(self.ts)

        # residuals (meas - kf state)
        r_t = np.abs(tm - tx); r_p = np.abs(pm - px)
        press_residual_proxy = float(np.mean(r_p))  # kPa mean abs residual

        # jumps
        dt_temp  = np.abs(np.diff(tm, prepend=tm[0]))
        dp_press = np.abs(np.diff(pm, prepend=pm[0]))
        temp_jump_rate  = float(np.mean(dt_temp > 2.5))
        pressure_jump_rate = float(np.mean(dp_press > 1.5))

        # SLA violation (2–8 C)
        temp_sla_violation = float(np.mean((tx < 2.0) | (tx > 8.0)))

        # timestamp metrics
        dts = np.diff(tvec)
        ts_jitter_sec = float(np.percentile(np.abs(dts - self.dt_nominal), 95)) if len(dts)>0 else 0.0
        non_monotonic_ts_rate = float(np.mean(dts < 0.0)) if len(dts)>0 else 0.0
        # missing packets proxy: gaps larger than 1.5x nominal
        missing_frac = float(np.mean(dts > 1.5*self.dt_nominal)) if len(dts)>0 else 0.0

        # route corridor deviation (mean)
        route_corridor_dev_km = float(np.mean(np.abs(np.array(self.route_dev_km)))) if len(self.route_dev_km) else 0.0

        # speed/accel spike rates
        sp = np.array(self.speed); ac = np.array(self.accel)
        speed_spike_rate = float(np.mean(np.abs(np.diff(sp, prepend=sp[0])) > 8.0)) if len(sp)>0 else 0.0  # >8 km/h step
        accel_spike_rate = float(np.mean(np.abs(ac) > 3.0)) if len(ac)>0 else 0.0  # >3 m/s^2

        # gnss mode + battery
        gnss_hiacc_mode = float(np.mean(np.asarray(self.gnss_mode))) if len(self.gnss_mode) else 0.0
        battery_pct = float(self.batt[-1]) if len(self.batt) else 1.0

        # calibration age: assume last calibrate never happened here (demo), so monotone
        # you can wire the real value from device metadata; we proxy via window length*interval/3600
        cal_age_hours = float(len(self.ts) * self.dt_nominal / 3600.0)

        # ---- NIS exceedance rates (2D chi-square) ----
        # per-sample Mahalanobis-ish with diagonal S (temp S + press S)
        # NIS = (v_t^2 / S_t) + (v_p^2 / S_p)  ~ ChiSquare(df=2) if correctly tuned
        nis_vals = (tv**2 / np.maximum(tS, 1e-9)) + (pv**2 / np.maximum(pS, 1e-9))
        nis95_rate = float(np.mean(nis_vals > 5.991))  # 95th % tile for df=2
        nis99_rate = float(np.mean(nis_vals > 9.210))  # 99th % tile for df=2

        # ---- IsolationForest feature vector ----
        feat = np.array([
            np.mean(r_t), np.mean(r_p),
            temp_jump_rate, pressure_jump_rate,
            temp_sla_violation,
            ts_jitter_sec/60.0,              # put jitter in minutes to stabilize scaling
            non_monotonic_ts_rate,
            missing_frac,
            route_corridor_dev_km,
            speed_spike_rate,
            accel_spike_rate,
            battery_pct,
            cal_age_hours/24.0,              # days to stabilize scale
            gnss_hiacc_mode
        ], dtype=float)

        # return features and all individual metrics needed for the output dict
        return feat, {
            "nis99_rate": nis99_rate,
            "nis95_rate": nis95_rate,
            "temp_sla_violation": temp_sla_violation,
            "temp_jump_rate": temp_jump_rate,
            "press_residual_proxy": press_residual_proxy,
            "pressure_jump_rate": pressure_jump_rate,
            "route_corridor_dev_km": route_corridor_dev_km,
            "speed_spike_rate": speed_spike_rate,
            "accel_spike_rate": accel_spike_rate,
            "ts_jitter_sec": ts_jitter_sec,
            "non_monotonic_ts_rate": non_monotonic_ts_rate,
            "missing_frac": missing_frac,
            "battery_pct": battery_pct,
            "cal_age_hours": cal_age_hours,
            "gnss_hiacc_mode": gnss_hiacc_mode
        }

    def update(self, packet):
        """
        packet: dict with keys:
          temp, press, ts (seconds), speed_kmh, accel_mps2, route_dev_km, battery_pct, gnss_hiacc (0/1)
        """
        t = float(packet["temp"]); p = float(packet["press"])
        self._ensure_kf(t, p)

        # KF step (get estimates + innovations for NIS)
        tx, tv, tS = self.kf_t.step(t)
        px, pv, pS = self.kf_p.step(p)

        # push into windows
        self.temp_z.append(t); self.press_z.append(p)
        self.temp_x.append(tx); self.press_x.append(px)
        self.temp_v.append(tv); self.press_v.append(pv)
        self.temp_S.append(tS); self.press_S.append(pS)

        self.ts.append(float(packet["ts"]))
        self.speed.append(float(packet["speed_kmh"]))
        self.accel.append(float(packet["accel_mps2"]))
        self.route_dev_km.append(float(packet["route_dev_km"]))
        self.batt.append(float(packet["battery_pct"]))
        self.gnss_mode.append(float(packet["gnss_hiacc"]))

        out = None
        if len(self.temp_z) == self.w and (self.window_i % self.step == 0):
            feat, base_metrics = self._features_from_window()

            # (re)fit IF periodically
            self.feature_buffer.append(feat)
            if (self.iforest is None) or (len(self.feature_buffer) % self.refit_every == 0):
                X = np.vstack(self.feature_buffer[-200:])  # cap history
                self.iforest = IsolationForest(
                    n_estimators=200, contamination=self.contam, random_state=42
                ).fit(X)

            # anomaly score (higher => more anomalous)
            score = float(-self.iforest.decision_function([feat])[0])
            # --- SLA-boosted anomaly (makes state jump more often) ---
            sla = base_metrics["temp_sla_violation"]
            score += 1.001 * sla  

            # simple trust score from anomaly score (squash & invert)
            # tweak gain for your data distribution
            trust_score = float(np.clip(1.0 / (1.0 + np.exp(4.0*(score-0.3))), 0.0, 1.0))

            # RL action
            s = bucket_anomaly(score)
            a = int(np.argmax(self.Q[s]))
            action_name = ACTIONS[a]

            # final metrics dict in requested shape
            metrics = dict(base_metrics)
            metrics.update({
                "trust_score": trust_score
            })

            out = {
                "anomaly_score": score,
                "state_bucket": s,
                "action": action_name,
                "metrics": metrics
            }

        self.window_i += 1
        return out

# ---------------- Simulated live stream (includes GPS/time-ish signals) ----------------
def simulated_stream(n=2000, start_ts=0.0, dt=60.0, spike_p=0.30, seed=0):
    rng = np.random.default_rng(seed)
    base_t, base_p = 6.5, 101.3
    speed = 40.0  # km/h baseline
    route_offset = 0.0  # km from corridor center
    battery = 1.0
    gnss = 0

    ts = start_ts
    for i in range(n):
        # physics-ish: slow sinus drifts + noise + occasional spikes
        temp = base_t + 0.4*np.sin(2*np.pi*i/144) + rng.normal(0,0.5)
        press= base_p + 0.5*np.sin(2*np.pi*i/200) + rng.normal(0,0.3)

        if rng.random() < spike_p:
            temp += rng.choice([8, -8, 12, -12, 16, -16])
            press+= rng.choice([6,-6])*0.5

        # speed (random walk) & accel
        accel = rng.normal(0, 0.6)
        speed = max(0.0, speed + accel)  # km/h
        # convert accel from Δspeed (km/h per tick) -> m/s^2 approx
        accel_mps2 = (accel * 1000/3600) / (dt if dt>0 else 1)

        # route deviation random walk with gentle pull to 0
        route_offset = route_offset*0.97 + rng.normal(0, 0.02)
        route_dev_km = route_offset

        # jittery timestamps & occasional backward clock issues
        jitter = rng.normal(0, 3.0)  # ±3s jitter
        backstep = -rng.choice([0, 2, 0, 0]) if rng.random()<0.01 else 0  # rare non-monotonic
        ts += dt + jitter + backstep

        # battery drains slowly with noise
        battery = max(0.0, battery - rng.uniform(0.00005, 0.0002))
        # gnss mode flips sometimes (e.g., high-accuracy on)
        if rng.random() < 0.02:
            gnss = 1 - gnss

        yield {
            "temp": temp,
            "press": press,
            "ts": ts,
            "speed_kmh": speed,
            "accel_mps2": accel_mps2,
            "route_dev_km": route_dev_km,
            "battery_pct": battery,
            "gnss_hiacc": gnss
        }

# ---------------- Run demo ----------------
if __name__ == "__main__":
    pipe = StreamingPipeline(w=60, step=15, refit_every=10, contam=0.07, dt_nominal=60.0)
    for pkt in simulated_stream():
        out = pipe.update(pkt)
        if out:
            m = out["metrics"]
            print(
                f"[{out['action']}] score={out['anomaly_score']:.3f} "
                f"trust={m['trust_score']:.2f} "
                f"nis95={m['nis95_rate']:.2f} nis99={m['nis99_rate']:.2f} "
                f"tempSLA={m['temp_sla_violation']:.2%} "
                f"tjitter95={m['ts_jitter_sec']:.1f}s "
                f"miss={m['missing_frac']:.1%} "
                f"routeDev={m['route_corridor_dev_km']:.2f}km "
                f"batt={m['battery_pct']:.2%}"
            )
            # Example: print the full dict when you want it
            # import pprint; pprint.pprint(m)
