"""
Inference script for a trained SAC model with discrete-action mapping
--------------------------------------------------------------------
Assumption: SAC was trained with a 1-D continuous action a in [0, 4].
At inference, we round/clip to map a -> {0,1,2,3,4} for:
    0: monitor, 1: escalate, 2: calibrate, 3: peer_check, 4: flag

Examples:
python evaluate_trained_sac.py \
  --model sb3_offline_sac/checkpoint.zip \
  --scaler sb3_offline_sac/scaler.npz

python evaluate_trained_sac.py \
  --model sb3_offline_sac/checkpoint.zip \
  --scaler sb3_offline_sac/scaler.npz \
  --csv offpolicy_device_history.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import SAC


class DeviceActionPredictorSAC:
    """Wrapper for making predictions with a trained SAC model (continuous->discrete mapping)."""

    FEATURES = [
        "nis99_rate", "nis95_rate", "temp_sla_violation", "temp_jump_rate",
        "press_residual_proxy", "pressure_jump_rate", "route_corridor_dev_km",
        "speed_spike_rate", "accel_spike_rate", "ts_jitter_sec", "non_monotonic_ts_rate",
        "missing_frac", "battery_pct", "cal_age_hours", "gnss_hiacc_mode", "trust_score"
    ]

    ACTION_NAMES = ["monitor", "escalate", "calibrate", "peer_check", "flag"]

    def __init__(self, model_path: str, scaler_path: str):
        """
        Args:
            model_path: path to trained SAC .zip
            scaler_path: npz with arrays 'mu' and 'sig' for feature normalization
        """
        self.model = SAC.load(model_path, device="cpu")
        scaler_data = np.load(scaler_path)
        self.mu = scaler_data["mu"].astype(np.float32)
        self.sig = scaler_data["sig"].astype(np.float32)

        # Avoid divide-by-zero if any std is 0
        self.sig = np.where(self.sig == 0.0, 1e-8, self.sig)

        # Prebuild the 5 discrete action bins as 1-D continuous actions in [0,4]
        self._discrete_bins = np.arange(5, dtype=np.float32).reshape(-1, 1)

    # ---------- feature utils ----------
    def preprocess_features(self, features) -> np.ndarray:
        """
        Accepts a dict mapping feature -> value, a pandas Series, or a numpy array
        (in the exact order of FEATURES). Returns a normalized 1D float32 array.
        """
        if isinstance(features, dict):
            arr = np.array([features[f] for f in self.FEATURES], dtype=np.float32)
        elif isinstance(features, pd.Series):
            arr = features[self.FEATURES].to_numpy(dtype=np.float32)
        else:
            # Assume it's already an array-like in the same order as FEATURES
            arr = np.asarray(features, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != len(self.FEATURES):
                raise ValueError(
                    f"Expected array of shape ({len(self.FEATURES)},), got {arr.shape}"
                )

        norm = (arr - self.mu) / self.sig
        return norm.astype(np.float32)

    # ---------- action mapping ----------
    @staticmethod
    def _to_scalar(x) -> float:
        """
        Robustly convert array-like or tensor-like to a Python float (first element).
        """
        if isinstance(x, th.Tensor):
            return float(x.detach().cpu().reshape(-1)[0].item())
        x_np = np.asarray(x)
        return float(x_np.reshape(-1)[0])

    @classmethod
    def continuous_to_discrete(cls, a_cont) -> int:
        """
        Map 1-D continuous action in [0,4] -> discrete id {0..4} by round+clip.
        Accepts array-like/tensor and extracts a scalar.
        """
        a_scalar = cls._to_scalar(a_cont)
        return int(np.clip(np.round(a_scalar), 0, 4))

    # ---------- Q-value evaluation for each discrete action ----------
    def q_values_over_bins(self, obs_np: np.ndarray) -> np.ndarray:
        """
        Evaluate twin critics on each discrete bin; take min(Q1, Q2) as conservative Q.
        Returns: np.array shape (5,)
        """
        self.model.policy.set_training_mode(False)
        obs_t = th.as_tensor(obs_np, dtype=th.float32).unsqueeze(0)  # (1, obs_dim)

        q_list = []
        with th.no_grad():
            for a_bin in self._discrete_bins:  # shape (1,)
                a_t = th.as_tensor(a_bin, dtype=th.float32).unsqueeze(0)  # (1, 1)
                # SB3 SAC critic exposes q1_forward and q2_forward
                q1 = self.model.critic.q1_forward(obs_t, a_t)
                q2 = self.model.critic.q2_forward(obs_t, a_t)
                q_min = th.minimum(q1, q2).cpu().numpy().squeeze().item()
                q_list.append(q_min)
        return np.array(q_list, dtype=np.float32)

    def predict_action(self, features, deterministic: bool = True):
        """
        Predict best action. For SAC, we sample/mean a continuous action and map to discrete.
        Confidence is derived by softmax over Q-values on the 5 discrete bins.
        """
        obs = self.preprocess_features(features)

        # SAC policy -> continuous action (shape (1,) or (1,1))
        a_cont, _ = self.model.predict(obs, deterministic=deterministic)
        a_id = self.continuous_to_discrete(a_cont)

        # Compute pseudo-probabilities via softmax over per-bin Q-values
        q_vals = self.q_values_over_bins(obs)
        # stable softmax
        exp_q = np.exp(q_vals - np.max(q_vals))
        probs = exp_q / np.sum(exp_q)

        return {
            "action_id": int(a_id),
            "action_name": self.ACTION_NAMES[a_id],
            "confidence": float(probs[a_id]),
            "all_action_probs": {self.ACTION_NAMES[i]: float(probs[i]) for i in range(5)},
        }

    def predict_batch(self, features_list, deterministic: bool = True):
        return [self.predict_action(f, deterministic) for f in features_list]


def run_single_example(predictor: DeviceActionPredictorSAC):
    # Example – very bad device
    device_data_bad = {
        "nis99_rate": 0.98, "nis95_rate": 0.95, "temp_sla_violation": 0.95, "temp_jump_rate": 0.7,
        "press_residual_proxy": 25.0, "pressure_jump_rate": 0.6, "route_corridor_dev_km": 5.0,
        "speed_spike_rate": 0.6, "accel_spike_rate": 0.5, "ts_jitter_sec": 45.0,
        "non_monotonic_ts_rate": 0.4, "missing_frac": 0.5, "battery_pct": 0.05,
        "cal_age_hours": 1200.0, "gnss_hiacc_mode": 0.0, "trust_score": 0.1
    }
    result = predictor.predict_action(device_data_bad)
    print("Prediction Results:")
    print(f"Recommended Action: {result['action_name']} (ID: {result['action_id']})")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nAll Action Probabilities:")
    for action, prob in result["all_action_probs"].items():
        print(f"  {action}: {prob:.3f}")


def run_csv(predictor: DeviceActionPredictorSAC, csv_path: Path, out_path: Path | None = None):
    df = pd.read_csv(csv_path)
    missing = [f for f in predictor.FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    results = []
    for _, row in df.iterrows():
        pred = predictor.predict_action(row)
        results.append({
            **{f: row[f] for f in predictor.FEATURES},
            "action_id": pred["action_id"],
            "action_name": pred["action_name"],
            "confidence": pred["confidence"],
            **{f"p_{k}": v for k, v in pred["all_action_probs"].items()},
        })

    out_df = pd.DataFrame(results)
    if out_path is None:
        out_path = csv_path.with_suffix(".predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to SAC .zip checkpoint")
    parser.add_argument("--scaler", required=True, help="Path to scaler .npz (mu/sig)")
    parser.add_argument("--csv", help="Optional CSV of feature rows to batch-predict")
    parser.add_argument("--out", help="Optional path for CSV predictions")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic SAC action instead of deterministic")
    args = parser.parse_args()

    predictor = DeviceActionPredictorSAC(model_path=args.model, scaler_path=args.scaler)

    if args.csv:
        out_path = Path(args.out) if args.out else None
        run_csv(predictor, Path(args.csv), out_path)
    else:
        run_single_example(predictor)


if __name__ == "__main__":
    main()
