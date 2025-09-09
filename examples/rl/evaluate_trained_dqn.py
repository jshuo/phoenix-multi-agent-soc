"""
Inference script for the trained DQN model
------------------------------------------
Use this to make predictions on new device data in production.

python evaluate_trained_dqn.py \             
  --model sb3_offline_dqn/checkpoint.zip \
  --scaler sb3_offline_dqn/scaler.npz \                      
  --csv offpolicy_device_history.csv

"""

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from pathlib import Path
import torch as th  

class DeviceActionPredictor:
    """Wrapper class for making predictions with the trained DQN model"""
    
    FEATURES = [
        "nis99_rate","nis95_rate","temp_sla_violation","temp_jump_rate",
        "press_residual_proxy","pressure_jump_rate","route_corridor_dev_km",
        "speed_spike_rate","accel_spike_rate","ts_jitter_sec","non_monotonic_ts_rate",
        "missing_frac","battery_pct","cal_age_hours","gnss_hiacc_mode","trust_score"
    ]
    
    ACTION_NAMES = ["monitor", "escalate", "calibrate", "peer_check", "flag"]
    
    def __init__(self, model_path="sb3_offline_dqn/checkpoint.zip", 
                 scaler_path="sb3_offline_dqn/scaler.npz"):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained DQN model
            scaler_path: Path to the feature scaler
        """
        self.model = DQN.load(model_path)
        scaler_data = np.load(scaler_path)
        self.mu = scaler_data['mu']
        self.sig = scaler_data['sig']
        
    def preprocess_features(self, features):
        """
        Normalize features using training statistics
        
        Args:
            features: Dictionary or pandas Series with feature values
        
        Returns:
            Normalized feature array
        """
        # Convert to numpy array in correct order
        if isinstance(features, dict):
            feature_array = np.array([features[feat] for feat in self.FEATURES])
        else:  # pandas Series or array-like
            feature_array = np.array([features[feat] for feat in self.FEATURES])
        
        # Normalize
        normalized = (feature_array - self.mu) / self.sig
        return normalized.astype(np.float32)
    
    def predict_action(self, features, deterministic=True):
        """
        Predict the best action for given device features
        
        Args:
            features: Dictionary or pandas Series with feature values
            deterministic: Whether to use deterministic (greedy) policy
        
        Returns:
            Dictionary with action_id, action_name, and confidence
        """
        # Preprocess features
        obs = self.preprocess_features(features)
        
        # Get prediction using the model's predict method (handles tensor conversion)
        action_id, _ = self.model.predict(obs, deterministic=deterministic)
        action_name = self.ACTION_NAMES[action_id]
        
        # Get Q-values for confidence estimation
        # Convert to tensor for the model
        obs_tensor = th.as_tensor(obs).float().unsqueeze(0)  # Add batch dimension
        
        # Get Q-values from the model
        with th.no_grad():
            q_values = self.model.q_net(obs_tensor)
            if hasattr(q_values, 'cpu'):
                q_values = q_values.cpu().numpy()[0]
            else:
                q_values = np.array(q_values)[0]
        
        # Simple confidence measure: softmax of Q-values
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        probs = exp_q / np.sum(exp_q)
        confidence = probs[action_id]
        
        return {
            "action_id": int(action_id),
            "action_name": action_name,
            "confidence": float(confidence),
            "all_action_probs": {
                self.ACTION_NAMES[i]: float(probs[i]) for i in range(5)
            }
        }
    
    def predict_batch(self, features_list, deterministic=True):
        """
        Predict actions for a batch of feature sets
        
        Args:
            features_list: List of feature dictionaries/Series
            deterministic: Whether to use deterministic policy
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict_action(features, deterministic) 
                for features in features_list]

def example_usage():
    """Example of how to use the predictor"""
    
    # Initialize predictor
    predictor = DeviceActionPredictor()
    
    # Example device data
    device_data = {
        "nis99_rate": 0.95,
        "nis95_rate": 0.89,
        "temp_sla_violation": 0.02,
        "temp_jump_rate": 0.01,
        "press_residual_proxy": 1.2,
        "pressure_jump_rate": 0.003,
        "route_corridor_dev_km": 0.15,
        "speed_spike_rate": 0.008,
        "accel_spike_rate": 0.012,
        "ts_jitter_sec": 0.5,
        "non_monotonic_ts_rate": 0.001,
        "missing_frac": 0.05,
        "battery_pct": 0.75,
        "cal_age_hours": 168.0,
        "gnss_hiacc_mode": 0.0,
        "trust_score": 0.82
    }
    
    device_data_bad = {
    "nis99_rate": 0.98,
    "nis95_rate": 0.95,
    "temp_sla_violation": 0.95,
    "temp_jump_rate": 0.7,
    "press_residual_proxy": 25.0,
    "pressure_jump_rate": 0.6,
    "route_corridor_dev_km": 5.0,
    "speed_spike_rate": 0.6,
    "accel_spike_rate": 0.5,
    "ts_jitter_sec": 45.0,
    "non_monotonic_ts_rate": 0.4,
    "missing_frac": 0.5,
    "battery_pct": 0.05,
    "cal_age_hours": 1200.0,
    "gnss_hiacc_mode": 0.0,
    "trust_score": 0.1
}
    
    # Make prediction
    # result = predictor.predict_action(device_data)
    result = predictor.predict_action(device_data_bad)
    
    print("Prediction Results:")
    print(f"Recommended Action: {result['action_name']} (ID: {result['action_id']})")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nAll Action Probabilities:")
    for action, prob in result['all_action_probs'].items():
        print(f"  {action}: {prob:.3f}")
    
    return result

if __name__ == "__main__":
    # Run example
    example_usage()