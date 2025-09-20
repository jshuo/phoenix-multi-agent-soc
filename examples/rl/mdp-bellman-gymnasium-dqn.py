import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import torch

class GridWorldEnv(gym.Env):
    """Custom GridWorld Environment that follows gymnasium interface"""
    
    def __init__(self, grid_size=4):
        super().__init__()
        self.grid_size = grid_size
        
        # Define action and observation spaces (required by gymnasium)
        self.action_space = spaces.Discrete(4)  # UP, RIGHT, DOWN, LEFT
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        
        # GridWorld setup - dynamically set start and goal based on grid size
        self.start = (grid_size - 1, 0)          # bottom-left
        self.goal = (0, grid_size - 1)           # top-right
        self.obstacle = (2, 3)
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.a2delta = {
            0: (-1, 0),   # UP
            1: (0, 1),    # RIGHT
            2: (1, 0),    # DOWN
            3: (0, -1),   # LEFT
        }
        
        self.current_state = None
        
    def state_to_idx(self, state):
        """Convert (row, col) state to index"""
        return state[0] * self.grid_size + state[1]
    
    def idx_to_state(self, idx):
        """Convert index to (row, col) state"""
        return (idx // self.grid_size, idx % self.grid_size)
    
    def state_to_coords(self, state):
        """Convert state tuple to coordinate array for neural network"""
        return np.array([state[0], state[1]], dtype=np.int32)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_state = self.start
        return self.state_to_coords(self.current_state), {}
    
    def step(self, action):
        """Execute action and return (observation, reward, terminated, truncated, info)"""
        dr, dc = self.a2delta[int(action)]
        r, c = self.current_state
        nr, nc = r + dr, c + dc
        
        # Check bounds AND obstacle before moving
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size) or (nr, nc) == self.obstacle:
            nr, nc = r, c  # Stay in current position - don't move into wall or obstacle
            
        self.current_state = (nr, nc)
        
        # Calculate reward
        if self.current_state == self.goal:
            reward = 10.0
        elif (r + dr, c + dc) == self.obstacle:  # Tried to move into obstacle
            reward = -5.0  # Penalty for attempting to move into obstacle
        else:
            reward = -0.1  # Small negative reward for each step
    
        terminated = (self.current_state == self.goal)
    
        return self.state_to_coords(self.current_state), reward, terminated, False, {}


class TrainingCallback(BaseCallback):
    """Callback to track training progress"""
    
    def __init__(self, env, log_freq=1000):
        super().__init__()
        self.env = env
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode info
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
        # Print progress
        if self.num_timesteps % self.log_freq == 0:
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Timesteps: {self.num_timesteps}, "
                      f"Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}")
        
        return True

def extract_q_values_from_dqn(model, env):
    """Extract Q-values from trained DQN model for visualization"""
    Q = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    
    # Get Q-values for each state
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if (r, c) == env.obstacle:
                # Obstacle states should have very negative Q-values
                Q[r, c] = np.full(env.action_space.n, -100.0)
                continue
                
            state = np.array([r, c], dtype=np.int32)
            # Get Q-values from the model
            obs_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model.q_net(obs_tensor).cpu().numpy()[0]
            Q[r, c] = q_values
    
    return Q

def train_dqn_stable_baselines3(grid_size=4, timesteps=10000):
    """Train DQN using Stable-Baselines3 with configurable grid size"""
    
    # Create environment
    env = GridWorldEnv(grid_size=grid_size)
    
    # Check environment compatibility
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment check passed!")
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        learning_rate=0.001,
        buffer_size=1000,
        learning_starts=100,
        batch_size=32,
        tau=1.0,  # Hard update
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=100,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log=None,
        policy_kwargs=dict(net_arch=[64, 64]),  # Small network for simple problem
        verbose=1,
        seed=0
    )
    
    # Create callback
    callback = TrainingCallback(env, log_freq=1000)
    
    print(f"Training DQN agent on {grid_size}x{grid_size} grid...")
    print(f"Total timesteps: {timesteps}")
    
    # Train the model
    model.learn(total_timesteps=timesteps, callback=callback)
    
    return env, model, callback

def plot_dqn_policy(env, model):
    """Plot greedy policy using trained DQN model"""
    # Scale figure size based on grid size
    fig_size = min(10, max(5, env.grid_size))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_size*2, fig_size))
    
    # Extract Q-values from DQN
    Q = extract_q_values_from_dqn(model, env)
    
    # Plot 1: Policy arrows
    ax1.set_title(f"Learned Policy (DQN {env.grid_size}x{env.grid_size} Grid)")
    
    # Grid lines
    for x in range(env.grid_size + 1):
        ax1.plot([x, x], [0, env.grid_size], color='black')
    for y in range(env.grid_size + 1):
        ax1.plot([0, env.grid_size], [y, y], color='black')
    
     # Draw arrows for best actions
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if (r, c) == env.goal or (r, c) == env.obstacle:
                continue
                
            best_action = int(np.argmax(Q[r, c]))
            dr, dc = env.a2delta[best_action]
            
            x, y = c + 0.5, env.grid_size - r - 0.5
            
            # Scale arrow size based on grid size
            arrow_scale = min(0.3, max(0.1, 0.8 / env.grid_size))
            ax1.annotate('', xy=(x + arrow_scale*dc, y - arrow_scale*dr),
                        xytext=(x - arrow_scale*dc, y + arrow_scale*dr),
                        arrowprops=dict(arrowstyle='->', lw=2))
    
        # Mark start, goal, and obstacle
        font_size = max(10, min(16, 100 // env.grid_size))
        start_x, start_y = env.start[1] + 0.5, env.grid_size - env.start[0] - 0.8
        goal_x, goal_y = env.goal[1] + 0.5, env.grid_size - env.goal[0] - 0.8
        obstacle_x, obstacle_y = env.obstacle[1] + 0.5, env.grid_size - env.obstacle[0] - 0.8
        
        ax1.text(start_x, start_y, "S", ha='center', va='center', 
                fontsize=font_size, fontweight='bold', color='blue')
        ax1.text(goal_x, goal_y, "G", ha='center', va='center', 
                fontsize=font_size, fontweight='bold', color='red')
        ax1.text(obstacle_x, obstacle_y, "X", ha='center', va='center', 
                fontsize=font_size, fontweight='bold', color='black')
    
    ax1.set_xlim(0, env.grid_size)
    ax1.set_ylim(0, env.grid_size)
    ax1.set_aspect('equal')
    
    # Plot 2: Value heatmap
    max_q_values = np.max(Q, axis=2)
    im = ax2.imshow(max_q_values, cmap='viridis', interpolation='nearest')
    ax2.set_title(f"State Values (Max Q-values)")
    
    # Add text annotations
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            ax2.text(c, r, f'{max_q_values[r, c]:.1f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    ax2.set_xticks(range(env.grid_size))
    ax2.set_yticks(range(env.grid_size))
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def test_trained_agent(env, model, episodes=5):
    """Test the trained DQN agent"""
    print(f"\n=== Testing Trained DQN Agent ===")
    
    total_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Start: {env.idx_to_state(env.state_to_idx(env.current_state))}")
        
        for step in range(50):  # Max steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            current_pos = env.idx_to_state(env.state_to_idx(env.current_state))
            print(f"  Step {step}: action={action}({env.actions[action]}), "
                  f"pos={current_pos}, reward={reward:.2f}")
            
            if terminated or truncated:
                print(f"  Episode finished! Total reward: {total_reward:.2f}, Steps: {steps}")
                break
        
        total_rewards.append(total_reward)
    
    print(f"\nAverage reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    return total_rewards

def run_dqn_experiment(grid_size=4, timesteps=10000):
    """Run complete DQN experiment for a given grid size"""
    print(f"\n{'='*20} DQN GRID SIZE {grid_size}x{grid_size} {'='*20}")
    
    # Train the agent
    env, model, callback = train_dqn_stable_baselines3(grid_size, timesteps)
    
    # Test the trained agent
    test_rewards = test_trained_agent(env, model)
    
    # Plot the learned policy and values
    plot_dqn_policy(env, model)
    
    # Close environment
    env.close()
    
    return env, model, test_rewards

if __name__ == "__main__":
    # Test different grid sizes
    grid_sizes = [5]
    
    # Run experiments for different grid sizes
    for grid_size in grid_sizes:
        # Scale timesteps based on grid complexity
        timesteps = max(20000, grid_size * grid_size * 200)
        run_dqn_experiment(grid_size, timesteps)
        print("\n" + "="*80 + "\n")