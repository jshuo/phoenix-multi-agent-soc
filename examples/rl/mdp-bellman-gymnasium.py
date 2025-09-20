import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """Custom GridWorld Environment that follows gymnasium interface"""
    
    def __init__(self, grid_size=4):
        super().__init__()
        self.grid_size = grid_size
        
        # Define action and observation spaces (required by gymnasium)
        self.action_space = spaces.Discrete(4)  # UP, RIGHT, DOWN, LEFT
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        
        # GridWorld setup - dynamically set start and goal based on grid size
        self.start = (grid_size - 1, 0)          # bottom-left
        self.goal = (0, grid_size - 1)           # top-right
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
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_state = self.start
        return self.state_to_idx(self.current_state), {}
    
    def step(self, action):
        """Execute action and return (observation, reward, terminated, truncated, info)"""
        dr, dc = self.a2delta[action]
        r, c = self.current_state
        nr, nc = r + dr, c + dc
        
        # Stay in place if hitting wall
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            nr, nc = r, c
            
        self.current_state = (nr, nc)
        
        # Calculate reward
        reward = 10.0 if self.current_state == self.goal else 0.0
        terminated = (self.current_state == self.goal)
        
        return self.state_to_idx(self.current_state), reward, terminated, False, {}

def train_q_learning_gymnasium(grid_size=4, episodes=200):
    """Train Q-learning using Gymnasium environment with configurable grid size"""
    
    # Create environment
    env = GridWorldEnv(grid_size=grid_size)
    
    # Q-learning hyperparameters
    alpha = 0.3          # learning rate
    gamma = 0.95         # discount factor
    max_steps = grid_size * grid_size * 2  # Scale max steps with grid size
    eps_start, eps_end = 1.0, 0.05
    
    def epsilon_at(ep):
        # linear decay
        return eps_end + (eps_start - eps_end) * max(0, (episodes - ep) / episodes)
    
    # Q-table
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(0)
    
    # Determine snapshot episodes based on total episodes
    snapshot_episodes = [1, max(1, episodes // 20), max(1, episodes // 4), episodes]
    snapshot_episodes = sorted(list(set(snapshot_episodes)))  # Remove duplicates and sort
    
    # Training
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        eps = epsilon_at(ep)
        
        for t in range(max_steps):
            # Epsilon-greedy action selection
            if rng.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Q-learning update rule
            if terminated:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state])
            
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Print snapshots
        if ep in snapshot_episodes:
            print(f"\nQ-table after episode {ep} (Grid size: {grid_size}x{grid_size}):")
            for r in range(env.grid_size):
                row_vals = []
                for c in range(env.grid_size):
                    idx = env.state_to_idx((r, c))
                    row_vals.append(np.round(np.max(Q[idx]), 2))
                print(row_vals)
    
    return env, Q

def plot_policy_gymnasium(env, Q):
    """Plot greedy policy using gymnasium environment"""
    # Scale figure size based on grid size
    fig_size = min(10, max(5, env.grid_size))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Grid lines
    for x in range(env.grid_size + 1):
        ax.plot([x, x], [0, env.grid_size], color='black')
    for y in range(env.grid_size + 1):
        ax.plot([0, env.grid_size], [y, y], color='black')
    
    # Draw arrows for best actions
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if (r, c) == env.goal:
                continue
                
            state_idx = env.state_to_idx((r, c))
            best_action = int(np.argmax(Q[state_idx]))
            dr, dc = env.a2delta[best_action]
            
            x, y = c + 0.5, env.grid_size - r - 0.5
            
            # Scale arrow size based on grid size
            arrow_scale = min(0.3, max(0.1, 0.8 / env.grid_size))
            ax.annotate('', xy=(x + arrow_scale*dc, y - arrow_scale*dr),
                       xytext=(x - arrow_scale*dc, y + arrow_scale*dr),
                       arrowprops=dict(arrowstyle='->', lw=2))
    
    # Mark start & goal with scaled font
    font_size = max(10, min(16, 100 // env.grid_size))
    start_x, start_y = env.start[1] + 0.5, env.grid_size - env.start[0] - 0.8
    goal_x, goal_y = env.goal[1] + 0.5, env.grid_size - env.goal[0] - 0.8
    
    ax.text(start_x, start_y, "S", ha='center', va='center', 
           fontsize=font_size, fontweight='bold', color='blue')
    ax.text(goal_x, goal_y, "G", ha='center', va='center', 
           fontsize=font_size, fontweight='bold', color='red')
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title(f"Learned Policy Arrows (Q-learning {env.grid_size}x{env.grid_size} Grid)", 
                fontsize=font_size)
    plt.tight_layout()
    plt.show()

def test_environment(grid_size=4):
    """Test the gymnasium environment with configurable grid size"""
    print(f"=== Testing Gymnasium GridWorld Environment ({grid_size}x{grid_size}) ===\n")
    
    env = GridWorldEnv(grid_size=grid_size)
    
    # Test environment interface
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Start state: {env.start}")
    print(f"Goal state: {env.goal}")
    
    # Test a few episodes
    for episode in range(2):
        print(f"\nEpisode {episode + 1}:")
        state, _ = env.reset()
        print(f"Initial state index: {state} -> {env.idx_to_state(state)}")
        
        for step in range(min(15, grid_size * 2)):  # Limit steps for large grids
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            print(f"  Step {step}: action={action}({env.actions[action]}), "
                  f"next_state={next_state}({env.idx_to_state(next_state)}), "
                  f"reward={reward}")
            
            if terminated:
                print("  Episode terminated - reached goal!")
                break
    
    env.close()

def run_experiment(grid_size, episodes=200):
    """Run complete experiment for a given grid size"""
    print(f"\n{'='*20} GRID SIZE {grid_size}x{grid_size} {'='*20}")
    
    # Test the environment
    test_environment(grid_size)
    
    print(f"\n{'-'*60}")
    
    # Train the agent
    print(f"Training Q-learning agent on {grid_size}x{grid_size} grid...")
    env, Q = train_q_learning_gymnasium(grid_size, episodes)
    
    # Plot the learned policy
    plot_policy_gymnasium(env, Q)
    
    # Close environment
    env.close()
    
    return env, Q

if __name__ == "__main__":
    # Test different grid sizes
    grid_sizes = [4]
    
    # Run experiments for different grid sizes
    for grid_size in grid_sizes:
        # Scale episodes based on grid complexity
        episodes = max(100, grid_size * grid_size * 5)
        run_experiment(grid_size, episodes)
        print("\n" + "="*80 + "\n")