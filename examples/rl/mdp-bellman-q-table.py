import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """Custom GridWorld Environment that follows gymnasium interface"""
    
    def __init__(self, grid_size=5, gamma=0.9):
        super().__init__()
        self.grid_size = grid_size
        self.gamma = gamma
        
        # Define action and observation spaces (required by gymnasium)
        self.action_space = spaces.Discrete(4)  # UP, RIGHT, DOWN, LEFT
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # Goal states and rewards
        self.goal_states = {(0, 1): 10, (0, 3): 5}
        self.current_state = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Random starting position (not on goal states)
        while True:
            i = self.np_random.integers(0, self.grid_size)
            j = self.np_random.integers(0, self.grid_size)
            if (i, j) not in self.goal_states:
                break
        
        self.current_state = np.array([i, j])
        return self.current_state.copy(), {}
    
    def step(self, action):
        """Execute action and return (observation, reward, terminated, truncated, info)"""
        i, j = self.current_state
        
        # Move based on action
        if action == 0:  # UP
            new_i, new_j = max(0, i-1), j
        elif action == 1:  # RIGHT
            new_i, new_j = i, min(self.grid_size-1, j+1)
        elif action == 2:  # DOWN
            new_i, new_j = min(self.grid_size-1, i+1), j
        elif action == 3:  # LEFT
            new_i, new_j = i, max(0, j-1)
        
        self.current_state = np.array([new_i, new_j])
        
        # Calculate reward
        if (new_i, new_j) in self.goal_states:
            reward = self.goal_states[(new_i, new_j)]
            terminated = True
        else:
            reward = -0.1  # Small negative reward for each step
            terminated = False
        
        return self.current_state.copy(), reward, terminated, False, {}

class GymnasiumRLAgent:
    """RL algorithms adapted to work with Gymnasium environments"""
    
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.grid_size = env.grid_size
        
        # Value functions
        self.V = np.zeros((self.grid_size, self.grid_size))
        self.Q = np.zeros((self.grid_size, self.grid_size, env.action_space.n))
        self.policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
    
    def q_learning(self, episodes=1000, alpha=0.1, epsilon=0.1):
        """Q-Learning using gymnasium environment"""
        print(f"Running Q-Learning with Gymnasium for {episodes} episodes...")
        
        rewards_per_episode = []
        epsilon_decay = 0.995
        min_epsilon = 0.01
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                i, j = state
                
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[i, j, :])
                
                # Take action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_i, next_j = next_state
                
                # Q-Learning update
                if terminated:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.Q[next_i, next_j, :])
                
                self.Q[i, j, action] += alpha * (target - self.Q[i, j, action])
                
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
                    
                state = next_state
            
            rewards_per_episode.append(total_reward)
            
            # Decay epsilon
            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
            
            # Progress reporting
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"  Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}")
        
        # Extract value function and policy
        self.V = np.max(self.Q, axis=2)
        self._extract_policy_from_q()
        
        return rewards_per_episode
    
    def test_with_builtin_env(self, env_name="FrozenLake-v1"):
        """Example using built-in gymnasium environment"""
        print(f"\nTesting with built-in Gymnasium environment: {env_name}")
        
        # Create built-in environment
        builtin_env = gym.make(env_name, is_slippery=False)
        
        # Simple Q-learning on built-in environment
        q_table = np.zeros((builtin_env.observation_space.n, builtin_env.action_space.n))
        
        episodes = 1000
        alpha = 0.1
        epsilon = 0.1
        
        rewards = []
        
        for episode in range(episodes):
            state, _ = builtin_env.reset()
            total_reward = 0
            
            for _ in range(100):  # Max steps per episode
                if np.random.random() < epsilon:
                    action = builtin_env.action_space.sample()
                else:
                    action = np.argmax(q_table[state, :])
                
                next_state, reward, terminated, truncated, _ = builtin_env.step(action)
                
                # Q-learning update
                if terminated or truncated:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(q_table[next_state, :])
                
                q_table[state, action] += alpha * (target - q_table[state, action])
                
                total_reward += reward
                
                if terminated or truncated:
                    break
                    
                state = next_state
            
            rewards.append(total_reward)
            
            if episode % 200 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"  Episode {episode}: Avg Reward = {avg_reward:.3f}")
        
        builtin_env.close()
        return rewards
    
    def _extract_policy_from_q(self):
        """Extract policy from Q-values"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.policy[i, j] = np.argmax(self.Q[i, j, :])
    
    def visualize_results(self):
        """Visualize value function and policy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Value function
        sns.heatmap(self.V, annot=True, fmt='.2f', cmap='viridis', ax=ax1)
        ax1.set_title('Value Function (Gymnasium)')
        ax1.set_ylabel('Row')
        ax1.set_xlabel('Column')
        
        # Policy
        arrows = ['↑', '→', '↓', '←']
        im = ax2.imshow(self.V, cmap='viridis', alpha=0.3)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) in [(0, 1), (0, 3)]:  # Goal states
                    ax2.text(j, i, 'GOAL', ha='center', va='center', 
                           fontweight='bold', color='red')
                else:
                    ax2.text(j, i, arrows[self.policy[i, j]], 
                           ha='center', va='center', fontsize=16)
        
        ax2.set_xlim(-0.5, self.grid_size - 0.5)
        ax2.set_ylim(-0.5, self.grid_size - 0.5)
        ax2.set_title('Policy (Gymnasium)')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        
        plt.tight_layout()
        plt.show()

def test_gymnasium_integration():
    """Test the gymnasium integration"""
    print("=== Testing GridWorld with Gymnasium ===\n")
    
    # Create custom gymnasium environment
    env = GridWorldEnv(grid_size=5, gamma=0.9)
    
    # Test basic environment functionality
    print("Testing environment interface:")
    state, _ = env.reset()
    print(f"Initial state: {state}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, state={state}, reward={reward}")
        if terminated:
            break
    
    print("\n" + "="*50 + "\n")
    
    # Train agent
    agent = GymnasiumRLAgent(env)
    rewards = agent.q_learning(episodes=10000)
    
    # Visualize results
    agent.visualize_results()
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg)
    plt.title('Q-Learning with Gymnasium: Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.show()
    
    # Test with built-in environment
    agent.test_with_builtin_env("FrozenLake-v1")
    
    return env, agent

if __name__ == "__main__":
    # You would need to install gymnasium first:
    # pip install gymnasium
    
    env, agent = test_gymnasium_integration()