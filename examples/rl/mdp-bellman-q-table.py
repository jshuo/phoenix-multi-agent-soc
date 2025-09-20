import numpy as np
import matplotlib.pyplot as plt

# ---- GridWorld setup ----
GRID = 4
START = (3, 0)   # bottom-left
GOAL = (0, 3)    # top-right
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
A2DELTA = {
    0: (-1, 0),   # UP
    1: (0, 1),    # RIGHT
    2: (1, 0),    # DOWN
    3: (0, -1),   # LEFT
}

def step(state, action):
    """Deterministic move, stays in place if hitting wall."""
    dr, dc = A2DELTA[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID and 0 <= nc < GRID):
        nr, nc = r, c
    next_state = (nr, nc)
    reward = 10.0 if next_state == GOAL else 0.0
    done = (next_state == GOAL)
    return next_state, reward, done

def state_to_idx(s): return s[0] * GRID + s[1]

# ---- Q-learning hyperparameters ----
alpha = 0.3          # learning rate
gamma = 0.95         # discount factor
episodes = 200       # training episodes
max_steps = 50
eps_start, eps_end = 1.0, 0.05

def epsilon_at(ep):
    # linear decay
    return eps_end + (eps_start - eps_end) * max(0, (episodes - ep) / episodes)

# ---- Q-table ----
Q = np.zeros((GRID*GRID, len(ACTIONS)))
rng = np.random.default_rng(0)

# ---- Training ----
for ep in range(1, episodes+1):
    s = START
    eps = epsilon_at(ep)
    for t in range(max_steps):
        si = state_to_idx(s)
        # epsilon-greedy action
        if rng.random() < eps:
            a = rng.integers(0, len(ACTIONS))
        else:
            a = int(np.argmax(Q[si]))
        s_next, r, done = step(s, a)
        sni = state_to_idx(s_next)

        # --- Q-learning update rule ---
        td_target = r + (0 if done else gamma * np.max(Q[sni]))
        td_error  = td_target - Q[si, a]
        Q[si, a] += alpha * td_error

        s = s_next
        if done: break

    # print snapshots
    if ep in (1, 10, 50, 200):
        print(f"\nQ-table after episode {ep}:")
        for r in range(GRID):
            row_vals = []
            for c in range(GRID):
                idx = state_to_idx((r, c))
                row_vals.append(np.round(np.max(Q[idx]), 2))
            print(row_vals)

# ---- Plot greedy policy ----
fig, ax = plt.subplots(figsize=(5,5))
# grid lines
for x in range(GRID+1):
    ax.plot([x, x], [0, GRID], color='black')
for y in range(GRID+1):
    ax.plot([0, GRID], [y, y], color='black')

# draw arrows for best actions
for r in range(GRID):
    for c in range(GRID):
        if (r, c) == GOAL: continue
        si = state_to_idx((r, c))
        best_a = int(np.argmax(Q[si]))
        dr, dc = A2DELTA[best_a]
        x, y = c + 0.5, GRID - r - 0.5
        ax.annotate('', xy=(x + 0.25*dc, y - 0.25*dr),
                    xytext=(x - 0.25*dc, y + 0.25*dr),
                    arrowprops=dict(arrowstyle='->', lw=2))

# mark start & goal
ax.text(START[1]+0.5, GRID-START[0]-0.8, "S", ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(GOAL[1]+0.5, GRID-GOAL[0]-0.8, "G", ha='center', va='center', fontsize=14, fontweight='bold')

ax.set_xlim(0, GRID); ax.set_ylim(0, GRID); ax.set_aspect('equal')
ax.set_title("Learned Policy Arrows (Q-learning)")
plt.show()
