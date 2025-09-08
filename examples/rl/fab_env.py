import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FabDispatchEnv(gym.Env):
    """
    Simple fab toy model:
      - One global queue with incoming jobs.
      - Two parallel machines; action selects which machine gets the next job (if idle).
      - Each job has a processing time and due_urgency; waiting increases lateness and defect risk.
      - Reward balances throughput (+1 per completed job) against lateness/WIP/defects penalties.
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 max_queue=10,
                 job_arrival_p=0.5,
                 pt_low_high=(3, 8),     # processing time range
                 defect_wait_thresh=6,   # waiting beyond this increases defect chance
                 seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.max_queue = max_queue
        self.job_arrival_p = job_arrival_p
        self.pt_low, self.pt_high = pt_low_high
        self.defect_wait_thresh = defect_wait_thresh

        # Observation: [queue_len, next_job_wait, m1_busy_remaining, m2_busy_remaining]
        # (You can extend with averages, due dates, etc.)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([max_queue, 100, 100, 100], dtype=np.float32),
            dtype=np.float32,
        )
        # Actions: 0 = assign to M1, 1 = assign to M2, 2 = no-op (if both busy or queue empty)
        self.action_space = spaces.Discrete(3)

        self.reset_state()

        # metrics
        self.total_completed = 0
        self.total_defects = 0
        self.total_lateness = 0.0
        self.t = 0

    def reset_state(self):
        self.queue_waits = []       # waiting time per queued job
        self.m_busy = [0, 0]        # remaining busy time for machine 0/1

    def _spawn_job(self):
        # with some probability, a job arrives; we track its wait time (starts at 0)
        if len(self.queue_waits) < self.max_queue and self.rng.random() < self.job_arrival_p:
            self.queue_waits.append(0)

    def _assign_if_possible(self, m_idx):
        if m_idx not in (0, 1):
            return False
        if self.m_busy[m_idx] == 0 and len(self.queue_waits) > 0:
            # pop the oldest job (FIFO) — you can change to SPT/EDD features later
            wait = self.queue_waits.pop(0)
            pt = self.rng.integers(self.pt_low, self.pt_high + 1)
            # machine becomes busy
            self.m_busy[m_idx] = int(pt)
            # lateness proxy: if wait > pt_high assume urgency missed proportionally
            lateness = max(0, wait - self.pt_high/2)
            self.total_lateness += lateness
            return True
        return False

    def _advance_time(self):
        # advance one time unit
        done_jobs = 0
        defects = 0
        # decrement busy timers; completed jobs get counted and may be defective if they waited too long before start
        for i in range(2):
            if self.m_busy[i] > 0:
                self.m_busy[i] -= 1
                if self.m_busy[i] == 0:
                    done_jobs += 1
        # waiting queue ages; longer waits → higher defect probability upon completion (proxy via thresholding)
        # we approximate defect accrual by sampling when jobs finish next step; here, apply a mild running penalty
        # (simplified: count a defect for each completed job if average wait exceeded threshold)
        avg_wait = np.mean(self.queue_waits) if self.queue_waits else 0
        if done_jobs > 0 and avg_wait > self.defect_wait_thresh:
            defects = done_jobs  # pessimistic proxy
        # age all queued jobs
        self.queue_waits = [w + 1 for w in self.queue_waits]
        return done_jobs, defects, avg_wait

    def _obs(self):
        q_len = len(self.queue_waits)
        next_wait = self.queue_waits[0] if q_len else 0
        return np.array([q_len, next_wait, self.m_busy[0], self.m_busy[1]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state()
        self.total_completed = 0
        self.total_defects = 0
        self.total_lateness = 0.0
        self.t = 0
        # warm up with a few arrivals
        for _ in range(3):
            self._spawn_job()
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        # spawn potential new job
        self._spawn_job()

        # apply action: try to assign
        assigned = False
        if action in (0, 1):
            assigned = self._assign_if_possible(action)

        # advance time
        done_jobs, defects, avg_wait = self._advance_time()

        # reward shaping
        throughput_reward = +1.0 * done_jobs
        wip_penalty = -0.05 * len(self.queue_waits)
        wait_penalty = -0.02 * avg_wait
        defect_penalty = -0.5 * defects
        reward = throughput_reward + wip_penalty + wait_penalty + defect_penalty

        self.total_completed += done_jobs
        self.total_defects += defects

        # episode termination: fixed horizon
        terminated = (self.t >= 200)
        truncated = False

        info = {
            "completed": done_jobs,
            "defects": defects,
            "avg_wait": float(avg_wait),
            "assigned": assigned,
            "wip": len(self.queue_waits),
        }
        return self._obs(), reward, terminated, truncated, info
