
import numpy as np
from src.policies import EpsilonGreedySchedule, epsilon_greedy
from src.utils import discounted_returns, TrainingLog
from tqdm import tqdm

class TabularQ:
    def __init__(self, n_actions):
        self.n_actions = n_actions; self.q = {}
    def _key(self, obs):
        planes = obs
        pac = tuple(np.argwhere(planes[4]==1.0)[0])
        ghosts = tuple(map(tuple, np.argwhere(planes[3]==1.0)))
        dot = tuple(np.argwhere(planes[2]==1.0)[0])
        return (pac, ghosts, dot)
    def values(self, obs):
        k = self._key(obs)
        if k not in self.q: self.q[k] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q[k]

def train_q_learning(env, alpha=0.5, gamma=0.99, episodes=500, max_steps=100, eps_schedule=None, log_discounted=True):
    if eps_schedule is None: eps_schedule = EpsilonGreedySchedule()
    Q_function = TabularQ(env.n_actions)
    rewards, disc_rewards, epsilons, t = [], [], [], 0
    for ep in tqdm(range(episodes)):
        obs = env.reset(); ep_rewards = []
        for step in range(max_steps):
            eps = eps_schedule.value(t)
            a = epsilon_greedy(Q_function.values(obs), eps, env.n_actions)

            next_state, r, done, _ = env.step(a)
            Qsa = Q_function.values(obs)
            max_next = np.max(Q_function.values(next_state))

            update_target = r + gamma * max_next * (0.0 if done else 1.0)
            Qsa[a] += alpha * (update_target - Qsa[a])
            obs = next_state
            ep_rewards.append(r); t += 1
            if done: 
                break
        rewards.append(sum(ep_rewards))
        if log_discounted: disc_rewards.append(discounted_returns(ep_rewards, gamma))
        epsilons.append(eps_schedule.value(t-1))
    log = TrainingLog(rewards, disc_rewards if log_discounted else None, epsilons)
    return Q_function, log

def greedy_action_from_q(Q_function):
    def _f(obs): return int(np.argmax(Q_function.values(obs)))
    return _f

def evaluate(env, policy, episodes=200, max_steps=200):
    wins, totals = 0, []
    for _ in range(episodes):
        obs = env.reset(); total = 0.0; done = False
        for _ in range(max_steps):
            a = policy(obs); obs, r, done, _ = env.step(a)
            total += r
            if done: 
                break
        totals.append(total)
        if total >= 0 and env.pacman == env.dot_pos: wins += 1
    return {"win_rate": wins / max(1, episodes), "mean_reward": float(np.mean(totals)) if totals else 0.0}
