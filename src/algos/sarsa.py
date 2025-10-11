
import numpy as np
from src.policies import EpsilonGreedySchedule, epsilon_greedy
from src.utils import discounted_returns, TrainingLog
from src.algos.q_learning import TabularQ
from tqdm import tqdm

def train_sarsa(env, alpha=0.5, gamma=0.99, episodes=500, max_steps=200, eps_schedule=None, log_discounted=True):
    if eps_schedule is None: eps_schedule = EpsilonGreedySchedule()
    Q_function = TabularQ(env.n_actions)
    rewards, disc_rewards, epsilons, t = [], [], [], 0

    for ep in tqdm(range(episodes)):
        obs = env.reset()
        eps = eps_schedule.value(t)
        a = epsilon_greedy(Q_function.values(obs), eps, env.n_actions)
        ep_rewards = []

        for step in range(max_steps):
            next_state, r, done, _ = env.step(a)
            ep_rewards.append(r); t += 1
            if not done:
                eps_next = eps_schedule.value(t)
                a_next = epsilon_greedy(Q_function.values(next_state), eps_next, env.n_actions)
                update_target = r + gamma * Q_function.values(next_state)[a_next]
            else:
                a_next = None; 
                update_target = r
            Qsa = Q_function.values(obs)
            Qsa[a] += alpha * (update_target - Qsa[a])
            obs, a = next_state, (a_next if a_next is not None else 0)
            if done: 
                break
        rewards.append(sum(ep_rewards))
        if log_discounted: disc_rewards.append(discounted_returns(ep_rewards, gamma))
        epsilons.append(eps_schedule.value(t-1))

    log = TrainingLog(rewards, disc_rewards if log_discounted else None, epsilons)
    return Q_function, log
