
import numpy as np
import os
import time
from src.utils import plot_training_curves, plot_overlay_rewards, show_metrics
from src.policies import EpsilonGreedySchedule
from src.algos.q_learning import train_q_learning, greedy_action_from_q
from src.algos.sarsa import train_sarsa
from src.algos.dqn import train_dqn

def eval_policy(env, policy, episodes=200, max_steps=100):
    wins, totals = 0, []
    for _ in range(episodes):
        obs = env.reset(); total = 0.0; done = False
        for _ in range(max_steps):
            a = policy(obs); obs, r, done, _ = env.step(a); total += r
            if done: 
                break
        totals.append(total)
        if env.pacman == env.dot_pos: wins += 1
    return {"win_rate": wins / max(1, episodes), "mean_reward": float(np.mean(totals)) if totals else 0.0}

def train_all(env, out_dir, episodes=300, max_steps=200, gamma=0.99, eps_mode="exp", episodes_for_evaluate=100, alpha=0.5):
    sched = lambda: EpsilonGreedySchedule(start=1.0, end=0.06, decay_steps=episodes*max_steps//2, mode=eps_mode)
    t0 = time.time()
    qfunc, qlog = train_q_learning(
    env, alpha=alpha, gamma=gamma, episodes=episodes, max_steps=max_steps,
    eps_schedule=sched(), log_discounted=False
    )
    train_q_time = time.time() - t0

    t1 = time.time()
    q_policy = greedy_action_from_q(qfunc)
    qmetrics = eval_policy(env, q_policy, episodes=episodes_for_evaluate)
    eval_q_time = (time.time() - t1)/episodes_for_evaluate
    print(f"Q-learning: training time = {train_q_time:.2f}s | evaluation time = {eval_q_time:.2f}s")

    t0 = time.time()
    sfunc, slog = train_sarsa(
    env, alpha=alpha, gamma=gamma, episodes=episodes, max_steps=max_steps,
    eps_schedule=sched(), log_discounted=False
    )
    train_s_time = time.time() - t0

    t1 = time.time()
    s_policy = greedy_action_from_q(sfunc)
    smetrics = eval_policy(env, s_policy, episodes=episodes_for_evaluate)
    eval_s_time = (time.time() - t1)/episodes_for_evaluate
    print(f"SARSA: training time = {train_s_time:.2f}s | evaluation time = {eval_s_time:.2f}s")

    t0 = time.time()
    d_policy, dlog = train_dqn(
    env, gamma=gamma, lr=1e-3, batch_size=64, buffer_size=10_000, target_update=500,
    episodes=episodes, max_steps=max_steps, eps_schedule=sched(), log_discounted=False, device="cpu"
    )
    train_d_time = time.time() - t0

    t1 = time.time()
    dmetrics = eval_policy(env, d_policy, episodes=episodes_for_evaluate)
    eval_d_time = (time.time() - t1)/episodes_for_evaluate
    print(f"DQN: training time = {train_d_time:.2f}s | evaluation time = {eval_d_time:.2f}s")


    os.makedirs(out_dir, exist_ok=True)
    plot_training_curves(qlog, title="Q-learning", out_path=os.path.join(out_dir, "q_learning_training.png"), show_epsilon=True)
    plot_training_curves(slog, title="SARSA", out_path=os.path.join(out_dir, "sarsa_training.png"), show_epsilon=True)
    plot_training_curves(dlog, title="DQN", out_path=os.path.join(out_dir, "dqn_training.png"), show_epsilon=True)
    show_metrics({"Q-learning": qmetrics, "SARSA": smetrics, "DQN": dmetrics}, title="Method comparison")
    plot_overlay_rewards({"Q-learning": qlog, "SARSA": slog, "DQN": dlog}, title="Training rewards overlay", out_path=os.path.join(out_dir, "overlay_rewards.png"))
    
    return {"Q-learning": (q_policy, qlog, qmetrics), "SARSA": (s_policy, slog, smetrics), "DQN": (d_policy, dlog, dmetrics)}

def optimal_path_render(env, policy, out_path):
    path = []
    obs = env.reset()
    path.append(env.pacman)
    done = False
    steps = 0

    while not done and steps < 200:
        a = policy(obs)
        obs, r, done, _ = env.step(a); path.append(env.pacman)
        steps += 1
    env.render(show=True, borders=True, path=path, save_path=out_path)


def sweep_rewards_q_sarsa(env, param, values, base, episodes=200, max_steps=150, title_prefix="Q SARSA sweep", out_path=None):
    logs_q = {}
    logs_s = {}
    fixed_text = ", ".join([f"{k}={v}" for k, v in base.items() if k != param and k != "eps_schedule"])
    title_q = f"{title_prefix} Q-learning ({fixed_text})"
    title_s = f"{title_prefix} SARSA ({fixed_text})"
    for v in values:
        cfg = dict(base)
        cfg[param] = v
        eps_schedule = cfg.get("eps_schedule", EpsilonGreedySchedule())
        alpha = cfg.get("alpha", 0.5)
        gamma = cfg.get("gamma", 0.99)
        qfunc, qlog = train_q_learning(env, alpha=alpha, gamma=gamma, episodes=episodes, max_steps=max_steps, eps_schedule=eps_schedule, log_discounted=True)
        sfunc, slog = train_sarsa(env, alpha=alpha, gamma=gamma, episodes=episodes, max_steps=max_steps, eps_schedule=eps_schedule, log_discounted=True)
        label = f"{param}={v}"
        logs_q[label] = qlog
        logs_s[label] = slog
    plot_overlay_rewards(logs_q, title=title_q, out_path=out_path if out_path is None else out_path.replace(".png", "_q.png"))
    plot_overlay_rewards(logs_s, title=title_s, out_path=out_path if out_path is None else out_path.replace(".png", "_s.png"))
    return logs_q, logs_s

def sweep_rewards_dqn(env, param, values, base, episodes=200, max_steps=150, title="DQN sweep", out_path=None):
    logs = {}
    fixed_text = ", ".join([f"{k}={v}" for k, v in base.items() if k != param and k != "eps_schedule"])
    title_full = f"{title} ({fixed_text})"
    for v in values:
        cfg = dict(base)
        cfg[param] = v
        eps_schedule = cfg.get("eps_schedule")
        gamma = cfg.get("gamma", 0.99)
        lr = cfg.get("lr", 1e-3)
        batch_size = cfg.get("batch_size", 64)
        buffer_size = cfg.get("buffer_size", 10_000)
        target_update = cfg.get("target_update", 500)
        if param == "gamma": gamma = v
        if param == "lr": lr = v
        policy, log = train_dqn(env, gamma=gamma, lr=lr, batch_size=batch_size, buffer_size=buffer_size, target_update=target_update,
                                episodes=episodes, max_steps=max_steps, eps_schedule=eps_schedule, log_discounted=True, device="cpu")
        label = f"{param}={v}"
        logs[label] = log
    plot_overlay_rewards(logs, title=title_full, out_path=out_path)
    return logs

