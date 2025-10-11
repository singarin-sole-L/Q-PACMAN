
import os
import numpy as np
import matplotlib.pyplot as plt

class TrainingLog:
    def __init__(self, episode_rewards, discounted_episode_rewards=None, epsilons=None):
        self.episode_rewards = episode_rewards
        self.discounted_episode_rewards = discounted_episode_rewards
        self.epsilons = epsilons

def discounted_returns(rewards, gamma):
    s, g = 0.0, 1.0
    for r in rewards:
        s += g * r; g *= gamma
    return s

def _moving_average(x, window):
    if window is None or window <= 1:
        return np.asarray(x)
    w = int(window)
    y = np.convolve(x, np.ones(w, dtype=float)/w, mode='valid')
    pad_left = w - 1
    if pad_left > 0:
        y = np.concatenate([np.full(pad_left, y[0]), y])
    return y

def plot_training_curves(
    log,
    title="",
    out_path=None,
    show_epsilon=True,
    smooth_window=25,     
    epsilon_style="-"    
):
    
    smooth_rewards = _moving_average(log.episode_rewards, smooth_window)
    smooth_episodes = np.arange(len(smooth_rewards))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(smooth_episodes, smooth_rewards, label="Reward per episode")

    if getattr(log, "discounted_episode_rewards", None) is not None:
        disc = _moving_average(log.discounted_episode_rewards, smooth_window)
        ax.plot(smooth_episodes, disc, label="Discounted reward per episode")

    if show_epsilon and getattr(log, "epsilons", None) is not None:
        eps = _moving_average(log.epsilons, smooth_window)
        ax2 = ax.twinx()
        ax2.plot(smooth_episodes, eps, linestyle=epsilon_style, label="epsilon", alpha=0.8)
        ax2.set_ylabel("epsilon")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Total")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=160)
    plt.show()

def plot_overlay_rewards(
    logs_dict,
    title="",
    out_path=None,
    smooth_window=25,       
):
    fig, ax = plt.subplots(figsize=(9,5))

    for label, log in logs_dict.items():
        smooth_rewards = _moving_average(log.episode_rewards, smooth_window)
        smooth_episodes = np.arange(len(smooth_rewards))
        ax.plot(smooth_episodes, smooth_rewards, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward per episode")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=160)
    plt.show()

def show_metrics(metrics, title="Comparison"):
    print(f"\n=== {title} ===")
    for m in metrics.keys():
        wr = metrics[m]['win_rate'] * 100
        mr = metrics[m]['mean_reward']
        print(f"{m}: Winrate: {wr:.4f}% | Mean reward: {mr:.2f}")
