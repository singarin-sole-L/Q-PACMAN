
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from src.policies import EpsilonGreedySchedule, epsilon_greedy
from src.utils import TrainingLog, discounted_returns
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, size):
        self.size = size; self.buf = []; self.idx = 0
    def add(self, tr):
        if len(self.buf) < self.size: self.buf.append(tr)
        else: self.buf[self.idx] = tr
        self.idx = (self.idx + 1) % self.size
    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.buf), size=batch_size)
        return [self.buf[i] for i in idxs]
    def __len__(self): return len(self.buf)

def train_dqn(env, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=10_000, target_update=500,
              episodes=500, max_steps=200, eps_schedule=None, log_discounted=True, device="cpu"):
    if eps_schedule is None: eps_schedule = EpsilonGreedySchedule()

    in_channels = env._obs().shape[0]
    n_actions = env.n_actions
    q = DQN(in_channels, n_actions).to(device)
    q_target = DQN(in_channels, n_actions).to(device)
    
    q_target.load_state_dict(q.state_dict())
    
    opt = optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size)
    rewards, disc_rewards, epsilons, t = [], [], [], 0

    for ep in tqdm(range(episodes)):
        obs = env.reset(); ep_rewards = []
        for step in range(max_steps):
            eps = eps_schedule.value(t)
            with torch.no_grad(): qvals = q(torch.from_numpy(obs).unsqueeze(0).to(device)).cpu().numpy()[0]
            a = epsilon_greedy(qvals, eps, n_actions)
            next_state, r, done, _ = env.step(a)
            buf.add((obs, a, r, next_state, done)); obs = next_state; ep_rewards.append(r)

            if len(buf) >= batch_size:
                ob, ac, re, nx, dn = zip(*buf.sample(batch_size))
                ob = torch.from_numpy(np.stack(ob)).to(device); nx = torch.from_numpy(np.stack(nx)).to(device)
                ac = torch.tensor(ac, dtype=torch.long, device=device); re = torch.tensor(re, dtype=torch.float32, device=device)
                dn = torch.tensor(dn, dtype=torch.float32, device=device)
                q_pred = q(ob).gather(1, ac.unsqueeze(1)).squeeze(1)
                with torch.no_grad(): q_next = q_target(nx).max(1).values; target = re + gamma * q_next * (1.0 - dn)
                loss = torch.nn.functional.mse_loss(q_pred, target)
                opt.zero_grad(); loss.backward(); opt.step()
                if t % target_update == 0: q_target.load_state_dict(q.state_dict())
            t += 1
            if done:
                break
        rewards.append(sum(ep_rewards))
        if log_discounted: disc_rewards.append(discounted_returns(ep_rewards, gamma))
        epsilons.append(eps_schedule.value(t-1))

    log = TrainingLog(rewards, disc_rewards if log_discounted else None, epsilons)
    
    def policy(obs_np):
        with torch.no_grad(): qvals = q(torch.from_numpy(obs_np).unsqueeze(0).to(device)).cpu().numpy()[0]
        return int(np.argmax(qvals))
    return policy, log
