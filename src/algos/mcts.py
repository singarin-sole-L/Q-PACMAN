
import numpy as np, math, random
import copy

class Node:
    def __init__(self, obs_key, n_actions=4):
        self.obs_key = obs_key
        self.N = np.zeros(n_actions, dtype=np.float32)
        self.W = np.zeros(n_actions, dtype=np.float32)
        self.Q = np.zeros(n_actions, dtype=np.float32)
        self.P = np.ones(n_actions, dtype=np.float32)/n_actions
        self.children = {}

def obs_to_key(obs):
    planes = obs
    pac = tuple(np.argwhere(planes[4]==1.0)[0])
    ghosts = tuple(map(tuple, np.argwhere(planes[3]==1.0)))
    dot = tuple(np.argwhere(planes[2]==1.0)[0])
    return (pac, ghosts, dot)

def ucb_scores(node, c):
    N_sum = max(1.0, float(np.sum(node.N)))
    U = c * node.P * (math.sqrt(N_sum) / (1.0 + node.N))
    return node.Q + U

def default_rollout(env, depth, gamma):
    total, g, done = 0.0, 1.0, False
    for _ in range(depth):
        if done: break
        a = random.randrange(env.n_actions)
        _, r, done, _ = env.step(a)
        total += g * r; g *= gamma
    return total

def plan_action(env, simulations=100, c=1, rollout_depth=20, gamma=0.99):
    
    root_obs = env._obs()
    root = Node(obs_to_key(root_obs), n_actions=env.n_actions)
    for _ in range(simulations):
        node = root; env_copy = copy.deepcopy(env); done = False; path = []
        while True:
            scores = ucb_scores(node, c)
            a = int(np.argmax(scores))
            obs, r, done, _ = env_copy.step(a)
            path.append((node, a, r))
            key = obs_to_key(obs)
            if a not in node.children:
                child = Node(key, n_actions=env.n_actions)
                node.children[a] = child
                value = default_rollout(env_copy, rollout_depth, gamma)
                for n, action, r_ in reversed(path):
                    n.N[action] += 1.0
                    n.W[action] += r_ + gamma * value
                    n.Q[action] = n.W[action] / n.N[action]
                    value = r_ + gamma * value
                break
            else:
                node = node.children[a]
                if done:
                    value = 0.0
                    for n, action, r_ in reversed(path):
                        n.N[action] += 1.0
                        n.W[action] += r_ + gamma * value
                        n.Q[action] = n.W[action] / n.N[action]
                        value = r_ + gamma * value
                    break
    return int(np.argmax(root.N))
