import random
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


UP, LEFT, RIGHT, DOWN = 0, 1, 2, 3
ACTIONS = [UP, LEFT, RIGHT, DOWN]
ACTION_GRID = {UP:(-1,0), DOWN:(1,0), LEFT:(0,-1), RIGHT:(0,1)}

class Ghost:
    def __init__(self, pos, move_prob=0.0):
        self.pos = tuple(pos)
        self.move_prob = float(move_prob)

    def step(self, grid, rng):
        if rng.random() > self.move_prob: return
        H, W = grid.shape
        actions = list(ACTIONS)
        rng.shuffle(actions)
        
        for action in actions:
            ai, aj = ACTION_GRID[action]
            ni, nj = self.pos[0]+ai, self.pos[1]+aj
            if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] != 1:
                self.pos = (ni, nj)
                return

class PacmanGridEnv:
    def __init__(self, grid, dot_pos, ghost_starts, pacman_start,
                 noise_prob=0.0, random_start=True, step_cost=-1.0,
                 dot_reward=10.0, ghost_penalty=-10.0, move_prob_ghost=0.0,
                 n_ghosts=1, seed=0, assets_dir=None):
        self.grid = grid
        self.dot_pos = tuple(dot_pos)
        self.ghost_starts = [tuple(g) for g in ghost_starts]
        self.pacman_start = tuple(pacman_start)
        self.noise_prob = float(noise_prob)
        self.random_start = bool(random_start)
        self.step_cost = float(step_cost)
        self.dot_reward = float(dot_reward)
        self.ghost_penalty = float(ghost_penalty)
        self.move_prob_ghost = float(move_prob_ghost)
        self.n_ghosts = int(n_ghosts)
        self.seed = int(seed)
        self.assets_dir = assets_dir
        self.rng = random.Random(self.seed)
        self.reset()
    
    @property
    def n_actions(self):
        return len(ACTIONS)

    def reset(self):
        H, W = self.grid.shape
        self.ghosts = []
        for k in range(self.n_ghosts):
            start = self.ghost_starts[min(k, len(self.ghost_starts)-1)]
            self.ghosts.append(Ghost(start, self.move_prob_ghost))

        if self.random_start:
            empties = [(i,j) for i in range(H) for j in range(W)
                       if self.grid[i,j]==0 and (i,j)!=self.dot_pos and (i,j) not in [ghost.pos for ghost in self.ghosts]]
            self.pacman = self.rng.choice(empties) if empties else self.pacman_start
        else: 
            self.pacman = self.pacman_start
        self.done = False
        return self._obs()
    
    def _valid(self, i, j):
        H, W = self.grid.shape
        return (0<=i<H) and (0<=j<W) and (self.grid[i,j] != 1)
    
    def step(self, action):
        if self.done: 
            return self._obs(), 0.0, True, {}
        if self.rng.random() < self.noise_prob:
            action = self.rng.randrange(self.n_actions)
        ai, aj = ACTION_GRID[action]
        ni, nj = self.pacman[0]+ai, self.pacman[1]+aj

        if self._valid(ni, nj): 
            self.pacman = (ni, nj)

        for ghost in self.ghosts: 
            ghost.step(self.grid, self.rng)

        reward = self.step_cost
        if self.pacman == self.dot_pos: 
            reward = self.dot_reward
            self.done = True
        elif self.pacman in [ghost.pos for ghost in self.ghosts]: 
            reward = self.ghost_penalty; self.done = True

        return self._obs(), reward, self.done, {}
    
    def _obs(self):
        H, W = self.grid.shape
        planes = np.zeros((5, H, W), dtype=np.float32)
        planes[0] = (self.grid == 0).astype(np.float32)
        planes[1] = (self.grid == 1).astype(np.float32)
        planes[2][self.dot_pos] = 1.0
        for ghost in self.ghosts: 
            planes[3][ghost.pos] = 1.0

        planes[4][self.pacman] = 1.0
        return planes
    
    def render(self, cell_size=48, show=True, borders=True, path=None, save_path=None):
        H, W = self.grid.shape

        fig, ax = plt.subplots(figsize=(W*cell_size/96.0, H*cell_size/96.0))
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis(); ax.set_aspect('equal')

        pacman_img = dot_img = ghost_img = None
        if self.assets_dir:
                p = Path(self.assets_dir)
                pacman_img = mpimg.imread(str(p / "pacman.png")) if (p/"pacman.png").exists() else None
                dot_img = mpimg.imread(str(p / "dot.png")) if (p/"dot.png").exists() else None
                ghost_img = mpimg.imread(str(p / "ghost.png")) if (p/"ghost.png").exists() else None

        for i in range(H):
            for j in range(W):
                if self.grid[i,j] == 1: 
                    ax.add_patch(plt.Rectangle((j,i),1, 1,fill=True,alpha=0.8))
                else: 
                    ax.add_patch(plt.Rectangle((j,i), 1,1,fill=False,alpha=0.3))
                if borders: 
                    ax.add_patch(plt.Rectangle((j,i), 1, 1, fill=False, linewidth=0.65))

        def place(img, cell, scale=0.9):
            if img is None: return False
            i,j = cell; margin = (1.0-scale)/2.0
            extent = (j+margin, j+1-margin, i+1-margin, i+margin)
            ax.imshow(img, extent=extent, aspect='auto'); return True
        
        if not place(dot_img, self.dot_pos):
            ax.plot(self.dot_pos[1]+0.5, self.dot_pos[0]+0.5, marker="o", markersize=cell_size/3.0)

        for g in self.ghosts:
            if not place(ghost_img, g.pos): ax.plot(g.pos[1]+0.5, g.pos[0]+0.5, marker="X", markersize=cell_size/3.0)

        if not place(pacman_img, self.pacman):
            ax.plot(self.pacman[1]+0.5, self.pacman[0]+0.5, marker=">", markersize=cell_size/3.0)
        if path:
            xs = [c[1]+0.5 for c in path]; ys = [c[0]+0.5 for c in path]
            ax.plot(xs, ys, linestyle="--")
        ax.set_xticks(range(W+1)); ax.set_yticks(range(H+1)); ax.grid(False)
        ax.set_xticklabels([]); ax.set_yticklabels([]); plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path, dpi=160)
        if show: 
            plt.show()
        else: 
            plt.close(fig)

def make_grid(H=10, W=10):
    grid = np.zeros((H, W), dtype=int)
    mid_x = W // 2
    for i in range(H // 2):
        grid[i, mid_x] = 1
    wall_y = int(H * 0.65)

    for j in range(int(W * 0.75)):
        grid[wall_y, j] = 1

    grid[wall_y, 2] = 0
    grid[wall_y, 5] = 0

    pacman_start = (H // 2, 1)

    dot_pos = (H - 2, W - 3)

    ghost_starts = [(1, 1), (H - 2, 1)]

    return grid, dot_pos, ghost_starts, pacman_start

