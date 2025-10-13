# 🟡 Reinforcement Learning Algorithms Applied to a Noisy Pacman Grid

This project explores and compares several **Reinforcement Learning (RL)** and **planning** algorithms applied to a simplified version of the classic **Pacman** game.  
The objective is simple: Pacman must reach the dot to win, while avoiding ghosts in a dynamic, noisy gridworld.

The full study, results, and detailed analysis are presented in the report  
👉 **[`report.pdf`](./report.pdf)** (recommended read).

---

## 📘 Overview

The environment is a 10×10 grid with customizable parameters:
- Random starting positions  
- Action noise (`p_n`)  
- Moving ghosts (`p_g`)  
- Reward shaping (dot, ghost, step penalties)

Implemented algorithms:
- **Q-learning**
- **SARSA**
- **Monte Carlo Tree Search (MCTS)**
- **Deep Q-Network (DQN)** (PyTorch implementation)

Each method is trained and compared through:
- Total reward evolution  
- Win-rate after training  
- Parameter sweeps (`α`, `γ`, learning rate)  
- Robustness under noise and dynamic environments  

---

## 🧩 Project Structure

```
├── src/
│   ├── env.py              # Pacman environment (grid, rendering, noise)
│   ├── algos/
│   │   ├── q_learning.py   # Tabular Q-learning
│   │   ├── sarsa.py        # SARSA implementation
│   │   ├── dqn.py          # Deep Q-Network (PyTorch)
│   │   └── mcts.py         # Monte Carlo Tree Search
│   ├── training.py         # Training loops and evaluation utilities
│   ├── policies.py         # Epsilon-greedy schedules (linear / exponential)
│   └── utils.py            # Plotting, logging, and metrics
├── assets/                 # Pacman, ghost, and dot images
├── main.ipynb              # Main notebook (visualization + experiments)
├── requirements.txt
└── report.pdf              # Full project report
```

---

## 🧠 Key Results

| Method | Static Env. | Noisy Env. | Dynamic Env. | Training Time | 
|:-------|:------------:|:-----------:|:-------------:|:--------------:|
| Q-learning | ✅ Fast convergence | ❌ Sensitive | ⚠️ Poor generalization | ⚡ Fast | 
| SARSA | ⚙️ Slower convergence | ❌ Sensitive | ⚠️ Unstable | ⚡ Fast | 
| MCTS | ⚙️ Planning-based | ⚠️ Reward-dependent | ❌ Unstable | None but slow "inference"| 
| DQN | ✅ Robust | ✅ Stable | ✅ Scalable | 🧠 Moderate | 

---

## 🎯 Example Visuals

### Agent Behavior for DQN (10×10, two ghosts) 
<img width="400" height="400" alt="optimal_path_found_by_DQN" src="https://github.com/user-attachments/assets/ac4a22e8-a7f9-4576-b109-b66bdfb26e90" />

### Comparison reward evolution in a deterministic environment
<img width="720" height="400" alt="overlay_base" src="https://github.com/user-attachments/assets/a845e594-8ae2-4bff-ad6a-f39abb13907e" />

### Comparison reward evolution in a stochastic environment
<img width="720" height="400" alt="overlay_dyn_0 2" src="https://github.com/user-attachments/assets/61518dcb-5179-497a-8048-b3392341159c" />

---
## ⚙️ Usage

### 1. Clone the repository
```bash
git clone https://github.com/singarin-sole-L/Q-PACMAN.git
cd Q-PACMAN
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run experiments
```bash
jupyter notebook main.ipynb
```

You can modify grid parameters and environment behavior directly inside the notebook.

---

## 🧾 Notes

- The DQN implementation follows a PyTorch-style structure inspired by the official [Reinforcement Q-learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).  
- MCTS performance strongly depends on **reward configuration** (step cost, dot reward, ghost penalty).  
- For more detailed explanations, mathematical formulations, and comparisons —  
  please refer to the **full report** (`report.pdf`).

---

## 🚀 Future Work

Extending this study to larger or continuous environments, with more complex dynamics (multiple ghosts, stochastic rewards),  
could provide deeper insight into the scalability and generalization of each algorithm.

---

## 🧑‍💻 Author

**SINGARIN-SOLE Livio**   
October 2025
