
import math
import random

class EpsilonGreedySchedule:
    def __init__(self, start=1.0, end=0.06, decay_steps=10_000, mode="exp"):
        self.start, self.end, self.decay_steps, self.mode = start, end, decay_steps, mode

    def value(self, t):
        if self.mode == "linear":
            frac = max(0.0, min(1.0, 1.0 - (t / float(max(1, self.decay_steps)))))
            return self.end + (self.start - self.end) * frac
        
        elif self.mode == "exp":
            if self.end <= 0: return 0.0
            k = math.log(self.start / self.end) / float(max(1, self.decay_steps))
            return max(self.end, self.start * math.exp(-k * t))
        return self.end

def epsilon_greedy(Q_values, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return max(range(n_actions), key=lambda a: Q_values[a])
