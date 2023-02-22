import torch
import numpy as np


class ScheduledOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, embedding_dim: int, warmup_steps: int):
        self._optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        self.init_lrate = np.power(embedding_dim, -0.5)

    def get_lrate_scale(self):
        return np.min([np.power(self.current_steps, -0.5), self.current_steps*np.power(self.warmup_steps, -1.5)])

    def _update_learning_rate(self):
        self.current_steps += 1
        lrate = self.init_lrate * self.get_lrate_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lrate

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()