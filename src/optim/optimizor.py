import torch
from typing import Dict

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from src.utils.register import Register

optimizer_register = Register()

class BaseOptimizer:
    def __init__(self, params, **kwargs):
        self.params = params
        self.optimizer_args = kwargs
        self.optimizer = self.create_optimizer()
        
    def create_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError("Subclasses must implement this method.")

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

@optimizer_register("Adam")
class AdamOptimizer(BaseOptimizer):
    def create_optimizer(self):
        return torch.optim.Adam(self.params, **self.optimizer_args)

@optimizer_register("SGD")
class SGDOptimizer(BaseOptimizer):
    def create_optimizer(self):
        return torch.optim.SGD(self.params, **self.optimizer_args)


def get_optimizor_from_cfg(params, cfg: dict):
    optimizer_type = cfg["type"]
    optimizer_args = cfg["args"]
    if optimizer_type == "SGD":
        return torch.optim.SGD(params, **optimizer_args)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(params, **optimizer_args)

