import torch
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from src.utils.register import Register

scheduler_register = Register()

@scheduler_register("step")
class StepLRScheduler:
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 scheduler_args: dict,
                 ):
        self.optimizer = optimizer
        self.lr_schedule = scheduler_args['lr_schedule']
        self.warm_up_epochs = scheduler_args.get('warm_up', 0)
        self.initial_lr = scheduler_args.get('lr', optimizer.defaults['lr'])
        self.last_epoch = -1
        if self.warm_up_epochs > 0:
            self.current_lr = self.initial_lr / self.warm_up_epochs
        else:
            self.current_lr = self.initial_lr

        # Update the learning rate for all param groups in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def set_last_epoch(self, last_epoch: int):
        self.last_epoch = -1
        if self.warm_up_epochs > 0:
            self.current_lr = self.initial_lr / self.warm_up_epochs
        else:
            self.current_lr = self.initial_lr
        while self.last_epoch < last_epoch:
            self.step()

    def step(self):
        # Increment the internal epoch counter
        self.last_epoch += 1
        
        if self.last_epoch < self.warm_up_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warm_up_epochs
            self.current_lr = self.initial_lr * warmup_factor
        elif self.last_epoch + 1 in self.lr_schedule:
            self.current_lr = self.initial_lr * self.lr_schedule[self.last_epoch + 1]
        
        # Update the learning rate for all param groups in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def get_lr(self):
        return self.current_lr