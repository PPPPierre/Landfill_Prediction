from optimizor import get_optimizor_from_cfg
from scheduler import scheduler_register

def create_optimizer_with_scheduler_from_cfg(params, cfg: dict):
    optimizer_cfg = cfg['optimizer']
    optimizer = get_optimizor_from_cfg(params, optimizer_cfg)
    
    scheduler_cfg = cfg['scheduler']
    scheduler_type = scheduler_cfg['type']
    scheduler_args = scheduler_cfg['args']  
    scheduler_cls = scheduler_register[scheduler_type]
    scheduler = scheduler_cls(optimizer, scheduler_args)

    return optimizer, scheduler

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    def plot_lr(optimizer, scheduler, num_epochs):
        lrs = []
        
        for _ in range(num_epochs):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Scheduling')
        plt.show()

    params = [torch.randn(10, requires_grad=True)]
    cfg = {
        "optimizer": {
            "type": "SGD",
            "args": {
                "lr": 0.1,
                "weight_decay": 0.0,
            }
        },
        "scheduler": {
            "type": "step",
            "args": {
                "lr_schedule": {
                    5: 0.5,
                    10: 0.1,
                    15: 0.01
                },
                "warm_up": 4
            }
        }
    }

    optimizer, scheduler = create_optimizer_with_scheduler_from_cfg(params, cfg)
    scheduler.set_last_epoch(4)
    plot_lr(optimizer, scheduler, 20)
