import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
    def lr_lambda(current_epoch):
        base_lr = optimizer.param_groups[0]['initial_lr']

        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return min_lr / base_lr + cosine_decay * (1.0 - min_lr / base_lr)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)