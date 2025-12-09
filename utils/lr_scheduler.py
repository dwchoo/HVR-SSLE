import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.lr_scheduler import LambdaLR 

LambdaLR = LambdaLR

class CosineWithScaledRestartsLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_epochs: int,
        num_training_epochs: int,
        num_cycles: int = 1,
        scaling_factor: float = 0.5,
        warmup_epochs_per_cycle: Optional[int] = None,
        last_epoch: int = -1,
        learning_rate_min: float = 0.,
        learning_rate_static: bool = False,
    ):
        self.num_warmup_epochs = num_warmup_epochs
        self.warmup_epochs_per_cycle = warmup_epochs_per_cycle if warmup_epochs_per_cycle is not None else num_warmup_epochs
        self.num_training_epochs = num_training_epochs
        self.num_cycles = num_cycles
        self.scaling_factor = scaling_factor

        self.learning_rate_min = learning_rate_min
        self.learning_rate_static = learning_rate_static

        if self.num_training_epochs > 0:
            self.cycle_epochs = self.num_training_epochs / self.num_cycles
        else:
            self.cycle_epochs = 0

        super(CosineWithScaledRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, current_epoch_zero_indexed: int):
        if self.learning_rate_static:
            return list(self.base_lrs) # Use initial learning rates

        # Use a 1-indexed epoch for calculations to match the logic of num_warmup_epochs, num_training_epochs
        calc_epoch = current_epoch_zero_indexed + 1
        new_lrs = []

        if self.num_training_epochs == 0 or calc_epoch > self.num_training_epochs:
            # Training finished or no training epochs, set to minimum learning rate
            for _ in self.base_lrs:
                new_lrs.append(self.learning_rate_min)
            return new_lrs

        if self.cycle_epochs == 0: # Should only happen if num_training_epochs > 0 but num_cycles is huge or zero
             for base_lr in self.base_lrs:
                new_lrs.append(base_lr * (self.scaling_factor ** 0)) # Effectively base_lr or learning_rate_min
             return [max(lr, self.learning_rate_min) for lr in new_lrs]


        # Determine current cycle (0-indexed)
        current_cycle_zero_indexed = math.floor((calc_epoch - 1) / self.cycle_epochs)
        current_cycle_zero_indexed = min(current_cycle_zero_indexed, self.num_cycles - 1)

        # Scaling factor for the current cycle's max LR
        cycle_scaling_factor = self.scaling_factor ** current_cycle_zero_indexed

        # Calculate epoch within the current cycle (1-indexed)
        epoch_in_cycle_one_indexed = (calc_epoch - 1) - (current_cycle_zero_indexed * self.cycle_epochs) + 1

        current_cycle_length = self.cycle_epochs

        if current_cycle_zero_indexed == 0:
            warmup_epochs_for_this_cycle = self.num_warmup_epochs
        else:
            warmup_epochs_for_this_cycle = self.warmup_epochs_per_cycle

        for base_lr in self.base_lrs:
            max_lr_for_this_cycle = base_lr * cycle_scaling_factor
            lr_val = 0.0

            if epoch_in_cycle_one_indexed <= warmup_epochs_for_this_cycle and warmup_epochs_for_this_cycle > 0:
                # Warmup phase: linear increase from 0 to max_lr_for_this_cycle
                warmup_progress = epoch_in_cycle_one_indexed / warmup_epochs_for_this_cycle
                lr_val = max_lr_for_this_cycle * warmup_progress
            else:
                # Cosine decay phase
                cosine_phase_duration = current_cycle_length - warmup_epochs_for_this_cycle
                if cosine_phase_duration <= 0: # Warmup covers the whole cycle or more
                    lr_val = max_lr_for_this_cycle
                else:
                    current_step_in_cosine = epoch_in_cycle_one_indexed - warmup_epochs_for_this_cycle
                    progress = current_step_in_cosine / cosine_phase_duration
                    progress = min(max(progress, 0.0), 1.0) # Clamp progress to [0, 1]
                    cosine_component = 0.5 * (1 + math.cos(math.pi * progress)) # Ranges from 1 down to 0

                    effective_peak_lr = max(max_lr_for_this_cycle, self.learning_rate_min)
                    lr_val = self.learning_rate_min + (effective_peak_lr - self.learning_rate_min) * cosine_component
            
            # Ensure final LR is not less than learning_rate_min, especially if warmup result is too low.
            # However, typically warmup starts from 0. Cosine part is already handled.
            # If lr_val from warmup is less than learning_rate_min, it will be.
            # The request is for cosine to reach min_lr. Warmup can be lower.
            new_lrs.append(lr_val)
            
        return new_lrs

    def step(self, epoch: Optional[int] = None, scaler=1):
        """
        Update the learning rate based on the current epoch.

        Args:
            epoch (int, optional): The current epoch number. If None, it will be inferred.
                Default: None.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            #epoch = epoch
            epoch = max(epoch, -1) 
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(epoch)):
            param_group['lr'] = lr * scaler



if __name__=="__main__":
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Example model, optimizer, and total training steps
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_training_steps = 1000
    num_warmup_steps = 100
    num_cycles = 2
    scaling_factor = 0.5
    warmup_steps_per_cycle = 50
    min_lr = 1e-6
    
    # Initialize the scheduler
    scheduler = CosineWithScaledRestartsLR(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        scaling_factor=scaling_factor,
        warmup_steps_per_cycle=warmup_steps_per_cycle,
        last_epoch=-1,
    )
    
    lrs = []
    for step in range(num_training_steps):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    plt.plot(range(num_training_steps), lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Cosine with Scaled Restarts and Warmup Learning Rate Schedule")
    plt.show()
    