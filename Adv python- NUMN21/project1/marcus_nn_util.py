from Final import Final_nn_classes as nn


class PlateauLrs(nn.LRScheduler):
    """Reduce LR when metric is not decreasing.

    Not that useful for FFNs?"""

    def __init__(
        self,
        patience: int = 50,
        factor: float = 0.5,
        thr: float = 1e-3,
        cooldown: int = 50,
        min_lr: float = 1e-3,
    ) -> None:
        self.patience = patience
        self.factor = factor
        self.thr = thr
        self.cooldown = cooldown
        self.min_lr = min_lr

        self.best = float("inf")  # assuming 'min' mode
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

        self.nsteps = 0

    def update(self, current_lr, metric):
        # improvement check
        if metric < self.best - self.thr:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # cooldown counting
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return current_lr

        # trigger lr reduction
        if self.num_bad_epochs > self.patience:
            new_lr = max(self.min_lr, current_lr * self.factor)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.nsteps += 1
            return new_lr

        return current_lr


class StepLrs(nn.LRScheduler):
    def __init__(
        self,
        step: float,
        interval: int,
        min_lr: float = 1e-3,
    ) -> None:
        self.step = step
        self.interval = interval
        self.min_lr = min_lr

    def update(self, current_lr, metric):
        if self.since_step > self.interval:
            self.since_step = 0
            return max(self.min_lr, current_lr - self.step)
        else:
            self.since_step += 1
            return current_lr
