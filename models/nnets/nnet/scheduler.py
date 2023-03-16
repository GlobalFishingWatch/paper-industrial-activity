import numpy as np


class CosineAnnealingScheduler:
    """Scheduler for cosine annealing

    Learning rate ramps up to `max_lr` from zero over
    `warmup` epochs. The learning rate then decays
    to `min_lr` following a cosine trajectory over
    `n0` epochs. The learning rate then jumps back
    to `max_lr` and this repeats except that the period
    doubles with each repetition.

    Args:
        warmup : int
        n0 : int
        max_lr : float
        min_lr : float

    Returns:
        learing rate value for specific epoch.
    """

    def __init__(
        self,
        warmup=10,
        n0=50,
        max_lr=1e-2,
        min_lr=0,
        length_scale=1,
        max_lr_scale=1,
    ):
        assert n0 > 1
        self.n0 = n0
        self.n = n0
        self.i0 = warmup + 1
        self.warmup = warmup
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.length_scale = length_scale
        self.max_lr_scale = max_lr_scale
        self.cycle = 0

    def __call__(self, epoch: int, lr: float) -> float:
        n = epoch + 1
        if n <= self.warmup:
            # Reset n0 in case we've restarted
            self.n = self.n0
            return (
                n / (self.warmup + 1) * (self.max_lr - self.min_lr)
                + self.min_lr
            )
        if n - self.i0 >= self.n:
            self.cycle += 1
            self.i0 = n
            self.n = int(round(self.n * self.length_scale))
        arg = np.pi * (n - self.i0) / (self.n - 1)
        delta = (self.max_lr - self.min_lr) * self.max_lr_scale**self.cycle
        return self.min_lr + 0.5 * delta * (1 + np.cos(arg))

    def epochs_for_cycles(self, cycles: int) -> int:
        """How many epochs are needed to complete cycles annealings

        Args:
            cycles : int

        Returns:
            epochs : int
        """
        cnt = self.warmup
        n = self.n0
        for _ in range(cycles):
            cnt += n
            n = int(round(n * self.length_scale))
        return cnt
