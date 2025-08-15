import threading

import numpy


class WelFord:
    """
    ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.
    """

    def __init__(self) -> None:
        self.mean: float = 0.
        self.variance: float = 0.
        self.num: int = 0
        self._lock = threading.Lock()

    def __repr__(self):
        return f"WelFord(mean={self.mean}, variance={self.variance}, num={self.num})"

    def update(self, batch: numpy.ndarray | float) -> (float, float):
        if isinstance(batch, numpy.ndarray):
            batch_mean = batch.mean().item()
            batch_var = batch.var().item()
            batch_size = batch.size
        else:
            batch_mean, batch_var, batch_size = batch, 0, 1
        return self.update_with_mean_var(batch_mean, batch_var, batch_size)

    def update_with_mean_var(self, batch_mean: float, batch_var: float, batch_size: int) -> (float, float):
        with self._lock:
            total_n = self.num + batch_size
            # update mean
            delta = batch_mean - self.mean
            self.mean += delta * (batch_size / total_n)
            # update variance
            old_rate, new_rate = (self.num / total_n), (batch_size / total_n)
            self.variance = (
                    self.variance * old_rate
                    + batch_var * new_rate
                    + (delta ** 2) * (old_rate * new_rate)
            )
            # update batch size
            self.num = total_n

        return self.mean, self.variance
