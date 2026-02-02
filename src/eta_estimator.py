import time
from dataclasses import dataclass


@dataclass
class ETAEstimator:
    total: int
    alpha: float = 0.2
    _start: float = dataclass(init=False, repr=False)
    _last_t: float = dataclass(init=False, repr=False)
    _ema_rate: float | None = None
    done: int = 0

    def __post_init__(self) -> None:
        now = time.time()
        self._start = now
        self._last_t = now

    def update(self, delta_done: int) -> None:
        now = time.time()
        dt = max(now - self._last_t, 1e-9)
        inst_rate = delta_done / dt

        if self._ema_rate is None:
            self._ema_rate = inst_rate
        else:
            self._ema_rate = self.alpha * inst_rate + (1 - self.alpha) * self._ema_rate

        self.done += delta_done
        self._last_t = now

    def eta_seconds(self) -> float | None:
        if self._ema_rate is None or self._ema_rate <= 0:
            return None
        remaining = max(self.total - self.done, 0)
        return remaining / self._ema_rate

    def fmt(self) -> str:
        eta = self.eta_seconds()
        if eta is None:
            return f"{self.done}/{self.total} (ETA unknown)"
        h = int(eta // 3600)
        m = int((eta % 3600) // 60)
        s = int(eta % 60)
        return f"{self.done}/{self.total} (ETA {h:d}:{m:02d}:{s:02d})"

