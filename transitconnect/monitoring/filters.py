"""
Telemetry filtering for TransitConnect.
Provides Exponentially Weighted Moving Average (EWMA) filters to smooth
raw signal strength, fade rate, and delay variations while preserving transient edge response.
"""

from typing import Optional

class EwmaFilter:
    def __init__(self, alpha: float = 0.2, initial_value: Optional[float] = None):
        """
        EWMA Filter: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        alpha in (0, 1]: Higher alpha reacts faster to changes; lower alpha smooths noise.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.value = initial_value
        self.initialized = initial_value is not None

    def update(self, sample: float) -> float:
        if not self.initialized or self.value is None:
            self.value = float(sample)
            self.initialized = True
        else:
            self.value = self.alpha * float(sample) + (1.0 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        return self.value

    def reset(self, initial_value: Optional[float] = None):
        self.value = initial_value
        self.initialized = initial_value is not None
