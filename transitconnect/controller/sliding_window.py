"""
Adaptive Sliding-Window Controller for TransitConnect.
Implements the formal relationship:
W = max(W_min, floor(W_0 * (1 - alpha * H)))

Where:
- W: Adapted sliding-window size (in packets)
- W_0: Baseline window size in stable channel conditions
- W_min: Minimum window floor to prevent connection stall
- alpha: Window shrinkage sensitivity factor [0, 1]
- H: Handover-risk score [0, 1]
"""

import math
from typing import Dict, Any

class AdaptiveSlidingWindow:
    def __init__(
        self,
        w0: int = 64,        # Default window size (packets)
        w_min: int = 4,      # Minimum window floor (packets)
        alpha: float = 0.85  # Sensitivity factor
    ):
        if w_min <= 0:
            raise ValueError("w_min must be at least 1 packet")
        if w0 < w_min:
            raise ValueError(f"w0 ({w0}) cannot be smaller than w_min ({w_min})")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.w0 = int(w0)
        self.w_min = int(w_min)
        self.alpha = float(alpha)
        self.current_w = self.w0

    def compute_window(self, risk_score_h: float) -> int:
        """Computes adapted window size W based on predicted risk H."""
        h = max(0.0, min(1.0, float(risk_score_h)))
        adapted = math.floor(self.w0 * (1.0 - self.alpha * h))
        self.current_w = max(self.w_min, int(adapted))
        return self.current_w

    def get_config(self) -> Dict[str, Any]:
        return {
            "w0": self.w0,
            "w_min": self.w_min,
            "alpha": self.alpha,
            "current_w": self.current_w
        }
