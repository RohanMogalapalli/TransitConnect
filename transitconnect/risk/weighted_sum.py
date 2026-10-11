"""
Mathematical Handover-Risk Model for TransitConnect.
Implements the formal weighted-sum formulation:
H = w1 * F + w2 * V + w3 * P + w4 * R + w5 * U

Where:
- H: Handover-risk score [0, 1]
- F: Normalized signal fade rate
- V: Normalized RTT variance
- P: Normalized packet loss ratio
- R: Normalized RTT condition
- U: Normalized vehicle velocity
- w1..w5: Non-negative weights summing to 1.0
"""

from typing import Tuple, Dict, Any, Optional
from ..monitoring.features import NormalizedFeatures, clamp

class RiskState:
    LOW = "LOW_RISK"
    MODERATE = "MODERATE_RISK"
    HIGH = "HIGH_RISK"

class WeightedSumRiskModel:
    def __init__(
        self,
        w1: float = 0.30,  # Fade rate weight
        w2: float = 0.20,  # RTT variance weight
        w3: float = 0.25,  # Packet loss weight
        w4: float = 0.15,  # RTT condition weight
        w5: float = 0.10,  # Velocity weight
        t1: float = 0.35,  # Low -> Moderate threshold
        t2: float = 0.70   # Moderate -> High threshold
    ):
        weights = [w1, w2, w3, w4, w5]
        total_w = sum(weights)
        if total_w <= 0.0:
            raise ValueError("Sum of weights must be positive")
        
        # Normalize weights to sum exactly to 1.0
        self.w1 = w1 / total_w
        self.w2 = w2 / total_w
        self.w3 = w3 / total_w
        self.w4 = w4 / total_w
        self.w5 = w5 / total_w

        self.t1 = float(t1)
        self.t2 = float(t2)
        if self.t1 >= self.t2:
            raise ValueError(f"Threshold t1 ({t1}) must be less than t2 ({t2})")

    def compute_risk(self, features: NormalizedFeatures) -> Tuple[float, str]:
        """
        Computes the continuous risk score H in [0, 1] and classifies it into RiskState.
        """
        H_raw = (
            self.w1 * features.F +
            self.w2 * features.V +
            self.w3 * features.P +
            self.w4 * features.R +
            self.w5 * features.U
        )
        H = clamp(H_raw, 0.0, 1.0)

        if H < self.t1:
            state = RiskState.LOW
        elif H < self.t2:
            state = RiskState.MODERATE
        else:
            state = RiskState.HIGH

        return H, state

    def get_config(self) -> Dict[str, Any]:
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "w4": self.w4,
            "w5": self.w5,
            "t1": self.t1,
            "t2": self.t2
        }
