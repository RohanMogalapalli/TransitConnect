"""
Feature preparation and normalization module for TransitConnect.
Extracts and scales the six core monitoring parameters:
- F: Normalized Signal Fade Rate [0, 1]
- V: Normalized RTT Variance [0, 1]
- P: Normalized Packet Loss Ratio [0, 1]
- R: Normalized RTT Condition [0, 1]
- U: Normalized Vehicle Velocity [0, 1]
- S: Normalized Radio Signal Condition [0, 1] (0 = excellent, 1 = severely attenuated)
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class NormalizedFeatures:
    F: float  # Signal fade rate [0, 1]
    V: float  # RTT variance [0, 1]
    P: float  # Packet loss [0, 1]
    R: float  # RTT condition [0, 1]
    U: float  # Vehicle velocity [0, 1]
    S: float  # Signal condition [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([self.F, self.V, self.P, self.R, self.U, self.S], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {
            "F_fade_rate": self.F,
            "V_rtt_variance": self.V,
            "P_packet_loss": self.P,
            "R_rtt_condition": self.R,
            "U_velocity": self.U,
            "S_signal_condition": self.S
        }

def clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(val)))

class FeatureExtractor:
    def __init__(
        self,
        max_fade_rate: float = 35.0,     # dB/s
        max_rtt_variance: float = 40.0,  # ms
        min_rtt_ms: float = 20.0,        # ms
        max_rtt_ms: float = 180.0,       # ms
        max_velocity_kmh: float = 350.0, # km/h
        best_rsrp_dbm: float = -55.0,    # dBm
        worst_rsrp_dbm: float = -115.0   # dBm
    ):
        self.max_fade_rate = max_fade_rate
        self.max_rtt_variance = max_rtt_variance
        self.min_rtt_ms = min_rtt_ms
        self.max_rtt_ms = max_rtt_ms
        self.max_velocity_kmh = max_velocity_kmh
        self.best_rsrp_dbm = best_rsrp_dbm
        self.worst_rsrp_dbm = worst_rsrp_dbm

    def normalize(
        self,
        fade_rate: float,
        rtt_var: float,
        loss_ratio: float,
        rtt: float,
        velocity_kmh: float,
        rsrp_dbm: float
    ) -> NormalizedFeatures:
        """
        Maps raw physical network readings into normalized [0, 1] representations
        where 0 indicates optimal/stable condition and 1 indicates severe degradation.
        """
        # 1. Fade rate: positive fade rate means signal is dropping
        F = clamp(max(0.0, fade_rate) / self.max_fade_rate)

        # 2. RTT Variance
        V = clamp(max(0.0, rtt_var) / self.max_rtt_variance)

        # 3. Packet Loss Ratio: already in [0, 1]
        P = clamp(loss_ratio)

        # 4. RTT Condition
        rtt_norm = (rtt - self.min_rtt_ms) / (self.max_rtt_ms - self.min_rtt_ms)
        R = clamp(rtt_norm)

        # 5. Vehicle Velocity
        U = clamp(velocity_kmh / self.max_velocity_kmh)

        # 6. Signal condition: invert RSRP (lower RSRP -> higher degradation score S)
        rsrp_norm = (self.best_rsrp_dbm - rsrp_dbm) / (self.best_rsrp_dbm - self.worst_rsrp_dbm)
        S = clamp(rsrp_norm)

        return NormalizedFeatures(F=F, V=V, P=P, R=R, U=U, S=S)
