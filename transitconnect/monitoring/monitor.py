"""
Telemetry Monitor for TransitConnect.
Performs streaming ingest of raw radio and transport telemetry, applies EWMA smoothing,
and produces normalized features for the handover risk estimator.
"""

from typing import Dict, Any, Tuple
from .filters import EwmaFilter
from .features import FeatureExtractor, NormalizedFeatures

class NetworkMobilityMonitor:
    def __init__(self, ewma_alpha: float = 0.25):
        self.fade_filter = EwmaFilter(alpha=ewma_alpha)
        self.rtt_filter = EwmaFilter(alpha=ewma_alpha)
        self.rtt_var_filter = EwmaFilter(alpha=ewma_alpha)
        self.rsrp_filter = EwmaFilter(alpha=ewma_alpha)
        self.loss_filter = EwmaFilter(alpha=ewma_alpha)
        
        self.extractor = FeatureExtractor()

    def process_sample(self, raw_sample: Dict[str, Any]) -> Tuple[NormalizedFeatures, Dict[str, float]]:
        """
        Processes a single raw telemetry reading and returns:
        1. NormalizedFeatures [F, V, P, R, U, S]
        2. Dict of smoothed physical measurements
        """
        raw_rsrp = float(raw_sample.get("rsrp_dbm", -75.0))
        raw_fade = float(raw_sample.get("signal_fade_rate", 0.0))
        raw_rtt = float(raw_sample.get("rtt_ms", 30.0))
        raw_rtt_var = float(raw_sample.get("rtt_variance_ms", 2.0))
        raw_loss = float(raw_sample.get("packet_loss_ratio", 0.01))
        velocity = float(raw_sample.get("velocity_kmh", 300.0))

        # Smooth readings
        smooth_rsrp = self.rsrp_filter.update(raw_rsrp)
        smooth_fade = self.fade_filter.update(raw_fade)
        smooth_rtt = self.rtt_filter.update(raw_rtt)
        smooth_rtt_var = self.rtt_var_filter.update(raw_rtt_var)
        smooth_loss = self.loss_filter.update(raw_loss)

        features = self.extractor.normalize(
            fade_rate=smooth_fade,
            rtt_var=smooth_rtt_var,
            loss_ratio=smooth_loss,
            rtt=smooth_rtt,
            velocity_kmh=velocity,
            rsrp_dbm=smooth_rsrp
        )

        smoothed_dict = {
            "rsrp_dbm": smooth_rsrp,
            "signal_fade_rate": smooth_fade,
            "rtt_ms": smooth_rtt,
            "rtt_variance_ms": smooth_rtt_var,
            "packet_loss_ratio": smooth_loss,
            "velocity_kmh": velocity
        }

        return features, smoothed_dict

    def reset(self):
        self.fade_filter.reset()
        self.rtt_filter.reset()
        self.rtt_var_filter.reset()
        self.rsrp_filter.reset()
        self.loss_filter.reset()
