"""
Dataset Generator for Handover Risk ML Models in TransitConnect.
Takes raw simulation telemetry traces and extracts normalized 6-feature vectors
labeled with ground-truth impending handover risk states (Low, Moderate, High)
and continuous target risk scores.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from ..monitoring.monitor import NetworkMobilityMonitor
from ..monitoring.features import NormalizedFeatures

FEATURE_COLUMNS = ["F_fade_rate", "V_rtt_variance", "P_packet_loss", "R_rtt_condition", "U_velocity", "S_signal_condition"]

class DatasetGenerator:
    def __init__(self, high_risk_window_s: float = 1.5, mod_risk_window_s: float = 3.5):
        self.high_risk_window_s = high_risk_window_s
        self.mod_risk_window_s = mod_risk_window_s

    def build_dataset_from_dataframe(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes trace dataframe through NetworkMobilityMonitor and computes
        ground-truth risk labels based on temporal proximity to handover events.
        """
        monitor = NetworkMobilityMonitor()
        
        # Identify timestamps of handover events
        ho_mask = trace_df["is_handover"] == 1
        ho_times = trace_df.loc[ho_mask, "timestamp_s"].values

        records = []
        for _, row in trace_df.iterrows():
            features, smoothed = monitor.process_sample(row.to_dict())
            ts = float(row["timestamp_s"])

            # Find time until nearest upcoming handover
            future_hos = ho_times[ho_times >= ts]
            if len(future_hos) > 0:
                time_to_ho = future_hos[0] - ts
            else:
                # Check recent past handover (during execution / settling)
                past_hos = ho_times[ho_times < ts]
                if len(past_hos) > 0:
                    time_to_ho = -(ts - past_hos[-1])
                else:
                    time_to_ho = 999.0

            # Determine risk label and continuous target score
            if row.get("is_handover", 0) == 1 or (0.0 <= time_to_ho <= self.high_risk_window_s):
                risk_label = 2  # HIGH
                risk_class = "HIGH_RISK"
                # Target risk score around 0.75 - 1.0
                target_score = 0.75 + 0.25 * features.P
            elif 0.0 <= time_to_ho <= self.mod_risk_window_s:
                risk_label = 1  # MODERATE
                risk_class = "MODERATE_RISK"
                target_score = 0.40 + 0.30 * (1.0 - time_to_ho / self.mod_risk_window_s)
            else:
                risk_label = 0  # LOW
                risk_class = "LOW_RISK"
                target_score = 0.05 + 0.25 * features.S

            target_score = min(1.0, max(0.0, float(target_score)))

            rec = {
                "timestamp_s": ts,
                "time_to_ho_s": round(time_to_ho, 3),
                "F_fade_rate": features.F,
                "V_rtt_variance": features.V,
                "P_packet_loss": features.P,
                "R_rtt_condition": features.R,
                "U_velocity": features.U,
                "S_signal_condition": features.S,
                "risk_class": risk_class,
                "risk_label": risk_label,
                "risk_score_target": round(target_score, 4)
            }
            records.append(rec)

        return pd.DataFrame(records)
