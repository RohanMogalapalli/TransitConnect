"""
Trace Loader for TransitConnect.
Reads, validates, and replays empirical and synthetic CSV mobility traces.
Supports both standard simulation traces and real-world drive-test datasets
such as 'Handover Dataset (ver 2)'.
"""

import os
import glob
from pathlib import Path
from typing import Union, Generator, Dict, Any, Optional, List
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "timestamp_s", "rsrp_dbm", "signal_fade_rate",
    "rtt_ms", "rtt_variance_ms", "packet_loss_ratio", "velocity_kmh"
]

def load_real_handover_dataset(dataset_dir: Union[str, Path] = "Handover Dataset (ver 2)/Processed Dataset") -> pd.DataFrame:
    """
    Loads and standardizes all CSV files from 'Handover Dataset (ver 2)/Processed Dataset'.
    Maps 3GPP RSRP index, CINR, Speed, and Handover Trigger to the TransitConnect 6-feature format.
    """
    path = Path(dataset_dir)
    if not path.exists():
        # Try finding relative to project root
        candidate = Path(__file__).resolve().parent.parent.parent / "Handover Dataset (ver 2)" / "Processed Dataset"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Handover dataset directory not found at {dataset_dir}")

    files = sorted(glob.glob(os.path.join(str(path), "*ver_1.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(str(path), "*.csv")))

    dfs = []
    cumulative_time = 0.0

    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]

        # Standardize sample timestamps (500ms intervals between drive test reports)
        n_samples = len(df)
        ts = cumulative_time + np.arange(n_samples) * 0.5
        df["timestamp_s"] = np.round(ts, 2)
        cumulative_time = float(ts[-1] + 1.0) if len(ts) > 0 else cumulative_time

        # 3GPP RSRP Index (typically 20 to 80) -> dBm conversion (index - 140)
        rsrp_col = None
        for col in ["Serving Cell RSRP", "Serving Cell RSRP    "]:
            if col in df.columns:
                rsrp_col = col
                break

        if rsrp_col:
            rsrp_raw = pd.to_numeric(df[rsrp_col], errors="coerce").fillna(50)
            df["rsrp_dbm"] = np.round(rsrp_raw - 140.0, 2)
        else:
            df["rsrp_dbm"] = -90.0

        # Velocity in km/h
        if "Speed (km/h)" in df.columns:
            df["velocity_kmh"] = pd.to_numeric(df["Speed (km/h)"], errors="coerce").fillna(30.0).clip(lower=5.0)
        else:
            df["velocity_kmh"] = 45.0

        # Handover Trigger
        if "Handover Trigger" in df.columns:
            df["is_handover"] = pd.to_numeric(df["Handover Trigger"], errors="coerce").fillna(0).astype(int)
        else:
            df["is_handover"] = 0

        # Cumulative handover count
        df["handover_count"] = df["is_handover"].cumsum()

        # Signal Fade Rate: d(RSRP) / dt (dB/s)
        rsrp_vals = df["rsrp_dbm"].values
        ts_vals = df["timestamp_s"].values
        dt = np.diff(ts_vals, prepend=ts_vals[0] - 0.5)
        dt = np.where(dt <= 0, 0.5, dt)
        df["signal_fade_rate"] = np.round(-np.diff(rsrp_vals, prepend=rsrp_vals[0]) / dt, 2)

        # Packet Loss Ratio derived from CINR and Handover state
        cinr = pd.to_numeric(df.get("Serving Cell CINR", 0), errors="coerce").fillna(0)
        # Low CINR (< 0 dB) increases error probability; handovers spike loss
        loss = 0.01 + 0.35 * np.clip((5.0 - cinr) / 20.0, 0.0, 1.0)
        loss = np.where(df["is_handover"] == 1, np.maximum(loss, 0.75), loss)
        df["packet_loss_ratio"] = np.round(loss, 4)

        # RTT in ms
        base_rtt = 25.0 + 35.0 * np.clip((5.0 - cinr) / 20.0, 0.0, 1.0)
        base_rtt = np.where(df["is_handover"] == 1, base_rtt + 150.0, base_rtt)
        df["rtt_ms"] = np.round(base_rtt, 2)
        df["rtt_variance_ms"] = np.round(df["rtt_ms"].rolling(10, min_periods=1).std().fillna(3.0), 2)

        # Handover interruption time in ms
        df["interruption_time_ms"] = np.where(df["is_handover"] == 1, 80.0, 0.0)

        df["source_dataset"] = os.path.basename(f)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

class TraceLoader:
    def __init__(self, trace_source: Union[str, Path, pd.DataFrame]):
        if isinstance(trace_source, (str, Path)):
            path = Path(trace_source)
            if not path.exists():
                raise FileNotFoundError(f"Trace file does not exist: {path}")
            self.df = pd.read_csv(path)
        elif isinstance(trace_source, pd.DataFrame):
            self.df = trace_source.copy()
        else:
            raise TypeError("trace_source must be a file path or pandas DataFrame")

        self._validate_and_clean()

    def _validate_and_clean(self):
        # Strip column whitespace
        self.df.columns = [c.strip() for c in self.df.columns]

        # Check if this is the Handover Dataset (ver 2) raw schema
        if "Serving Cell RSRP" in self.df.columns and "rsrp_dbm" not in self.df.columns:
            rsrp_raw = pd.to_numeric(self.df["Serving Cell RSRP"], errors="coerce").fillna(50)
            self.df["rsrp_dbm"] = rsrp_raw - 140.0

        if "Handover Trigger" in self.df.columns and "is_handover" not in self.df.columns:
            self.df["is_handover"] = pd.to_numeric(self.df["Handover Trigger"], errors="coerce").fillna(0).astype(int)

        if "Speed (km/h)" in self.df.columns and "velocity_kmh" not in self.df.columns:
            self.df["velocity_kmh"] = pd.to_numeric(self.df["Speed (km/h)"], errors="coerce").fillna(30.0).clip(lower=5.0)

        # Validate required columns or create sensible fallbacks
        if "timestamp_s" not in self.df.columns:
            if "Timestamp" in self.df.columns:
                self.df["timestamp_s"] = np.arange(len(self.df)) * 0.5
            else:
                self.df["timestamp_s"] = np.arange(len(self.df)) * 0.05

        if "velocity_kmh" not in self.df.columns:
            self.df["velocity_kmh"] = 300.0

        if "rsrp_dbm" not in self.df.columns:
            self.df["rsrp_dbm"] = -80.0

        if "signal_fade_rate" not in self.df.columns:
            rsrp = self.df["rsrp_dbm"].values
            ts = self.df["timestamp_s"].values
            dt = np.diff(ts, prepend=ts[0] - 0.5)
            dt = np.where(dt <= 0, 0.5, dt)
            self.df["signal_fade_rate"] = np.round(-np.diff(rsrp, prepend=rsrp[0]) / dt, 2)

        if "rtt_ms" not in self.df.columns:
            self.df["rtt_ms"] = 30.0

        if "rtt_variance_ms" not in self.df.columns:
            self.df["rtt_variance_ms"] = self.df["rtt_ms"].rolling(10, min_periods=1).std().fillna(2.0)

        if "packet_loss_ratio" not in self.df.columns:
            self.df["packet_loss_ratio"] = 0.01

        if "is_handover" not in self.df.columns:
            self.df["is_handover"] = 0

        if "interruption_time_ms" not in self.df.columns:
            self.df["interruption_time_ms"] = np.where(self.df["is_handover"] == 1, 80.0, 0.0)

        if "handover_count" not in self.df.columns:
            self.df["handover_count"] = self.df["is_handover"].cumsum()

        self.df = self.df.sort_values(by="timestamp_s").reset_index(drop=True)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def iterate(self) -> Generator[Dict[str, Any], None, None]:
        for _, row in self.df.iterrows():
            yield row.to_dict()
