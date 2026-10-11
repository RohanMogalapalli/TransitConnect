"""
Unit tests for simulation modules in TransitConnect.
"""

import pytest
import pandas as pd
from transitconnect.simulation.corridor_sim import HsrCorridorSimulator
from transitconnect.simulation.trace_loader import TraceLoader
from transitconnect.simulation.ns3_runner import is_ns3_available

def test_corridor_sim_basic_execution():
    sim = HsrCorridorSimulator(
        speed_kmh=300.0,
        num_bs=4,
        bs_spacing_m=1500.0,
        sim_time_s=10.0,
        sample_interval_s=0.1,
        seed=42
    )
    df = sim.run()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "rsrp_dbm" in df.columns
    assert "signal_fade_rate" in df.columns
    assert "rtt_ms" in df.columns
    assert "packet_loss_ratio" in df.columns
    assert "is_handover" in df.columns
    # Speed check
    assert (df["velocity_kmh"] == 300.0).all()

def test_corridor_sim_handover_triggering():
    # Long enough simulation to trigger handovers across base stations
    sim = HsrCorridorSimulator(
        speed_kmh=350.0,
        num_bs=4,
        bs_spacing_m=1200.0,
        sim_time_s=40.0,
        sample_interval_s=0.05,
        seed=123
    )
    df = sim.run()
    assert df["handover_count"].max() >= 1

def test_trace_loader():
    data = {
        "timestamp_s": [0.0, 0.1, 0.2],
        "rsrp_dbm": [-60.0, -65.0, -70.0],
        "signal_fade_rate": [0.0, 50.0, 50.0],
        "rtt_ms": [25.0, 30.0, 35.0],
        "rtt_variance_ms": [1.0, 2.0, 3.0],
        "packet_loss_ratio": [0.01, 0.02, 0.05],
        "velocity_kmh": [300.0, 300.0, 300.0]
    }
    df = pd.DataFrame(data)
    loader = TraceLoader(df)
    res_df = loader.get_dataframe()
    assert len(res_df) == 3
    rows = list(loader.iterate())
    assert len(rows) == 3
    assert rows[0]["rsrp_dbm"] == -60.0

def test_ns3_availability_check():
    # Verify the function executes without exception
    available = is_ns3_available()
    assert isinstance(available, bool)
