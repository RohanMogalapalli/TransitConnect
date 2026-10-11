"""
Unit tests for telemetry monitoring and feature extraction in TransitConnect.
"""

import pytest
from transitconnect.monitoring.filters import EwmaFilter
from transitconnect.monitoring.features import FeatureExtractor, NormalizedFeatures
from transitconnect.monitoring.monitor import NetworkMobilityMonitor

def test_ewma_filter():
    f = EwmaFilter(alpha=0.5, initial_value=10.0)
    assert f.get() == 10.0
    val = f.update(20.0)
    assert val == 15.0  # 0.5 * 20 + 0.5 * 10
    val = f.update(20.0)
    assert val == 17.5

def test_feature_extractor_bounds():
    ext = FeatureExtractor()
    # Test optimal condition
    feat_best = ext.normalize(fade_rate=0.0, rtt_var=0.0, loss_ratio=0.0, rtt=20.0, velocity_kmh=100.0, rsrp_dbm=-55.0)
    assert 0.0 <= feat_best.F <= 1.0
    assert 0.0 <= feat_best.V <= 1.0
    assert 0.0 <= feat_best.P <= 1.0
    assert 0.0 <= feat_best.R <= 1.0
    assert 0.0 <= feat_best.U <= 1.0
    assert 0.0 <= feat_best.S <= 1.0
    assert feat_best.P == 0.0
    assert feat_best.S == 0.0

    # Test extreme degradation
    feat_worst = ext.normalize(fade_rate=100.0, rtt_var=100.0, loss_ratio=1.0, rtt=300.0, velocity_kmh=500.0, rsrp_dbm=-130.0)
    assert feat_worst.F == 1.0
    assert feat_worst.V == 1.0
    assert feat_worst.P == 1.0
    assert feat_worst.R == 1.0
    assert feat_worst.U == 1.0
    assert feat_worst.S == 1.0

def test_network_mobility_monitor():
    monitor = NetworkMobilityMonitor()
    sample = {
        "rsrp_dbm": -70.0,
        "signal_fade_rate": 5.0,
        "rtt_ms": 35.0,
        "rtt_variance_ms": 3.0,
        "packet_loss_ratio": 0.02,
        "velocity_kmh": 300.0
    }
    features, smoothed = monitor.process_sample(sample)
    assert isinstance(features, NormalizedFeatures)
    assert "rsrp_dbm" in smoothed
    assert smoothed["velocity_kmh"] == 300.0
