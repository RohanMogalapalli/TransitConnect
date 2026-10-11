"""
Unit tests for mathematical and ML handover-risk models in TransitConnect.
"""

import pytest
import pandas as pd
from transitconnect.monitoring.features import NormalizedFeatures
from transitconnect.risk.weighted_sum import WeightedSumRiskModel, RiskState
from transitconnect.risk.dataset_gen import DatasetGenerator
from transitconnect.risk.ml_model import MLRiskPredictor

def test_weighted_sum_risk_model():
    model = WeightedSumRiskModel(w1=0.3, w2=0.2, w3=0.25, w4=0.15, w5=0.10, t1=0.35, t2=0.70)
    
    # Low risk input
    low_feat = NormalizedFeatures(F=0.1, V=0.1, P=0.01, R=0.1, U=0.5, S=0.1)
    h_low, state_low = model.compute_risk(low_feat)
    assert 0.0 <= h_low < 0.35
    assert state_low == RiskState.LOW

    # High risk input
    high_feat = NormalizedFeatures(F=0.9, V=0.8, P=0.7, R=0.8, U=1.0, S=0.9)
    h_high, state_high = model.compute_risk(high_feat)
    assert h_high >= 0.70
    assert state_high == RiskState.HIGH

    # Moderate risk input
    mod_feat = NormalizedFeatures(F=0.5, V=0.5, P=0.4, R=0.5, U=0.8, S=0.5)
    h_mod, state_mod = model.compute_risk(mod_feat)
    assert 0.35 <= h_mod < 0.70
    assert state_mod == RiskState.MODERATE

def test_dataset_generator():
    gen = DatasetGenerator()
    trace = pd.DataFrame({
        "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0],
        "rsrp_dbm": [-60.0, -80.0, -95.0, -100.0, -65.0],
        "signal_fade_rate": [0.0, 20.0, 15.0, 5.0, -35.0],
        "rtt_ms": [25.0, 35.0, 80.0, 150.0, 30.0],
        "rtt_variance_ms": [1.0, 4.0, 15.0, 30.0, 2.0],
        "packet_loss_ratio": [0.01, 0.05, 0.20, 0.80, 0.02],
        "velocity_kmh": [300.0, 300.0, 300.0, 300.0, 300.0],
        "is_handover": [0, 0, 0, 1, 0]
    })
    ds = gen.build_dataset_from_dataframe(trace)
    assert len(ds) == 5
    assert "risk_label" in ds.columns
    assert "risk_score_target" in ds.columns
    # Handover at index 3 must have high risk
    assert ds.loc[3, "risk_label"] == 2

def test_ml_risk_predictor_train_and_predict():
    gen = DatasetGenerator()
    # Create synthetic dataset with multiple handovers
    ts = [i * 0.1 for i in range(100)]
    trace = pd.DataFrame({
        "timestamp_s": ts,
        "rsrp_dbm": [-60.0 - (i % 20) * 2 for i in range(100)],
        "signal_fade_rate": [10.0 if (i % 20) > 10 else 0.0 for i in range(100)],
        "rtt_ms": [30.0 + (i % 20) * 3 for i in range(100)],
        "rtt_variance_ms": [2.0 + (i % 20) * 0.5 for i in range(100)],
        "packet_loss_ratio": [0.8 if (i % 20) == 19 else 0.02 for i in range(100)],
        "velocity_kmh": [300.0 for _ in range(100)],
        "is_handover": [1 if (i % 20) == 19 else 0 for i in range(100)]
    })
    ds = gen.build_dataset_from_dataframe(trace)
    
    predictor = MLRiskPredictor()
    metrics = predictor.train_models(ds, model_type="rf")
    assert metrics["accuracy"] > 0.70
    assert "feature_importances" in metrics

    # Test inference
    test_feat = NormalizedFeatures(F=0.8, V=0.8, P=0.8, R=0.8, U=0.9, S=0.8)
    score, state = predictor.predict(test_feat)
    assert 0.0 <= score <= 1.0
    assert state in [RiskState.LOW, RiskState.MODERATE, RiskState.HIGH]
