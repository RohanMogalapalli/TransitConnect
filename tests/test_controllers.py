"""
Unit tests for sliding window, dynamic FEC, and joint controller in TransitConnect.
"""

import pytest
from transitconnect.controller.sliding_window import AdaptiveSlidingWindow
from transitconnect.controller.dynamic_fec import DynamicFecModel
from transitconnect.controller.joint_controller import JointAdaptiveController
from transitconnect.risk.weighted_sum import RiskState

def test_adaptive_sliding_window_bounds():
    w = AdaptiveSlidingWindow(w0=64, w_min=4, alpha=0.85)
    # H = 0 -> maximum window
    assert w.compute_window(0.0) == 64
    # H = 1.0 with alpha=0.85 -> floor(64 * 0.15) = 9
    assert w.compute_window(1.0) == 9
    
    # When alpha=1.0, W_0 * (1 - H) drops below w_min -> clamped to w_min floor (4)
    w_floor = AdaptiveSlidingWindow(w0=64, w_min=4, alpha=1.0)
    assert w_floor.compute_window(1.0) == 4
    
    # Intermediate H
    w_mid = w.compute_window(0.5)
    assert 4 <= w_mid <= 64

def test_dynamic_fec_bounds():
    f = DynamicFecModel(e0=0.05, e_max=0.60, beta=0.55)
    # H = 0 -> minimum baseline redundancy
    assert round(f.compute_redundancy(0.0), 2) == 0.05
    # H = 1 -> capped at e_max
    assert round(f.compute_redundancy(1.0), 2) == 0.60
    # Intermediate H
    e_mid = f.compute_redundancy(0.5)
    assert 0.05 <= e_mid <= 0.60

def test_joint_controller_coordination_and_recovery():
    ctrl = JointAdaptiveController(w0=64, w_min=4, e0=0.05, e_max=0.60, recovery_step_w=4, recovery_step_e=0.05)
    
    # 1. Low risk state
    w, e, state = ctrl.update_control(0.1, RiskState.LOW)
    assert w == 64
    assert e == 0.05
    assert state == RiskState.LOW

    # 2. Trigger high risk (Handover approaching)
    w_hi, e_hi, state_hi = ctrl.update_control(0.85, RiskState.HIGH)
    assert w_hi < 64
    assert e_hi > 0.05
    assert state_hi == RiskState.HIGH

    # 3. Handover completes -> Risk drops to low
    # Must enter RECOVERY state rather than snapping instantly to w0
    w_rec, e_rec, state_rec = ctrl.update_control(0.1, RiskState.LOW)
    assert state_rec == "RECOVERY"
    assert w_rec == w_hi + 4  # Incremental recovery step
    assert e_rec == max(0.05, round(e_hi - 0.05, 4))
