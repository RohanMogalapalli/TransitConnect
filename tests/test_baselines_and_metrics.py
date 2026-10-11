"""
Unit tests for baseline protocols and evaluation metrics.
"""

import pytest
from transitconnect.baselines import (
    UnadaptedProtocol,
    FixedFecProtocol,
    WindowOnlyProtocol,
    FecOnlyProtocol,
    ReactiveTcpProtocol
)
from transitconnect.evaluation.metrics import (
    calculate_pdr,
    calculate_goodput_mbps,
    calculate_fec_overhead_percent,
    aggregate_protocol_metrics
)

def test_unadapted_protocol():
    proto = UnadaptedProtocol(w0=32)
    res = proto.transmit_and_evaluate_block(num_packets=32, channel_loss_ratio=0.0, rtt_ms=30.0)
    assert res["delivered_data"] == 32
    assert res["fec_overhead_ratio"] == 0.0

def test_fixed_fec_protocol():
    proto = FixedFecProtocol(w0=32, static_fec_ratio=0.20)
    res = proto.transmit_and_evaluate_block(num_packets=32, channel_loss_ratio=0.10, rtt_ms=30.0)
    assert res["sent_parity"] > 0
    assert res["fec_overhead_ratio"] > 0.0

def test_reactive_tcp_aimd():
    proto = ReactiveTcpProtocol(w_init=16, w_max=64, w_min=2)
    # Success without loss -> Additive Increase
    res = proto.transmit_and_evaluate_block(num_packets=16, channel_loss_ratio=0.0, rtt_ms=30.0)
    assert proto.w_current == 17

    # Severe loss -> Multiplicative Decrease
    res = proto.transmit_and_evaluate_block(num_packets=17, channel_loss_ratio=1.0, rtt_ms=100.0)
    assert proto.w_current <= 8

def test_metrics_calculations():
    pdr = calculate_pdr(delivered_data=95, sent_data=100)
    assert pdr == 95.0

    goodput = calculate_goodput_mbps(delivered_data=1000, total_duration_s=10.0, packet_size_bytes=1400)
    assert goodput > 0.0

    overhead = calculate_fec_overhead_percent(sent_parity=20, total_sent=100)
    assert overhead == 20.0

    step_records = [
        {"sent_data": 64, "sent_parity": 10, "delivered_data": 64, "lost_unrecovered": 0, "recovered_by_fec": 0, "rtt_ms": 30.0, "interruption_time_ms": 0.0},
        {"sent_data": 32, "sent_parity": 15, "delivered_data": 30, "lost_unrecovered": 2, "recovered_by_fec": 5, "rtt_ms": 50.0, "interruption_time_ms": 50.0}
    ]
    summary = aggregate_protocol_metrics(step_records, sim_time_s=1.0, speed_kmh=300.0, protocol_name="Test")
    assert summary["pdr_percent"] == round((94 / 96) * 100, 2)
    assert summary["lost_unrecovered_packets"] == 2
