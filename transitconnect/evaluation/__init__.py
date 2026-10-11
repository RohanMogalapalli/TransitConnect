"""
Evaluation module for TransitConnect.
"""

from .metrics import (
    calculate_pdr,
    calculate_goodput_mbps,
    calculate_throughput_mbps,
    calculate_fec_overhead_percent,
    aggregate_protocol_metrics
)
from .benchmark import BenchmarkSuite, SPEED_PROFILES_KMH
from .visualizer import BenchmarkVisualizer

__all__ = [
    "calculate_pdr",
    "calculate_goodput_mbps",
    "calculate_throughput_mbps",
    "calculate_fec_overhead_percent",
    "aggregate_protocol_metrics",
    "BenchmarkSuite",
    "SPEED_PROFILES_KMH",
    "BenchmarkVisualizer"
]
