"""
Simulation module for TransitConnect.
Supports both native ns-3 C++ execution in WSL2, high-fidelity physics-based Python corridor simulation,
and real-world drive-test trace ingestion (Handover Dataset ver 2).
"""

from .ns3_runner import run_ns3_simulation, is_ns3_available
from .corridor_sim import HsrCorridorSimulator
from .trace_loader import TraceLoader, load_real_handover_dataset

__all__ = [
    "run_ns3_simulation",
    "is_ns3_available",
    "HsrCorridorSimulator",
    "TraceLoader",
    "load_real_handover_dataset"
]
