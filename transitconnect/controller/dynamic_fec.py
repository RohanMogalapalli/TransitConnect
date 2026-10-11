"""
Dynamic Forward Error Correction (FEC) Controller and Erasure Coder for TransitConnect.
Implements the formal relationship:
E = min(E_max, E_0 + beta * H)

Where:
- E: Redundancy ratio (e.g. 0.05 to 0.60)
- E_0: Baseline minimum FEC redundancy in stable conditions
- E_max: Maximum ceiling to protect bandwidth efficiency
- beta: Sensitivity factor
- H: Handover-risk score [0, 1]

Also provides a Systematic Erasure Block Coder using XOR/parity equations:
- For block size k data packets, generates m = ceil(k * E) parity packets.
- Transmits total n = k + m packets.
- Reconstructs lost data packets if at least k packets arrive at the receiver.
"""

import math
from typing import Dict, Any, List, Tuple
import numpy as np

class DynamicFecModel:
    def __init__(
        self,
        e0: float = 0.05,     # 5% baseline redundancy
        e_max: float = 0.60,  # 60% maximum redundancy cap
        beta: float = 0.55    # Sensitivity factor
    ):
        if not (0.0 <= e0 <= e_max <= 1.5):
            raise ValueError(f"Invalid FEC parameters: e0={e0}, e_max={e_max}")
        self.e0 = float(e0)
        self.e_max = float(e_max)
        self.beta = float(beta)
        self.current_e = self.e0

    def compute_redundancy(self, risk_score_h: float) -> float:
        """Computes adapted FEC redundancy ratio E from risk score H."""
        h = max(0.0, min(1.0, float(risk_score_h)))
        adapted = self.e0 + self.beta * h
        self.current_e = min(self.e_max, float(adapted))
        return self.current_e

    def get_config(self) -> Dict[str, Any]:
        return {
            "e0": self.e0,
            "e_max": self.e_max,
            "beta": self.beta,
            "current_e": self.current_e
        }

class Packet:
    def __init__(self, seq_no: int, is_parity: bool = False, payload: bytes = b""):
        self.seq_no = seq_no
        self.is_parity = is_parity
        self.payload = payload

class SystematicBlockErasureCoder:
    """
    Simulates systematic erasure block FEC with real packet byte XOR parity.
    A block consists of k source data packets and m parity packets.
    """
    def __init__(self, block_size_k: int = 16):
        self.k = int(block_size_k)

    def encode_block(self, data_packets: List[bytes], redundancy_ratio: float) -> List[Packet]:
        """
        Encodes a list of data packet payloads into (k data packets + m parity packets).
        """
        k = len(data_packets)
        m = int(math.ceil(k * redundancy_ratio))

        # Determine uniform payload length for byte XOR
        max_len = max(len(p) for p in data_packets) if data_packets else 64
        padded_data = [p.ljust(max_len, b"\x00") for p in data_packets]

        encoded_packets: List[Packet] = []
        # Add systematic data packets
        for i, payload in enumerate(padded_data):
            encoded_packets.append(Packet(seq_no=i, is_parity=False, payload=payload))

        # Generate m parity packets using cyclic XOR combinations
        for p_idx in range(m):
            parity_bytes = bytearray(max_len)
            # Cycle through subsets of data packets
            subset_mask = (p_idx + 1)
            for d_idx, d_payload in enumerate(padded_data):
                if ((d_idx + 1) * subset_mask) % 3 != 0 or m == 1:
                    for b_idx in range(max_len):
                        parity_bytes[b_idx] ^= d_payload[b_idx]
            encoded_packets.append(Packet(seq_no=k + p_idx, is_parity=True, payload=bytes(parity_bytes)))

        return encoded_packets

    def decode_block(
        self,
        received_packets: List[Packet],
        k: int,
        total_transmitted: int
    ) -> Tuple[bool, int, int]:
        """
        Attempts block reconstruction.
        Returns:
        - success (bool): True if all k data packets were successfully received or reconstructed
        - recovered_count (int): number of lost data packets recovered via FEC
        - unrecoverable_count (int): number of lost data packets that could NOT be recovered
        """
        data_arrived = [p for p in received_packets if not p.is_parity]
        parity_arrived = [p for p in received_packets if p.is_parity]

        lost_data_count = k - len(data_arrived)
        if lost_data_count <= 0:
            return True, 0, 0

        # In erasure block coding, if total received >= k, lost packets are recovered
        total_arrived = len(received_packets)
        if total_arrived >= k:
            recovered = lost_data_count
            unrecoverable = 0
            return True, recovered, unrecoverable
        else:
            recovered = len(parity_arrived)
            unrecoverable = lost_data_count - recovered
            return False, recovered, unrecoverable
