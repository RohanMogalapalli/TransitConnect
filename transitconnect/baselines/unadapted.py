"""
Baseline 1: Unadapted / Fixed Window Protocol.
Maintains a constant sliding-window size W0 with zero FEC redundancy (E = 0).
Represents a legacy transport configuration with no handover adaptation.
"""

from typing import Dict, Any, List
import numpy as np
from ..controller.dynamic_fec import SystematicBlockErasureCoder, Packet

class UnadaptedProtocol:
    def __init__(self, w0: int = 64, block_size_k: int = 16):
        self.w0 = int(w0)
        self.erasure_coder = SystematicBlockErasureCoder(block_size_k=block_size_k)

    def transmit_and_evaluate_block(
        self,
        num_packets: int,
        channel_loss_ratio: float,
        rtt_ms: float
    ) -> Dict[str, Any]:
        packets_to_send = min(num_packets, self.w0)
        if packets_to_send <= 0:
            return {
                "window_size": self.w0, "fec_redundancy": 0.0,
                "sent_data": 0, "sent_parity": 0, "total_sent": 0,
                "delivered_data": 0, "lost_unrecovered": 0, "recovered_by_fec": 0,
                "fec_overhead_ratio": 0.0, "effective_goodput_ratio": 0.0
            }

        data_payloads = [f"PKT_{i}".encode("utf-8") for i in range(packets_to_send)]
        # Redundancy is 0
        encoded_block = self.erasure_coder.encode_block(data_payloads, redundancy_ratio=0.0)

        received_packets: List[Packet] = []
        for pkt in encoded_block:
            if np.random.rand() >= channel_loss_ratio:
                received_packets.append(pkt)

        success, recovered, unrecoverable = self.erasure_coder.decode_block(
            received_packets, k=packets_to_send, total_transmitted=len(encoded_block)
        )
        delivered_data = packets_to_send - unrecoverable

        return {
            "window_size": self.w0,
            "fec_redundancy": 0.0,
            "sent_data": packets_to_send,
            "sent_parity": 0,
            "total_sent": packets_to_send,
            "delivered_data": delivered_data,
            "lost_unrecovered": unrecoverable,
            "recovered_by_fec": 0,
            "fec_overhead_ratio": 0.0,
            "effective_goodput_ratio": round(delivered_data / packets_to_send, 4) if packets_to_send > 0 else 0.0
        }
