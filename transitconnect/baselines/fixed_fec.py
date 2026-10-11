"""
Baseline 2: Fixed FEC Protocol.
Maintains a constant sliding-window size W0 with a statically configured FEC redundancy (e.g. 15%).
Represents a statically protected protocol without handover risk awareness.
"""

from typing import Dict, Any, List
import numpy as np
from ..controller.dynamic_fec import SystematicBlockErasureCoder, Packet

class FixedFecProtocol:
    def __init__(self, w0: int = 64, static_fec_ratio: float = 0.15, block_size_k: int = 16):
        self.w0 = int(w0)
        self.static_fec = float(static_fec_ratio)
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
                "window_size": self.w0, "fec_redundancy": self.static_fec,
                "sent_data": 0, "sent_parity": 0, "total_sent": 0,
                "delivered_data": 0, "lost_unrecovered": 0, "recovered_by_fec": 0,
                "fec_overhead_ratio": 0.0, "effective_goodput_ratio": 0.0
            }

        data_payloads = [f"PKT_{i}".encode("utf-8") for i in range(packets_to_send)]
        encoded_block = self.erasure_coder.encode_block(data_payloads, redundancy_ratio=self.static_fec)
        total_sent = len(encoded_block)
        parity_count = total_sent - packets_to_send

        received_packets: List[Packet] = []
        for pkt in encoded_block:
            if np.random.rand() >= channel_loss_ratio:
                received_packets.append(pkt)

        success, recovered, unrecoverable = self.erasure_coder.decode_block(
            received_packets, k=packets_to_send, total_transmitted=total_sent
        )
        delivered_data = packets_to_send - unrecoverable

        return {
            "window_size": self.w0,
            "fec_redundancy": self.static_fec,
            "sent_data": packets_to_send,
            "sent_parity": parity_count,
            "total_sent": total_sent,
            "delivered_data": delivered_data,
            "lost_unrecovered": unrecoverable,
            "recovered_by_fec": recovered,
            "fec_overhead_ratio": round(parity_count / total_sent, 4) if total_sent > 0 else 0.0,
            "effective_goodput_ratio": round(delivered_data / total_sent, 4) if total_sent > 0 else 0.0
        }
