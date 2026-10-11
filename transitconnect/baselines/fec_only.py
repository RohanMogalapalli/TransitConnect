"""
Baseline 4: Adaptive FEC Only Protocol (Ablation).
Adapts FEC redundancy E(H) dynamically with predicted handover risk,
but maintains a fixed sliding-window size W0.
Isolates the performance contribution of dynamic error correction without window flow control.
"""

from typing import Dict, Any, List
import numpy as np
from ..controller.dynamic_fec import DynamicFecModel, SystematicBlockErasureCoder, Packet

class FecOnlyProtocol:
    def __init__(self, w0: int = 64, e0: float = 0.05, e_max: float = 0.60, beta: float = 0.55, block_size_k: int = 16):
        self.w0 = int(w0)
        self.fec_model = DynamicFecModel(e0=e0, e_max=e_max, beta=beta)
        self.erasure_coder = SystematicBlockErasureCoder(block_size_k=block_size_k)

    def transmit_and_evaluate_block(
        self,
        num_packets: int,
        channel_loss_ratio: float,
        rtt_ms: float,
        risk_score_h: float = 0.0
    ) -> Dict[str, Any]:
        current_e = self.fec_model.compute_redundancy(risk_score_h)
        packets_to_send = min(num_packets, self.w0)
        if packets_to_send <= 0:
            return {
                "window_size": self.w0, "fec_redundancy": current_e,
                "sent_data": 0, "sent_parity": 0, "total_sent": 0,
                "delivered_data": 0, "lost_unrecovered": 0, "recovered_by_fec": 0,
                "fec_overhead_ratio": 0.0, "effective_goodput_ratio": 0.0
            }

        data_payloads = [f"PKT_{i}".encode("utf-8") for i in range(packets_to_send)]
        encoded_block = self.erasure_coder.encode_block(data_payloads, redundancy_ratio=current_e)
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
            "fec_redundancy": current_e,
            "sent_data": packets_to_send,
            "sent_parity": parity_count,
            "total_sent": total_sent,
            "delivered_data": delivered_data,
            "lost_unrecovered": unrecoverable,
            "recovered_by_fec": recovered,
            "fec_overhead_ratio": round(parity_count / total_sent, 4) if total_sent > 0 else 0.0,
            "effective_goodput_ratio": round(delivered_data / total_sent, 4) if total_sent > 0 else 0.0
        }
