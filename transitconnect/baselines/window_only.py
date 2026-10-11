"""
Baseline 3: Adaptive Window Only Protocol (Ablation).
Adapts sliding-window size W(H) based on handover risk, but applies no Forward Error Correction (E = 0).
Isolates the performance contribution of flow-control adaptation without error correction.
"""

from typing import Dict, Any, List
import numpy as np
from ..controller.sliding_window import AdaptiveSlidingWindow
from ..controller.dynamic_fec import SystematicBlockErasureCoder, Packet

class WindowOnlyProtocol:
    def __init__(self, w0: int = 64, w_min: int = 4, alpha: float = 0.85, block_size_k: int = 16):
        self.window_model = AdaptiveSlidingWindow(w0=w0, w_min=w_min, alpha=alpha)
        self.erasure_coder = SystematicBlockErasureCoder(block_size_k=block_size_k)

    def transmit_and_evaluate_block(
        self,
        num_packets: int,
        channel_loss_ratio: float,
        rtt_ms: float,
        risk_score_h: float = 0.0
    ) -> Dict[str, Any]:
        current_w = self.window_model.compute_window(risk_score_h)
        packets_to_send = min(num_packets, current_w)
        if packets_to_send <= 0:
            return {
                "window_size": current_w, "fec_redundancy": 0.0,
                "sent_data": 0, "sent_parity": 0, "total_sent": 0,
                "delivered_data": 0, "lost_unrecovered": 0, "recovered_by_fec": 0,
                "fec_overhead_ratio": 0.0, "effective_goodput_ratio": 0.0
            }

        data_payloads = [f"PKT_{i}".encode("utf-8") for i in range(packets_to_send)]
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
            "window_size": current_w,
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
