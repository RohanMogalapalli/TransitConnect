"""
Baseline 5: Reactive Loss-Based TCP Protocol (AIMD / Reno-like).
Implements classical Additive Increase, Multiplicative Decrease (AIMD) congestion control.
- In the absence of loss: increases congestion window by 1 packet per round-trip.
- Upon observed loss: cuts congestion window by half: W = max(W_min, floor(W / 2)).
- Zero FEC redundancy (E = 0).
- Purely reactive: reacts only after packet drops have manifested in the data stream.
"""

import math
from typing import Dict, Any, List
import numpy as np
from ..controller.dynamic_fec import SystematicBlockErasureCoder, Packet

class ReactiveTcpProtocol:
    def __init__(self, w_init: int = 32, w_max: int = 64, w_min: int = 2, block_size_k: int = 16):
        self.w_current = int(w_init)
        self.w_max = int(w_max)
        self.w_min = int(w_min)
        self.erasure_coder = SystematicBlockErasureCoder(block_size_k=block_size_k)

    def transmit_and_evaluate_block(
        self,
        num_packets: int,
        channel_loss_ratio: float,
        rtt_ms: float
    ) -> Dict[str, Any]:
        packets_to_send = min(num_packets, self.w_current)
        if packets_to_send <= 0:
            return {
                "window_size": self.w_current, "fec_redundancy": 0.0,
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

        # Reactive AIMD window adjustment
        if unrecoverable > 0:
            # Multiplicative Decrease upon loss
            self.w_current = max(self.w_min, math.floor(self.w_current / 2))
        else:
            # Additive Increase
            self.w_current = min(self.w_max, self.w_current + 1)

        return {
            "window_size": self.w_current,
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
