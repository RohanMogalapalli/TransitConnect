"""
Joint Adaptive Transmission Controller for TransitConnect.
Coordinates the Adaptive Sliding Window (W) and Dynamic FEC (E)
based on the common handover-risk state, and provides gradual recovery smoothing
to avoid threshold oscillation following handover completion.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from .sliding_window import AdaptiveSlidingWindow
from .dynamic_fec import DynamicFecModel, SystematicBlockErasureCoder, Packet
from ..risk.weighted_sum import RiskState

class JointAdaptiveController:
    def __init__(
        self,
        w0: int = 64,
        w_min: int = 4,
        alpha: float = 0.85,
        e0: float = 0.05,
        e_max: float = 0.60,
        beta: float = 0.55,
        t1: float = 0.35,
        t2: float = 0.70,
        recovery_step_w: int = 4,
        recovery_step_e: float = 0.05,
        block_size_k: int = 16
    ):
        self.window_model = AdaptiveSlidingWindow(w0=w0, w_min=w_min, alpha=alpha)
        self.fec_model = DynamicFecModel(e0=e0, e_max=e_max, beta=beta)
        self.erasure_coder = SystematicBlockErasureCoder(block_size_k=block_size_k)

        self.t1 = t1
        self.t2 = t2
        self.recovery_step_w = recovery_step_w
        self.recovery_step_e = recovery_step_e
        self.block_size_k = block_size_k

        # Current active transmission configuration
        self.current_w = w0
        self.current_e = e0
        self.in_recovery = False
        self.prev_state = RiskState.LOW

    def update_control(self, risk_score_h: float, risk_state: str) -> Tuple[int, float, str]:
        """
        Updates (W, E) configuration given the current risk score and qualitative state.
        Applies gradual recovery smoothing when emerging from high/moderate risk.
        """
        target_w = self.window_model.compute_window(risk_score_h)
        target_e = self.fec_model.compute_redundancy(risk_score_h)

        if risk_state == RiskState.HIGH:
            # High risk: immediate conservative adaptation
            self.current_w = target_w
            self.current_e = target_e
            self.in_recovery = False
            effective_state = RiskState.HIGH

        elif risk_state == RiskState.MODERATE:
            # Moderate risk: proportional adaptation
            self.current_w = target_w
            self.current_e = target_e
            self.in_recovery = False
            effective_state = RiskState.MODERATE

        else:
            # Low risk state
            if self.prev_state in (RiskState.HIGH, RiskState.MODERATE) or self.in_recovery:
                # Gradual recovery smoothing to avoid oscillation
                if self.current_w < self.window_model.w0 or self.current_e > self.fec_model.e0:
                    self.in_recovery = True
                    self.current_w = min(self.window_model.w0, self.current_w + self.recovery_step_w)
                    self.current_e = max(self.fec_model.e0, self.current_e - self.recovery_step_e)
                    effective_state = "RECOVERY"
                else:
                    self.in_recovery = False
                    self.current_w = self.window_model.w0
                    self.current_e = self.fec_model.e0
                    effective_state = RiskState.LOW
            else:
                self.current_w = self.window_model.w0
                self.current_e = self.fec_model.e0
                self.in_recovery = False
                effective_state = RiskState.LOW

        self.prev_state = risk_state
        return self.current_w, round(self.current_e, 4), effective_state

    def transmit_and_evaluate_block(
        self,
        num_packets: int,
        channel_loss_ratio: float,
        rtt_ms: float
    ) -> Dict[str, Any]:
        """
        Simulates transmission of packets subject to sliding window capacity,
        dynamic FEC encoding, channel packet loss, and erasure recovery.
        """
        # Constrain packets sent in this cycle by current window W
        packets_to_send = min(num_packets, self.current_w)
        if packets_to_send <= 0:
            return {
                "sent_data": 0, "sent_parity": 0, "total_sent": 0,
                "delivered_data": 0, "lost_unrecovered": 0, "recovered_by_fec": 0,
                "fec_overhead_ratio": 0.0, "effective_goodput_ratio": 0.0
            }

        # Generate mock payloads
        data_payloads = [f"PKT_{i}".encode("utf-8") for i in range(packets_to_send)]
        encoded_block = self.erasure_coder.encode_block(data_payloads, self.current_e)
        
        total_sent = len(encoded_block)
        parity_count = total_sent - packets_to_send

        # Simulate channel packet drops based on channel_loss_ratio
        received_packets: List[Packet] = []
        for pkt in encoded_block:
            # Independent channel erasure probability
            if np.random.rand() >= channel_loss_ratio:
                received_packets.append(pkt)

        # Decode block at receiver
        success, recovered, unrecoverable = self.erasure_coder.decode_block(
            received_packets, k=packets_to_send, total_transmitted=total_sent
        )

        delivered_data = packets_to_send - unrecoverable
        fec_overhead_ratio = parity_count / total_sent if total_sent > 0 else 0.0

        return {
            "window_size": self.current_w,
            "fec_redundancy": self.current_e,
            "sent_data": packets_to_send,
            "sent_parity": parity_count,
            "total_sent": total_sent,
            "delivered_data": delivered_data,
            "lost_unrecovered": unrecoverable,
            "recovered_by_fec": recovered,
            "fec_overhead_ratio": round(fec_overhead_ratio, 4),
            "effective_goodput_ratio": round(delivered_data / total_sent, 4) if total_sent > 0 else 0.0
        }

    def reset(self):
        self.current_w = self.window_model.w0
        self.current_e = self.fec_model.e0
        self.in_recovery = False
        self.prev_state = RiskState.LOW
