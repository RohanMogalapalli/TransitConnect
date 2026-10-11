"""
High-Speed Rail Corridor Simulator (Python Engine).
Provides physics-based wireless channel modeling, multi-cell macro deployment,
3GPP A3 Handover hysteresis and Time-to-Trigger (TTT), and packet degradation dynamics.
Produces identical schema and telemetry semantics to the ns-3 C++ simulation.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

class HsrCorridorSimulator:
    def __init__(
        self,
        speed_kmh: float = 300.0,
        num_bs: int = 5,
        bs_spacing_m: float = 2000.0,
        sim_time_s: Optional[float] = None,
        sample_interval_s: float = 0.05,
        hysteresis_db: float = 3.0,
        time_to_trigger_s: float = 0.256,
        handover_delay_s: float = 0.080,
        seed: int = 42
    ):
        self.speed_kmh = float(speed_kmh)
        self.speed_ms = self.speed_kmh * (1000.0 / 3600.0)
        self.num_bs = int(num_bs)
        self.bs_spacing_m = float(bs_spacing_m)
        self.sample_interval_s = float(sample_interval_s)
        self.hysteresis_db = float(hysteresis_db)
        self.time_to_trigger_s = float(time_to_trigger_s)
        self.handover_delay_s = float(handover_delay_s)
        self.seed = int(seed)

        if sim_time_s is None:
            total_dist = (self.num_bs - 1) * self.bs_spacing_m + 1500.0
            self.sim_time_s = max(10.0, total_dist / max(1.0, self.speed_ms))
        else:
            self.sim_time_s = float(sim_time_s)

        # Base stations positions: linear along track at y = 50m, z = 30m
        self.bs_positions = [
            (i * self.bs_spacing_m, 50.0, 30.0) for i in range(self.num_bs)
        ]
        self.tx_power_dbm = 46.0  # 40 Watts
        self.ant_gain_db = 15.0

        np.random.seed(self.seed)

    def _calculate_rsrp(self, ue_x: float, ue_y: float, ue_z: float, bs_idx: int) -> float:
        bs_x, bs_y, bs_z = self.bs_positions[bs_idx]
        distance = math.sqrt((ue_x - bs_x)**2 + (ue_y - bs_y)**2 + (ue_z - bs_z)**2)
        distance = max(1.0, distance)

        # 3GPP Path loss (2.1 GHz carrier)
        path_loss_db = 38.84 + 30.0 * math.log10(distance)
        shadow_fading_db = np.random.normal(0.0, 3.5)
        rsrp = self.tx_power_dbm + self.ant_gain_db - path_loss_db + shadow_fading_db
        return rsrp

    def run(self) -> pd.DataFrame:
        """Runs the simulation and returns telemetry dataframe."""
        records: List[Dict[str, Any]] = []

        total_steps = int(self.sim_time_s / self.sample_interval_s)
        start_x = -500.0

        serving_bs = 0
        target_bs = -1
        ttt_timer = 0.0
        is_executing_ho = False
        ho_exec_timer = 0.0
        handover_count = 0

        prev_rsrp = self._calculate_rsrp(start_x, 0.0, 3.0, 0)
        prev_time = 0.0
        rtt_ewma = 30.0
        rtt_var_ewma = 2.0

        for step in range(total_steps):
            now = step * self.sample_interval_s
            train_x = start_x + now * self.speed_ms

            # Compute RSRP for all BSs
            rsrp_list = [self._calculate_rsrp(train_x, 0.0, 3.0, i) for i in range(self.num_bs)]
            serving_rsrp = rsrp_list[serving_bs]

            # Signal fade rate d(RSRP)/dt
            dt = now - prev_time
            fade_rate = (prev_rsrp - serving_rsrp) / dt if dt > 1e-4 else 0.0
            prev_rsrp = serving_rsrp
            prev_time = now

            # Candidate target cell
            best_target_id = -1
            best_target_rsrp = -140.0
            for i in range(self.num_bs):
                if i != serving_bs and rsrp_list[i] > best_target_rsrp:
                    best_target_rsrp = rsrp_list[i]
                    best_target_id = i

            is_handover_active = False
            interruption_ms = 0.0

            if is_executing_ho:
                ho_exec_timer -= self.sample_interval_s
                is_handover_active = True
                interruption_ms = self.sample_interval_s * 1000.0
                if ho_exec_timer <= 0.0:
                    is_executing_ho = False
                    serving_bs = target_bs
                    target_bs = -1
                    handover_count += 1
            else:
                if best_target_id >= 0 and (best_target_rsrp > serving_rsrp + self.hysteresis_db):
                    target_bs = best_target_id
                    ttt_timer += self.sample_interval_s
                    if ttt_timer >= self.time_to_trigger_s:
                        is_executing_ho = True
                        ho_exec_timer = self.handover_delay_s
                        ttt_timer = 0.0
                        is_handover_active = True
                        interruption_ms = self.handover_delay_s * 1000.0
                else:
                    ttt_timer = 0.0
                    target_bs = -1

            # RTT & Loss dynamics
            dist_from_serving = abs(train_x - self.bs_positions[serving_bs][0])
            dist_factor = min(1.0, dist_from_serving / self.bs_spacing_m)

            base_rtt = 25.0 + dist_factor * 20.0
            loss_ratio = 0.01

            if is_handover_active:
                loss_ratio = float(np.random.uniform(0.75, 1.00))
                instant_rtt = base_rtt + float(np.random.uniform(180.0, 280.0))
            elif ttt_timer > 0.0:
                ttt_ratio = ttt_timer / self.time_to_trigger_s
                loss_ratio = 0.05 + 0.35 * ttt_ratio
                instant_rtt = base_rtt + 40.0 * ttt_ratio
            elif serving_rsrp < -95.0:
                loss_ratio = 0.08
                instant_rtt = base_rtt + 25.0
            else:
                instant_rtt = base_rtt

            instant_rtt += float(np.random.uniform(-3.0, 3.0))
            instant_rtt = max(10.0, instant_rtt)

            # EWMA
            alpha = 0.125
            rtt_var_ewma = (1.0 - alpha) * rtt_var_ewma + alpha * abs(instant_rtt - rtt_ewma)
            rtt_ewma = (1.0 - alpha) * rtt_ewma + alpha * instant_rtt

            records.append({
                "timestamp_s": round(now, 4),
                "train_x": round(train_x, 4),
                "velocity_kmh": round(self.speed_kmh, 2),
                "serving_bs": int(serving_bs),
                "target_bs": int(target_bs),
                "rsrp_dbm": round(serving_rsrp, 2),
                "signal_fade_rate": round(fade_rate, 2),
                "rtt_ms": round(instant_rtt, 2),
                "rtt_variance_ms": round(rtt_var_ewma, 2),
                "packet_loss_ratio": round(loss_ratio, 4),
                "is_handover": 1 if is_handover_active else 0,
                "handover_count": int(handover_count),
                "interruption_time_ms": round(interruption_ms, 2)
            })

        return pd.DataFrame(records)
