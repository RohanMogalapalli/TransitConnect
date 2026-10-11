"""
Ablation Benchmark Suite for TransitConnect.
Executes the full 5-way ablation matrix plus Reactive TCP baseline across
four high-speed rail velocity regimes: 150 km/h, 250 km/h, 300 km/h, and 350 km/h.
Evaluates both ns-3 generated traces and high-fidelity physics corridor simulations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from ..simulation.corridor_sim import HsrCorridorSimulator
from ..simulation.ns3_runner import run_ns3_simulation, is_ns3_available
from ..monitoring.monitor import NetworkMobilityMonitor
from ..risk.weighted_sum import WeightedSumRiskModel
from ..risk.ml_model import MLRiskPredictor
from ..controller.joint_controller import JointAdaptiveController
from ..baselines.unadapted import UnadaptedProtocol
from ..baselines.fixed_fec import FixedFecProtocol
from ..baselines.window_only import WindowOnlyProtocol
from ..baselines.fec_only import FecOnlyProtocol
from ..baselines.reactive_tcp import ReactiveTcpProtocol
from .metrics import aggregate_protocol_metrics

SPEED_PROFILES_KMH = [150.0, 250.0, 300.0, 350.0]

class BenchmarkSuite:
    def __init__(
        self,
        use_ns3: bool = True,
        num_bs: int = 5,
        bs_spacing_m: float = 2000.0,
        sim_time_s: Optional[float] = None,
        seed: int = 42,
        ml_model_path: Optional[str] = None
    ):
        self.use_ns3 = use_ns3 and is_ns3_available()
        self.num_bs = num_bs
        self.bs_spacing_m = bs_spacing_m
        self.sim_time_s = sim_time_s
        self.seed = seed
        self.risk_model = WeightedSumRiskModel()
        self.ml_predictor = MLRiskPredictor(ml_model_path) if ml_model_path else None

    def get_simulation_trace(self, speed_kmh: float) -> pd.DataFrame:
        """Obtains trace using ns-3 if available and requested, otherwise Python corridor simulator."""
        if self.use_ns3:
            try:
                res = run_ns3_simulation(
                    speed_kmh=speed_kmh,
                    num_bs=self.num_bs,
                    bs_spacing_m=self.bs_spacing_m,
                    sim_time_s=self.sim_time_s,
                    seed=self.seed
                )
                return res["dataframe"]
            except Exception as e:
                print(f"Warning: ns-3 simulation failed ({e}), falling back to Python corridor engine.")

        sim = HsrCorridorSimulator(
            speed_kmh=speed_kmh,
            num_bs=self.num_bs,
            bs_spacing_m=self.bs_spacing_m,
            sim_time_s=self.sim_time_s,
            seed=self.seed
        )
        return sim.run()

    def run_speed_benchmark(self, speed_kmh: float, trace_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Runs all 6 protocol configurations over a single speed trace."""
        if trace_df is None:
            trace_df = self.get_simulation_trace(speed_kmh)

        duration_s = float(trace_df["timestamp_s"].max()) - float(trace_df["timestamp_s"].min())
        if duration_s <= 0.0:
            duration_s = 10.0

        # Instantiate protocol controllers
        unadapted = UnadaptedProtocol(w0=64)
        fixed_fec = FixedFecProtocol(w0=64, static_fec_ratio=0.15)
        window_only = WindowOnlyProtocol(w0=64, w_min=4, alpha=0.85)
        fec_only = FecOnlyProtocol(w0=64, e0=0.05, e_max=0.60, beta=0.55)
        reactive_tcp = ReactiveTcpProtocol(w_init=32, w_max=64, w_min=2)
        transit_joint = JointAdaptiveController(w0=64, w_min=4, alpha=0.85, e0=0.05, e_max=0.60, beta=0.55)

        monitor = NetworkMobilityMonitor()

        # Step record lists
        unadapted_records = []
        fixed_fec_records = []
        win_only_records = []
        fec_only_records = []
        tcp_records = []
        joint_records = []

        np.random.seed(self.seed + int(speed_kmh))

        for _, row in trace_df.iterrows():
            features, smoothed = monitor.process_sample(row.to_dict())
            
            # Handover-risk calculation
            H, state = self.risk_model.compute_risk(features)
            if self.ml_predictor and self.ml_predictor.model:
                h_ml, state_ml = self.ml_predictor.predict(features)
                # Ensemble blend: 60% math model + 40% ML model
                H = 0.60 * H + 0.40 * h_ml

            loss_ratio = float(row.get("packet_loss_ratio", 0.01))
            rtt_ms = float(row.get("rtt_ms", 30.0))
            interruption_ms = float(row.get("interruption_time_ms", 0.0))

            # Application demands 64 packets per cycle
            traffic_demand = 64

            # 1. Unadapted
            res_u = unadapted.transmit_and_evaluate_block(traffic_demand, loss_ratio, rtt_ms)
            res_u["rtt_ms"] = rtt_ms
            res_u["interruption_time_ms"] = interruption_ms
            unadapted_records.append(res_u)

            # 2. Fixed FEC
            res_ff = fixed_fec.transmit_and_evaluate_block(traffic_demand, loss_ratio, rtt_ms)
            res_ff["rtt_ms"] = rtt_ms
            res_ff["interruption_time_ms"] = interruption_ms
            fixed_fec_records.append(res_ff)

            # 3. Adaptive Window Only
            res_wo = window_only.transmit_and_evaluate_block(traffic_demand, loss_ratio, rtt_ms, risk_score_h=H)
            res_wo["rtt_ms"] = rtt_ms
            res_wo["interruption_time_ms"] = interruption_ms
            win_only_records.append(res_wo)

            # 4. Adaptive FEC Only
            res_fo = fec_only.transmit_and_evaluate_block(traffic_demand, loss_ratio, rtt_ms, risk_score_h=H)
            res_fo["rtt_ms"] = rtt_ms
            res_fo["interruption_time_ms"] = interruption_ms
            fec_only_records.append(res_fo)

            # 5. Reactive TCP
            res_tcp = reactive_tcp.transmit_and_evaluate_block(traffic_demand, loss_ratio, rtt_ms)
            res_tcp["rtt_ms"] = rtt_ms
            res_tcp["interruption_time_ms"] = interruption_ms
            tcp_records.append(res_tcp)

            # 6. TransitConnect Joint
            transit_joint.update_control(H, state)
            res_j = transit_joint.transmit_and_evaluate_block(traffic_demand, loss_ratio, rtt_ms)
            res_j["rtt_ms"] = rtt_ms
            res_j["interruption_time_ms"] = interruption_ms
            res_j["risk_score_h"] = H
            joint_records.append(res_j)

        results = [
            aggregate_protocol_metrics(unadapted_records, duration_s, speed_kmh, "1. Unadapted (Fixed W, No FEC)"),
            aggregate_protocol_metrics(fixed_fec_records, duration_s, speed_kmh, "2. Fixed FEC (Fixed W, Static E)"),
            aggregate_protocol_metrics(win_only_records, duration_s, speed_kmh, "3. Adaptive Window Only"),
            aggregate_protocol_metrics(fec_only_records, duration_s, speed_kmh, "4. Adaptive FEC Only"),
            aggregate_protocol_metrics(tcp_records, duration_s, speed_kmh, "5. Reactive Loss-Based TCP"),
            aggregate_protocol_metrics(joint_records, duration_s, speed_kmh, "6. TransitConnect Joint Controller")
        ]

        return results

    def run_full_suite(self, speeds: Optional[List[float]] = None) -> pd.DataFrame:
        """Executes full multi-speed ablation suite across all specified velocities."""
        if speeds is None:
            speeds = SPEED_PROFILES_KMH

        all_summaries = []
        for sp in speeds:
            print(f"--> Benchmarking speed {sp} km/h...")
            res = self.run_speed_benchmark(sp)
            all_summaries.extend(res)

        df = pd.DataFrame(all_summaries)
        return df
