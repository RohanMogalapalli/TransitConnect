"""
CLI utility to execute a single TransitConnect simulation run.
Supports running via ns-3 (in WSL2) or Python corridor engine,
applying any selected protocol configuration, and printing metrics.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transitconnect.simulation.corridor_sim import HsrCorridorSimulator
from transitconnect.simulation.ns3_runner import run_ns3_simulation, is_ns3_available
from transitconnect.monitoring.monitor import NetworkMobilityMonitor
from transitconnect.risk.weighted_sum import WeightedSumRiskModel
from transitconnect.risk.ml_model import MLRiskPredictor
from transitconnect.controller.joint_controller import JointAdaptiveController
from transitconnect.baselines import (
    UnadaptedProtocol,
    FixedFecProtocol,
    WindowOnlyProtocol,
    FecOnlyProtocol,
    ReactiveTcpProtocol
)
from transitconnect.evaluation.metrics import aggregate_protocol_metrics

def main():
    parser = argparse.ArgumentParser(description="TransitConnect Simulation Runner")
    parser.add_argument("--speed", type=float, default=300.0, help="Train velocity in km/h (default: 300)")
    parser.add_argument("--engine", choices=["ns3", "python"], default="ns3", help="Simulation engine (default: ns3)")
    parser.add_argument(
        "--protocol",
        choices=["joint", "unadapted", "fixed_fec", "window_only", "fec_only", "reactive_tcp"],
        default="joint",
        help="Protocol configuration (default: joint)"
    )
    parser.add_argument("--numBs", type=int, default=5, help="Number of base stations (default: 5)")
    parser.add_argument("--bsSpacing", type=float, default=2000.0, help="Distance between base stations in meters")
    parser.add_argument("--simTime", type=float, default=50.0, help="Simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for telemetry logs")
    args = parser.parse_args()

    print("=" * 65)
    print(f" TransitConnect Simulation: {args.speed} km/h | Engine: {args.engine} | Protocol: {args.protocol}")
    print("=" * 65)

    # 1. Acquire Trace
    if args.engine == "ns3" and is_ns3_available():
        print(f"Running ns-3 simulation in WSL2 ({args.numBs} base stations, spacing {args.bsSpacing}m)...")
        res = run_ns3_simulation(
            speed_kmh=args.speed,
            num_bs=args.numBs,
            bs_spacing_m=args.bsSpacing,
            sim_time_s=args.simTime,
            seed=args.seed
        )
        trace_df = res["dataframe"]
        print(f"ns-3 completed: {res['total_samples']} samples, {res['handover_count']} handovers detected.")
    else:
        if args.engine == "ns3":
            print("Note: ns-3 not detected in WSL, falling back to high-fidelity Python corridor engine.")
        sim = HsrCorridorSimulator(
            speed_kmh=args.speed,
            num_bs=args.numBs,
            bs_spacing_m=args.bsSpacing,
            sim_time_s=args.simTime,
            seed=args.seed
        )
        trace_df = sim.run()
        print(f"Python simulator completed: {len(trace_df)} samples, {trace_df['handover_count'].max()} handovers.")

    # 2. Select Protocol
    if args.protocol == "joint":
        controller = JointAdaptiveController()
    elif args.protocol == "unadapted":
        controller = UnadaptedProtocol()
    elif args.protocol == "fixed_fec":
        controller = FixedFecProtocol()
    elif args.protocol == "window_only":
        controller = WindowOnlyProtocol()
    elif args.protocol == "fec_only":
        controller = FecOnlyProtocol()
    else:
        controller = ReactiveTcpProtocol()

    # 3. Risk Engine
    risk_model = WeightedSumRiskModel()
    ml_path = ROOT / "data" / "models" / "handover_risk_rf.joblib"
    ml_pred = MLRiskPredictor(str(ml_path)) if ml_path.exists() else None
    monitor = NetworkMobilityMonitor()

    step_records = []
    log_rows = []

    for _, row in trace_df.iterrows():
        features, smoothed = monitor.process_sample(row.to_dict())
        H, state = risk_model.compute_risk(features)
        if ml_pred and ml_pred.model:
            h_ml, _ = ml_pred.predict(features)
            H = 0.60 * H + 0.40 * h_ml

        loss_ratio = float(row.get("packet_loss_ratio", 0.01))
        rtt_ms = float(row.get("rtt_ms", 30.0))
        interruption_ms = float(row.get("interruption_time_ms", 0.0))

        if args.protocol == "joint":
            w, e, eff_state = controller.update_control(H, state)
            step_res = controller.transmit_and_evaluate_block(64, loss_ratio, rtt_ms)
        elif args.protocol in ("window_only", "fec_only"):
            step_res = controller.transmit_and_evaluate_block(64, loss_ratio, rtt_ms, risk_score_h=H)
        else:
            step_res = controller.transmit_and_evaluate_block(64, loss_ratio, rtt_ms)

        step_res["rtt_ms"] = rtt_ms
        step_res["interruption_time_ms"] = interruption_ms
        step_res["risk_score_h"] = H
        step_records.append(step_res)

        log_row = dict(row)
        log_row.update(step_res)
        log_rows.append(log_row)

    dur = float(trace_df["timestamp_s"].max()) - float(trace_df["timestamp_s"].min())
    summary = aggregate_protocol_metrics(step_records, dur, args.speed, args.protocol)

    print("\nSimulation Performance Summary:")
    print("-" * 45)
    print(f"  Packet Delivery Ratio (PDR): {summary['pdr_percent']:.2f}%")
    print(f"  Effective Goodput:          {summary['goodput_mbps']:.3f} Mbps")
    print(f"  Raw Throughput:             {summary['raw_throughput_mbps']:.3f} Mbps")
    print(f"  Average End-to-End Delay:   {summary['avg_e2e_delay_ms']:.2f} ms")
    print(f"  Average RTT:                {summary['avg_rtt_ms']:.2f} ms")
    print(f"  Handover Interruption Time: {summary['total_interruption_ms']:.1f} ms")
    print(f"  FEC Bandwidth Overhead:     {summary['fec_overhead_percent']:.2f}%")
    print("-" * 45)

    if args.output:
        pd.DataFrame(log_rows).to_csv(args.output, index=False)
        print(f"Telemetry log saved to: {args.output}")

if __name__ == "__main__":
    main()
