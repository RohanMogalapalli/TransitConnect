"""
CLI script to generate training dataset and train Scikit-learn Handover-Risk models.
Supports training on:
1. Real-world drive-test data ('Handover Dataset (ver 2)')
2. High-speed rail simulation traces (ns-3 / physics corridor)
3. Combined real + simulation dataset
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transitconnect.simulation.corridor_sim import HsrCorridorSimulator
from transitconnect.simulation.trace_loader import load_real_handover_dataset
from transitconnect.risk.dataset_gen import DatasetGenerator
from transitconnect.risk.ml_model import MLRiskPredictor

def main():
    parser = argparse.ArgumentParser(description="TransitConnect ML Model Trainer")
    parser.add_argument(
        "--source",
        choices=["combined", "real", "simulation"],
        default="combined",
        help="Training dataset source (default: combined [Real Drive Test + HSR Simulation])"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f" TransitConnect ML Training Pipeline | Source: {args.source.upper()}")
    print("=" * 70)

    dataset_gen = DatasetGenerator()
    dataframes = []

    # 1. Real-World Dataset ('Handover Dataset (ver 2)')
    if args.source in ("real", "combined"):
        print("\n[Phase 1] Ingesting real-world drive test data from 'Handover Dataset (ver 2)'...")
        real_raw_df = load_real_handover_dataset()
        print(f"  -> Loaded {len(real_raw_df)} real drive-test measurement reports ({real_raw_df['is_handover'].sum()} handovers)")
        labeled_real_df = dataset_gen.build_dataset_from_dataframe(real_raw_df)
        labeled_real_df["data_origin"] = "real_drive_test"
        dataframes.append(labeled_real_df)

    # 2. Simulation Traces (150 - 350 km/h)
    if args.source in ("simulation", "combined"):
        print("\n[Phase 2] Generating multi-cell high-speed rail simulation traces (150 - 350 km/h)...")
        speeds = [150.0, 250.0, 300.0, 350.0]
        seeds = [42, 101]
        for sp in speeds:
            for s in seeds:
                sim = HsrCorridorSimulator(speed_kmh=sp, num_bs=5, bs_spacing_m=2000.0, sim_time_s=60.0, seed=s)
                trace_df = sim.run()
                labeled_sim_df = dataset_gen.build_dataset_from_dataframe(trace_df)
                labeled_sim_df["data_origin"] = f"hsr_sim_{int(sp)}kmh"
                dataframes.append(labeled_sim_df)

    full_dataset = pd.concat(dataframes, ignore_index=True)
    print(f"\n[Phase 3] Total Training Dataset: {len(full_dataset)} labeled samples.")
    print("Class distribution:")
    print(full_dataset["risk_class"].value_counts().to_string())

    # Save dataset to CSV
    dataset_csv = ROOT / "data" / f"handover_risk_{args.source}_dataset.csv"
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    full_dataset.to_csv(dataset_csv, index=False)
    print(f"Saved dataset to {dataset_csv}")

    # Train Random Forest
    print("\n[Phase 4] Training Random Forest Risk Predictor...")
    rf_predictor = MLRiskPredictor()
    rf_metrics = rf_predictor.train_models(full_dataset, model_type="rf")
    rf_model_path = ROOT / "data" / "models" / f"handover_risk_rf.joblib"
    rf_predictor.save(str(rf_model_path))

    print(f"  Random Forest Accuracy: {rf_metrics['accuracy'] * 100:.2f}%")
    print(f"  Random Forest Macro F1: {rf_metrics['macro_f1']:.4f}")
    print(f"  Random Forest Regressor MSE: {rf_metrics['mse']:.4f}")
    print("  Feature Importances:")
    for feat, imp in rf_metrics["feature_importances"].items():
        print(f"    - {feat:<20}: {imp:.4f}")
    print(f"  Saved model to {rf_model_path}")

    # Train Gradient Boosting
    print("\n[Phase 5] Training Gradient Boosting Risk Predictor...")
    gbdt_predictor = MLRiskPredictor()
    gbdt_metrics = gbdt_predictor.train_models(full_dataset, model_type="gbdt")
    gbdt_model_path = ROOT / "data" / "models" / f"handover_risk_gbdt.joblib"
    gbdt_predictor.save(str(gbdt_model_path))

    print(f"  Gradient Boosting Accuracy: {gbdt_metrics['accuracy'] * 100:.2f}%")
    print(f"  Gradient Boosting Macro F1: {gbdt_metrics['macro_f1']:.4f}")
    print(f"  Saved model to {gbdt_model_path}")

    print("\n" + "=" * 70)
    print(f" SUCCESS: Models trained and validated on {args.source.upper()} dataset!")
    print("=" * 70)

if __name__ == "__main__":
    main()
