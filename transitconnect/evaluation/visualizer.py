"""
Publication-Quality Visualization Generator for TransitConnect.
Generates high-resolution PNG (300 DPI) and vector PDF figures from simulation results:
1. Packet Delivery Ratio (PDR) vs Speed
2. Goodput vs Speed
3. End-to-End Delay vs Speed
4. Handover Interruption & Packet Loss
5. Reliability vs FEC Overhead Trade-Off
6. Synchronized Time-Series Handover Adaptation Dynamics (350 km/h)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

PROTOCOL_PALETTE = {
    "1. Unadapted (Fixed W, No FEC)": "#7f7f7f",     # Gray
    "2. Fixed FEC (Fixed W, Static E)": "#bcbd22",   # Olive
    "3. Adaptive Window Only": "#17becf",            # Cyan
    "4. Adaptive FEC Only": "#ff7f0e",               # Orange
    "5. Reactive Loss-Based TCP": "#d62728",         # Red
    "6. TransitConnect Joint Controller": "#1f77b4"  # Deep Blue
}

class BenchmarkVisualizer:
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent.parent.parent / "output" / "figures"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_pdr_vs_speed(self, benchmark_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        sns.barplot(
            data=benchmark_df,
            x="speed_kmh",
            y="pdr_percent",
            hue="protocol",
            palette=PROTOCOL_PALETTE,
            ax=ax
        )
        ax.set_title("Packet Delivery Ratio (PDR) vs. Train Speed", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Train Velocity (km/h)", fontsize=12, labelpad=8)
        ax.set_ylabel("Packet Delivery Ratio (%)", fontsize=12, labelpad=8)
        ax.set_ylim(40, 105)
        ax.legend(title="Protocol Configuration", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        fig.savefig(self.output_dir / "pdr_vs_speed.png", dpi=300)
        fig.savefig(self.output_dir / "pdr_vs_speed.pdf")
        plt.close(fig)

    def plot_goodput_vs_speed(self, benchmark_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        sns.lineplot(
            data=benchmark_df,
            x="speed_kmh",
            y="goodput_mbps",
            hue="protocol",
            palette=PROTOCOL_PALETTE,
            marker="o",
            markersize=8,
            linewidth=2.2,
            ax=ax
        )
        ax.set_title("Effective Goodput vs. Train Speed", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Train Velocity (km/h)", fontsize=12, labelpad=8)
        ax.set_ylabel("Goodput (Mbps)", fontsize=12, labelpad=8)
        ax.legend(title="Protocol Configuration", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        fig.savefig(self.output_dir / "goodput_vs_speed.png", dpi=300)
        fig.savefig(self.output_dir / "goodput_vs_speed.pdf")
        plt.close(fig)

    def plot_delay_vs_speed(self, benchmark_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        sns.barplot(
            data=benchmark_df,
            x="speed_kmh",
            y="avg_e2e_delay_ms",
            hue="protocol",
            palette=PROTOCOL_PALETTE,
            ax=ax
        )
        ax.set_title("Average End-to-End Delay vs. Train Speed", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Train Velocity (km/h)", fontsize=12, labelpad=8)
        ax.set_ylabel("Average Delay (ms)", fontsize=12, labelpad=8)
        ax.legend(title="Protocol Configuration", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        fig.savefig(self.output_dir / "delay_vs_speed.png", dpi=300)
        fig.savefig(self.output_dir / "delay_vs_speed.pdf")
        plt.close(fig)

    def plot_fec_overhead_tradeoff(self, benchmark_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        sns.scatterplot(
            data=benchmark_df,
            x="fec_overhead_percent",
            y="pdr_percent",
            hue="protocol",
            style="speed_kmh",
            palette=PROTOCOL_PALETTE,
            s=120,
            ax=ax
        )
        ax.set_title("Reliability (PDR) vs. FEC Overhead Trade-Off", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("FEC Bandwidth Overhead (%)", fontsize=12, labelpad=8)
        ax.set_ylabel("Packet Delivery Ratio (%)", fontsize=12, labelpad=8)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        fig.savefig(self.output_dir / "fec_overhead_tradeoff.png", dpi=300)
        fig.savefig(self.output_dir / "fec_overhead_tradeoff.pdf")
        plt.close(fig)

    def plot_time_series_adaptation(self, time_series_df: pd.DataFrame, speed_kmh: float = 350.0):
        """
        Plots a synchronized 4-panel timeline for a high-speed rail run showing:
        1. RSRP (dBm)
        2. Signal Fade Rate & RTT Variance
        3. Handover-Risk Score H(t) with T1 & T2 thresholds
        4. Coordinated Sliding Window W(t) and Dynamic FEC E(t)
        """
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, dpi=300)
        ts = time_series_df["timestamp_s"]

        # Panel 1: RSRP & Handover Events
        axes[0].plot(ts, time_series_df["rsrp_dbm"], color="#1f77b4", label="Serving Cell RSRP (dBm)", linewidth=1.8)
        if "is_handover" in time_series_df.columns:
            ho_times = time_series_df.loc[time_series_df["is_handover"] == 1, "timestamp_s"]
            for ht in ho_times:
                axes[0].axvline(x=ht, color="#d62728", linestyle="--", alpha=0.7, label="Handover Event" if ht == ho_times.iloc[0] else "")
        axes[0].set_ylabel("RSRP (dBm)", fontsize=11)
        axes[0].set_title(f"TransitConnect Operational Dynamics at {int(speed_kmh)} km/h", fontsize=13, fontweight="bold")
        axes[0].legend(loc="upper right", framealpha=0.8)

        # Panel 2: Fade Rate & RTT Variance
        ax2_twin = axes[1].twinx()
        l1 = axes[1].plot(ts, time_series_df["signal_fade_rate"], color="#ff7f0e", label="Fade Rate (dB/s)", linewidth=1.5)
        l2 = ax2_twin.plot(ts, time_series_df["rtt_variance_ms"], color="#9467bd", label="RTT Var (ms)", linewidth=1.5, linestyle=":")
        axes[1].set_ylabel("Fade Rate (dB/s)", fontsize=11, color="#ff7f0e")
        ax2_twin.set_ylabel("RTT Var (ms)", fontsize=11, color="#9467bd")
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        axes[1].legend(lines, labels, loc="upper right", framealpha=0.8)

        # Panel 3: Risk Score H(t) and Thresholds
        axes[2].plot(ts, time_series_df["risk_score_h"], color="#d62728", label="Handover-Risk Score H(t)", linewidth=2.0)
        axes[2].axhline(y=0.35, color="#2ca02c", linestyle="--", alpha=0.8, label="T1 (Low/Mod Threshold)")
        axes[2].axhline(y=0.70, color="#d62728", linestyle="--", alpha=0.8, label="T2 (Mod/High Threshold)")
        axes[2].set_ylabel("Risk Score H", fontsize=11)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(loc="upper right", framealpha=0.8)

        # Panel 4: Sliding Window W(t) & Dynamic FEC E(t)
        ax4_twin = axes[3].twinx()
        l3 = axes[3].plot(ts, time_series_df["window_size"], color="#2ca02c", label="Sliding Window W (pkts)", linewidth=2.0)
        l4 = ax4_twin.plot(ts, time_series_df["fec_redundancy"] * 100.0, color="#8c564b", label="FEC Redundancy E (%)", linewidth=2.0, linestyle="-.")
        axes[3].set_ylabel("Window W (pkts)", fontsize=11, color="#2ca02c")
        ax4_twin.set_ylabel("FEC Redundancy (%)", fontsize=11, color="#8c564b")
        axes[3].set_xlabel("Simulation Time (s)", fontsize=12)
        lines2 = l3 + l4
        labels2 = [l.get_label() for l in lines2]
        axes[3].legend(lines2, labels2, loc="upper right", framealpha=0.8)

        plt.tight_layout()
        fig.savefig(self.output_dir / "time_series_handover_adaptation_350kmh.png", dpi=300)
        fig.savefig(self.output_dir / "time_series_handover_adaptation_350kmh.pdf")
        plt.close(fig)

    def generate_all_plots(self, benchmark_df: pd.DataFrame, sample_time_series_df: Optional[pd.DataFrame] = None):
        """Generates the full set of evaluation figures."""
        self.plot_pdr_vs_speed(benchmark_df)
        self.plot_goodput_vs_speed(benchmark_df)
        self.plot_delay_vs_speed(benchmark_df)
        self.plot_fec_overhead_tradeoff(benchmark_df)
        if sample_time_series_df is not None:
            self.plot_time_series_adaptation(sample_time_series_df)
