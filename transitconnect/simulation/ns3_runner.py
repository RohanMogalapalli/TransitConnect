"""
ns-3 runner for TransitConnect.
Executes the native C++ high-speed rail corridor simulation binary inside WSL2 Ubuntu
and returns telemetry data and execution metadata.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Returns absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent

def is_ns3_available() -> bool:
    """Checks if WSL2 and the compiled ns-3 binary are available."""
    try:
        res = subprocess.run(
            ["wsl", "-d", "Ubuntu", "bash", "-c", "which pkg-config"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return res.returncode == 0
    except Exception:
        return False

def build_ns3_binary() -> bool:
    """Builds the ns-3 C++ simulator inside WSL if not already compiled."""
    root = get_project_root()
    script_path = (root / "ns3_sim" / "build.sh").as_posix()
    # Convert windows path to wsl path
    wsl_script = f"/mnt/{script_path[0].lower()}{script_path[2:]}"
    try:
        logger.info(f"Building ns-3 simulator via {wsl_script}...")
        res = subprocess.run(
            ["wsl", "-d", "Ubuntu", "bash", wsl_script],
            capture_output=True,
            text=True,
            timeout=60
        )
        if res.returncode != 0:
            logger.error(f"Failed to build ns-3 simulator: {res.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error invoking WSL build: {e}")
        return False

def run_ns3_simulation(
    speed_kmh: float = 300.0,
    num_bs: int = 5,
    bs_spacing_m: float = 2000.0,
    sim_time_s: Optional[float] = None,
    hysteresis_db: float = 3.0,
    time_to_trigger_s: float = 0.256,
    handover_delay_s: float = 0.080,
    seed: int = 42,
    output_trace_csv: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the ns-3 HSR simulation in WSL2 Ubuntu and returns a dictionary with
    the resulting DataFrame, summary statistics, and trace path.
    """
    root = get_project_root()
    
    # Calculate simulation time if not provided: allow train to travel across the entire corridor + margins
    if sim_time_s is None:
        total_distance_m = (num_bs - 1) * bs_spacing_m + 1500.0 # extra travel
        speed_ms = speed_kmh * (1000.0 / 3600.0)
        sim_time_s = max(10.0, total_distance_m / speed_ms)

    if output_trace_csv is None:
        traces_dir = root / "data" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        output_trace_csv = str(traces_dir / f"ns3_trace_{int(speed_kmh)}kmh_seed{seed}.csv")

    out_path = Path(output_trace_csv).resolve()
    wsl_out_path = f"/mnt/{out_path.as_posix()[0].lower()}{out_path.as_posix()[2:]}"
    
    bin_path = root / "ns3_sim" / "transitconnect_hsr_sim"
    wsl_bin_path = f"/mnt/{bin_path.as_posix()[0].lower()}{bin_path.as_posix()[2:]}"

    # Build binary if it doesn't exist
    if not (root / "ns3_sim" / "transitconnect_hsr_sim").exists():
        built = build_ns3_binary()
        if not built:
            raise RuntimeError("ns-3 binary could not be built in WSL2.")

    cmd = [
        "wsl", "-d", "Ubuntu", "bash", "-c",
        f"{wsl_bin_path} "
        f"--speed={speed_kmh} "
        f"--numBs={num_bs} "
        f"--bsSpacing={bs_spacing_m} "
        f"--simTime={sim_time_s:.2f} "
        f"--hysteresis={hysteresis_db} "
        f"--ttt={time_to_trigger_s} "
        f"--handoverDelay={handover_delay_s} "
        f"--seed={seed} "
        f"--outputTrace={wsl_out_path}"
    ]

    logger.info(f"Running ns-3 command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"ns-3 simulation failed (code {result.returncode}):\n{result.stderr}")

    if not out_path.exists():
        raise FileNotFoundError(f"Expected ns-3 output trace not found at {out_path}")

    df = pd.read_csv(out_path)
    
    total_samples = len(df)
    total_handovers = int(df["handover_count"].max()) if "handover_count" in df.columns else 0
    avg_rsrp = float(df["rsrp_dbm"].mean()) if "rsrp_dbm" in df.columns else 0.0
    avg_rtt = float(df["rtt_ms"].mean()) if "rtt_ms" in df.columns else 0.0

    return {
        "engine": "ns-3",
        "speed_kmh": speed_kmh,
        "num_bs": num_bs,
        "sim_time_s": sim_time_s,
        "total_samples": total_samples,
        "handover_count": total_handovers,
        "avg_rsrp_dbm": avg_rsrp,
        "avg_rtt_ms": avg_rtt,
        "trace_file": str(out_path),
        "dataframe": df
    }
