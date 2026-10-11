# TransitConnect: Complete User & Developer Guide

This document provides step-by-step instructions on:
1. [Starting and Ending Project Sessions](#1-starting-and-ending-project-sessions)
2. [Adding New Datasets (Field Logs & Real Traces)](#2-adding-new-datasets)
3. [Training Machine Learning Models](#3-training-machine-learning-models)
4. [Running Simulations (ns-3 & Python Corridor Engines)](#4-running-simulations)
5. [Running the Full Multi-Speed Ablation Benchmarking Suite](#5-running-benchmarks)
6. [Visualizing Results & Interactive Dashboard](#6-visualizing-results)
7. [Troubleshooting & Frequently Asked Questions](#7-troubleshooting)

---

## 1. Starting and Ending Project Sessions

### 1.1 Starting a Session
Whenever you open a new terminal in VS Code or PowerShell:

1. **Navigate to the Project Root**:
   ```powershell
   cd c:\projects\TranssitConnect
   ```

2. **Activate the Dedicated Python Virtual Environment (`.venv`)**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(Your terminal prompt should now show `(.venv)`).*

3. **Verify Dependencies and ns-3**:
   ```powershell
   python --version       # Expected: Python 3.12.x
   pytest -v              # Runs 20 automated unit tests (should show: 20 passed)
   ```

### 1.2 Ending a Session
1. **Stop Running Background Servers**:
   - If the Streamlit dashboard or a simulation is running in your terminal, press `Ctrl + C` to stop it.
2. **Deactivate the Virtual Environment**:
   ```powershell
   deactivate
   ```

---

## 2. Adding New Datasets

TransitConnect supports both **real-world drive-test field logs** (e.g. LTE/5G scanner logs) and **synthetic mobility traces**.

### 2.1 Where to Place New Dataset Files
Place new CSV or log files in either:
- `c:\projects\TranssitConnect\data\traces\<your_dataset>.csv`
- Or inside a subfolder in the project root, such as `c:\projects\TranssitConnect\Handover Dataset (ver 2)\`

### 2.2 Supported Column Names
The system's [`TraceLoader`](file:///c:/projects/TranssitConnect/transitconnect/simulation/trace_loader.py) automatically adapts and normalizes various common naming conventions:

| Standard Parameter | Accepted Alternative Column Names in Your CSV | Expected Units / Range |
|:---|:---|:---|
| **Timestamp** | `timestamp_s`, `Timestamp`, `time` | Seconds (e.g. `0.0, 0.5, 1.0`) or string (`MM:SS.s`) |
| **Signal Power** | `rsrp_dbm`, `Serving Cell RSRP`, `rssi`, `RSRP` | dBm ($-140$ to $-44$) or 3GPP Index (0 to 97) |
| **Handover Event** | `is_handover`, `Handover Trigger`, `handover` | `1` during handover, `0` otherwise |
| **Velocity / Speed**| `velocity_kmh`, `Speed (km/h)`, `speed` | km/h (e.g. `45.0` to `350.0`) |
| **Signal Quality** | `Serving Cell CINR`, `sinr`, `snr`, `cqi` | dB ($-15$ to $+25$) |
| **Delay / RTT** | `rtt_ms`, `latency`, `ping_ms` | Milliseconds (optional, estimated from CINR if missing) |
| **Packet Loss** | `packet_loss_ratio`, `loss_rate`, `bler` | Ratio $[0.0, 1.0]$ (optional, estimated if missing) |

### 2.3 How to Load and Validate Your New Dataset
To verify that TransitConnect can read your new file without errors, run a quick one-liner:

```powershell
python -c "from transitconnect.simulation.trace_loader import TraceLoader; loader = TraceLoader('data/traces/your_file.csv'); print(loader.get_dataframe().head())"
```

---

## 3. Training Machine Learning Models

TransitConnect uses **Random Forest** and **Gradient Boosting** models to predict upcoming handover degradation from 6 normalized features ($F, V, P, R, U, S$).

### 3.1 Training Options via CLI
Run [`scripts/train_ml.py`](file:///c:/projects/TranssitConnect/scripts/train_ml.py) with your desired data source flag:

#### Option A: Train on Combined Data (Recommended)
Combines real drive-test field reports from `Handover Dataset (ver 2)` with high-speed rail simulation traces (150–350 km/h) to create a robust model (11,754 samples):
```powershell
python scripts/train_ml.py --source combined
```

#### Option B: Train Exclusively on Real Drive-Test Field Logs
Trains strictly on the 2,154 real-world LTE drive-test measurements from `Handover Dataset (ver 2)`:
```powershell
python scripts/train_ml.py --source real
```

#### Option C: Train Exclusively on HSR Simulation Traces
Trains on the 9,600 physics-based multi-cell rail corridor simulation samples:
```powershell
python scripts/train_ml.py --source simulation
```

### 3.2 What the Script Produces
When training finishes, the script outputs:
1. **Model Accuracy & Macro F1 Score** (typically 88–94% accuracy).
2. **Gini Feature Importance Ranking** (shows which indicators most strongly signal an impending handover).
3. **Saved Models**:
   - `data/models/handover_risk_rf.joblib` (Random Forest)
   - `data/models/handover_risk_gbdt.joblib` (Gradient Boosting)
4. **Processed Dataset CSV**:
   - `data/handover_risk_combined_dataset.csv` or `data/handover_risk_real_dataset.csv`.

---

## 4. Running Simulations

You can run individual simulations using the command-line utility [`scripts/run_simulation.py`](file:///c:/projects/TranssitConnect/scripts/run_simulation.py).

### 4.1 Running with ns-3 C++ Engine (Native in WSL2)
Runs the realistic multi-cell High-Speed Rail simulation binary inside Ubuntu WSL2:
```powershell
python scripts/run_simulation.py --speed 350 --engine ns3 --protocol joint
```

### 4.2 Running with Python Physics Corridor Engine
Runs the pure-Python high-fidelity rail corridor simulation (ideal for fast standalone experimentation):
```powershell
python scripts/run_simulation.py --speed 350 --engine python --protocol joint
```

### 4.3 Command-Line Arguments for `run_simulation.py`:
- `--speed <km/h>`: Train velocity (e.g. `150`, `250`, `300`, `350`). Default: `300`.
- `--engine <ns3|python>`: Choose simulation engine. Default: `ns3`.
- `--protocol <name>`: Choose protocol configuration:
  * `joint`: Proposed TransitConnect Joint Controller ($W(H) + E(H)$)
  * `unadapted`: Fixed sliding window (64 packets), zero FEC
  * `fixed_fec`: Fixed sliding window, static 15% FEC
  * `window_only`: Handover-risk adaptive window only
  * `fec_only`: Dynamic FEC only
  * `reactive_tcp`: Classic loss-based AIMD TCP (halves window on loss)
- `--simTime <seconds>`: Simulation duration (e.g. `40`, `60`). Default: `50.0`.
- `--numBs <4..6>`: Number of macro cellular base stations. Default: `5`.
- `--bsSpacing <meters>`: Distance between base stations. Default: `2000.0`.
- `--output <path>`: Save telemetry step logs to CSV.

*Example Output*:
```
Simulation Performance Summary:
---------------------------------------------
  Packet Delivery Ratio (PDR): 99.37%
  Effective Goodput:          13.325 Mbps
  Raw Throughput:             14.609 Mbps
  Average End-to-End Delay:   16.01 ms
  Average RTT:                32.03 ms
  Handover Interruption Time: 360.0 ms
  FEC Bandwidth Overhead:     8.21%
---------------------------------------------
```

---

## 5. Running Benchmarks

To reproduce the complete academic benchmark matrix comparing all 6 protocols across all 4 train speeds (150, 250, 300, and 350 km/h):

```powershell
python scripts/run_benchmarks.py
```

### What this Command Generates:
1. **Full CSV Metric Matrix**:
   - Saved to: [`output/results/ablation_benchmark_results.csv`](file:///c:/projects/TranssitConnect/output/results/ablation_benchmark_results.csv)
2. **Academic Markdown Report**:
   - Saved to: [`output/results/benchmark_summary_report.md`](file:///c:/projects/TranssitConnect/output/results/benchmark_summary_report.md)
3. **Publication-Quality Figures (300 DPI PNG & Vector PDF)**:
   - Saved to: [`output/figures/`](file:///c:/projects/TranssitConnect/output/figures/)

---

## 6. Visualizing Results

### 6.1 Interactive Streamlit Web Dashboard
The easiest way to demonstrate and interact with the project is the Streamlit web dashboard.

#### Launching the Dashboard:
```powershell
streamlit run app/dashboard.py
```
*(Streamlit will open automatically in your browser at `http://localhost:8501`).*

#### Dashboard Features:
- **Data Source Dropdown**:
  - `Real Drive-Test (Handover Dataset ver 2)`: Visualizes real-world LTE field measurements.
  - `ns-3 C++ (WSL2)`: Runs ns-3 native simulations in the background.
  - `Python Physics Corridor Engine`: Instant interactive physics corridor.
- **Interactive Tuning Sliders**:
  - Train Speed (100–400 km/h)
  - Base station count & spacing
  - Risk formula weights ($w_1..w_5$)
  - Risk decision thresholds ($T_1, T_2$)
- **Tabs**:
  - **Tab 1: Telemetry & Risk Adaptation**: Synchronized timeline charts of RSRP, fade rate, delay variance, Risk Score $H(t)$, sliding window $W(t)$, and dynamic FEC $E(t)$.
  - **Tab 2: Comparative Ablation Benchmarks**: Interactive bar charts and sortable data table comparing all 6 protocols. Includes a **Download CSV** button.
  - **Tab 3: ML Models & Feature Importance**: Visual bar chart ranking feature contributions from the trained models.
  - **Tab 4: Architecture & Mathematical Formulations**: Overview of the mathematical formulations ($H$, $W$, $E$) and recovery smoothing.

### 6.2 Viewing Exported Publication Plots
All generated figures are saved in both PNG and vector PDF format inside [`output/figures/`](file:///c:/projects/TranssitConnect/output/figures/):
- `pdr_vs_speed.png` & `.pdf`: Packet Delivery Ratio across velocities
- `goodput_vs_speed.png` & `.pdf`: Effective Goodput (Mbps) comparison
- `delay_vs_speed.png` & `.pdf`: Average end-to-end latency comparison
- `fec_overhead_tradeoff.png` & `.pdf`: Reliability vs. FEC overhead Pareto curve
- `time_series_handover_adaptation_350kmh.png` & `.pdf`: Synchronized 4-panel timeline plot during a 350 km/h handover

---

## 7. Troubleshooting & FAQs

### Q1: What if `pytest` fails with `ModuleNotFoundError: No module named 'transitconnect'`?
**Solution**: Ensure your virtual environment is activated (`.\.venv\Scripts\Activate.ps1`) and that `pytest.ini` is present in the project root with `pythonpath = .`.

### Q2: What if WSL gives an error when building or running ns-3?
**Solution**: Rebuild the ns-3 C++ binary inside WSL using:
```powershell
wsl -d Ubuntu bash /mnt/c/projects/TranssitConnect/ns3_sim/build.sh
```
Ensure required packages are present in WSL: `wsl -d Ubuntu -u root apt-get install -y ns3 libns3-dev build-essential pkg-config libgsl-dev libsqlite3-dev`.

### Q3: How do I evaluate a single new CSV trace I recorded?
**Solution**:
```python
from transitconnect.simulation.trace_loader import TraceLoader
from transitconnect.evaluation.benchmark import BenchmarkSuite

loader = TraceLoader("path/to/my_trace.csv")
suite = BenchmarkSuite()
results = suite.run_speed_benchmark(speed_kmh=120.0, trace_df=loader.get_dataframe())
print(results)
```
