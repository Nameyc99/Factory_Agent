---
title: Smart Factory Alert Agent

---

# Smart Factory Alert Agent

A lightweight prototype for smart factory anomaly detection, including a synthetic dataset generator, a versatile alert agent (rule-based, ML, or hybrid), and demo + evaluation tools.

---

## Table of Contents 
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data format & generator](#data-format--generator)
- [Usage: Agent CLI](#usage-agent-cli)
- [Alerts & output formats](#alerts--output-formats)
- [Evaluation & report](#evaluation--report)
- [Troubleshooting & edge cases](#troubleshooting--edge-cases)
- [Reproducibility & randomness](#reproducibility--randomness)
- [AI tools & credits](#ai-tools--credits)
- [Deliverables & submission checklist](#deliverables--submission-checklist)
- [Extending the project (optional ideas)](#extending-the-project-optional-ideas)

---

## Quick start

Run the following commands to reproduce the default demo locally:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # (use .venv\Scripts\activate on Windows)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the default dataset (200 rows, 1min freq, 8% anomalies)
python generate_smartfactory_dataset.py

# 4. Run demo mode of the agent in hybrid mode
python smartfactory_alert_agent.py --data ./data/smartfactory_dataset.csv --model hybrid --demo --out_dir ./out

```

## Repository layout

```
.
├── generate_smartfactory_dataset.py
├── smartfactory_alert_agent.py
├── requirements.txt
├── README.md
├── report.pdf
├── data/ # synthetic datasets
├── out/ # demo outputs (alerts, summary, models)
└── report_figures/ # report diagrams & figures
```

## Installation

- **Python requirement:** Python 3.10 or higher is recommended.
- Works on Linux, macOS, and Windows.  
- Ensure you have `pip` and `venv` available.

### Step-by-step

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # On Linux/macOS
# .venv\Scripts\activate       # On Windows PowerShell

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

## Data format & generator

The dataset is stored as a CSV with the following header and columns:

| Column     | Type    | Description                                                                 |
|------------|---------|-----------------------------------------------------------------------------|
| `timestamp`| string  | Timestamp in `YYYY-MM-DD HH:MM:SS` format (24h).                            |
| `temp`     | float   | Machine temperature in °C.                                                  |
| `pressure` | float   | Machine pressure in bar.                                                    |
| `vibration`| float   | Vibration in g.                                                             |
| `label`    | string  | Ground-truth status: `normal`, `abnormal`, or `unknown`.                    |

### Default dataset characteristics
- Rows: **200**
- Frequency: **1 minute**
- Anomaly percentage: **8%**
- Missing value percentage: **2%**
- Random seed: **42**

### Commands

Generate a dataset with defaults:

```bash
python generate_smartfactory_dataset.py
```

Generate with custom arguments:
```bash
python generate_smartfactory_dataset.py --rows 300 --freq 5min --out ./data/my.csv
```

Example CSV (first 3 rows)
```
timestamp,temp,pressure,vibration,label
2025-01-01 00:00:00,46.2,1.03,0.03,normal
2025-01-01 00:01:00,45.9,1.01,0.02,normal
2025-01-01 00:02:00,53.8,1.04,0.08,abnormal
```

## Usage: Agent CLI

The alert agent consumes the generated dataset and performs anomaly detection using three possible modes:

- **`rule`**: fixed thresholds on sensor values.
- **`ml`**: IsolationForest anomaly detection.
- **`hybrid`**: combines rule-based and ML detectors (default).

### Command-line options

| Flag            | Type     | Default  | Description                                                                 |
|-----------------|----------|----------|-----------------------------------------------------------------------------|
| `--data`        | str      | (req.)   | Path to input CSV dataset.                                                  |
| `--model`       | str      | hybrid   | Detection mode: `rule`, `ml`, or `hybrid`.                                  |
| `--train`       | flag     | off      | If present with `ml`/`hybrid`, trains the IsolationForest.                  |
| `--out_dir`     | str      | `./out`  | Directory for outputs (alerts, summary, models).                            |
| `--realtime`    | flag     | off      | Simulate streaming with row-by-row alerts.                                  |
| `--speed`       | float    | 0.05     | Sleep interval (seconds) between rows in realtime mode.                     |
| `--window`      | int      | 1        | Rolling window size for smoothing.                                          |
| `--threshold`   | float    | auto     | ML anomaly score threshold.                                                 |
| `--seed`        | int      | 42       | Random seed for reproducibility.                                            |
| `--save_model`  | flag     | off      | Save trained ML model to `out_dir`.                                         |
| `--log_json`    | flag     | off      | Save alerts as JSONL to `out_dir/alerts.jsonl`.                             |
| `--report`      | flag     | off      | Generate evaluation report to `out_dir/summary.txt`.                        |
| `--demo`        | flag     | off      | Run on first 200 rows with sample alerts and summary.                       |

### Examples

**Hybrid realtime demo:**

```bash
python smartfactory_alert_agent.py --data ./data/smartfactory_dataset.csv --model hybrid --realtime --speed 0.1 --log_json --out_dir ./out

```

Train and save an ML model:

```bash
python smartfactory_alert_agent.py --data ./data/smartfactory_dataset.csv --model ml --train --save_model --out_dir ./out
```

Rule-based quick check with JSON logging:
```bash
python smartfactory_alert_agent.py --data ./data/smartfactory_dataset.csv --model rule --log_json --out_dir ./out
```

Each run saves outputs into the specified out_dir (default ./out).

## Alerts & output formats

- **Console alerts**: compact, human-readable lines:

```bash
[ALERT] 2025-01-01 00:10:00 | abnormal | temp=53.2, pressure=1.01, vibration=0.02 | suggestion=Check cooling system
```


- **JSONL schema** (`out_dir/alerts.jsonl`):

```json
{
  "timestamp": "2025-01-01 00:10:00",
  "row_index": 10,
  "sensor_values": {"temp": 53.2, "pressure": 1.01, "vibration": 0.02},
  "detector_flags": {"rule": true, "ml": false},
  "anomaly_score": 0.76,
  "label_pred": "abnormal",
  "suggestion": "Check cooling system"
}
```

- **Summary output** (out_dir/summary.txt): counts of normal/abnormal/unknown, total alerts, metrics summary.
- **Saved ML model artifacts**: model_<timestamp>.joblib in out_dir.
    
## Evaluation & report

- Use `--report` to generate evaluation metrics after processing a dataset with labels.
- Metrics include:
  - Confusion matrix
  - Precision, recall, F1-score for `abnormal` class
  - ROC AUC / PR AUC if ML-based detection was used
- Only rows labeled `normal` or `abnormal` are included in evaluation; rows labeled `unknown` are ignored.
- Outputs are saved to `out_dir/summary.txt` and optional plots can be found in `report_figures/`.

---

## Troubleshooting & edge cases

- **Missing CSV**: Ensure the file exists and path is correct.
- **Too few normal rows for ML training**: Agent may print  
  `"Falling back to rule-based detection — not enough normal rows for ML training"`.
- **Timestamp parse errors**: Skip affected rows; check CSV formatting.
- **NaNs**: All-sensor-NaN rows are labeled `unknown`. Partial NaNs are handled in detection.

---

## Reproducibility & randomness

- Controlled via `--seed` (default **42**).
- Setting the same seed ensures identical datasets and agent outputs across runs.
- Change the seed with `--seed <int>`.

---

## AI tools & credits

- **ChatGPT (free)**: Prompt Assistant, code generation, docstrings, and README drafting.

Document usage in reports with screenshots or logs. 

---

## Deliverables & submission checklist

Required files for submission:

- `generate_smartfactory_dataset.py`
- `smartfactory_alert_agent.py`
- `requirements.txt`
- `data/smartfactory_dataset.csv` (sample)
- `report.pdf` (2–3 pages)
- `README.md`
- Demo screenshots of alerts and outputs

---    
    
## Extending the project (optional ideas)

- Add unique device IDs for multi-machine scenarios.
- Explore time-series models (LSTM, Prophet) for anomaly prediction.
- Build a web dashboard to visualize alerts.
- Aggregate alerts over a time window for operational insights.

