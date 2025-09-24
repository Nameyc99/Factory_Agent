# README.md

## Project Description

The Smart Factory Anomaly Detection project simulates real-time monitoring of industrial sensors in a smart factory environment. The system supports **rule-based** and **machine learning (Isolation Forest)** anomaly detection, providing actionable alerts and visualizations of anomalous behavior. 

A **synthetic dataset** with configurable anomalies and missing values can be generated to test the pipeline end-to-end.

---

## Features

- **Synthetic Dataset Generation**  
  - Script: `generate_smartfactory_dataset.py`  
  - Generates sensor readings (`temp`, `pressure`, `vibration`) with anomalies and optional missing values.
  - Configurable row count, sampling frequency, and anomaly percentage.

- **Data Preprocessing Pipeline**  
  - Script: `preprocess_data.py`  
  - Handles missing value imputation (`mean`, `median`, `ffill`, `bfill`, `drop`).  
  - Supports scaling (`standard`, `minmax`) and train/test split.  
  - Saves cleaned datasets and scaler objects for downstream tasks.

- **Isolation Forest Training**  
  - Script: `train_isolation_forest.py`  
  - Trains an Isolation Forest model on preprocessed data.  
  - Supports grid search for hyperparameter tuning.  
  - Produces evaluation metrics, confusion matrix, ROC AUC, and a recommended threshold.

- **Real-Time Simulation Agent**  
  - Script: `smartfactory_alert_agent.py`  
  - Supports **rule-based detection** configurable via JSON rules.  
  - Supports **ML-based detection** using pre-trained Isolation Forest.  
  - Prints console alerts with human-friendly suggestions for rule-based anomalies.  
  - Generates plots for each feature highlighting anomalies (rule, ML, or both).  

---

## Installation

1. **Python Version**: `>=3.11`  
2. **Required Packages**:

```bash
pip install pandas numpy scikit-learn joblib matplotlib
```

3. **Optional Virtual Environment:**
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Usage Examples

### 1. Generate Synthetic Dataset

```bash
python generate_smartfactory_dataset.py \
  --rows 500 \
  --freq 1min \
  --anomaly_pct 0.1 \
  --out ./data/smart_factory_data.csv
```

### 2. Preprocess 

```
python preprocess_data.py \
  --input ./data/smart_factory_data.csv \
  --output ./data/smart_factory_preprocessed.csv \
  --impute ffill \
  --scale standard \
  --split 0.8 \
  --report \
```

### 3. Train Isolation Forest

```
python train_isolation_forest.py \
  --train ./out/train.csv \
  --test ./out/test.csv \
  --scaler ./out/scaler.joblib \
  --output_model ./out/isolation_forest.joblib \
  --contamination 0.01 \
  --n_estimators 100 \
  --report
```

### 4. Train Isolation Forest with Grid Search
```
python train_isolation_forest.py \
  --train ./out/train.csv \
  --test ./out/test.csv \
  --scaler ./out/scaler.joblib \
  --output_model ./out/isolation_forest.joblib \
  --grid_search \
  --report
 ```


### 5. Run Alert Agent (Rule + ML Detection)
```
python smartfactory_alert_agent.py \
  --input ./data/smart_factory_data.csv \
  --scaler ./out/scaler.joblib \
  --model ./out/isolation_forest.joblib \
  --rules ./config/rules.json \
  --out_dir ./out \
  --report \
  --delay 0.1 \
  --use_sleep
```

### 6. Run Alert Agent (Rule-Only Detection)
```
python smartfactory_alert_agent.py \
  --input ./data/smart_factory_data.csv \
  --scaler ./out/scaler.joblib \
  --rules ./config/rules.json \
  --out_dir ./out \
  --report
```

### 7. Limit Number of Rows (for Testing)
```
python smartfactory_alert_agent.py \
  --input ./data/smart_factory_data.csv \
  --scaler ./out/scaler.joblib \
  --model ./out/isolation_forest.joblib \
  --rules ./config/rules.json \
  --limit 50 \
  --out_dir ./out \
  --report
```

Here’s the **Project Structure** section in Markdown, standalone:

## Project Structure

```
smart-factory-anomaly-detection/
│
├─ data/                       # Synthetic datasets generated
│   └─ smart_factory_data.csv
|   └─smart_factory_preprocessed.csv
│
├─ out/                        # Output artifacts (preprocessed data, models, alerts, plots)
│   ├─ train.csv
│   ├─ test.csv
│   ├─ scaler.joblib
│   ├─ isolation\_forest.joblib
│   ├─ alerts.csv
│   ├─ alerts\_summary.json
│   ├─ threshold.json
│   └─ plots/
│       ├─ temp\_anomalies.png
│       ├─ pressure\_anomalies.png
│       └─ vibration\_anomalies.png
│
├─ config/                     # Configuration files
│   └─ rules.json              # Rule-based alert thresholds and severity
│
├─ generate\_smartfactory\_dataset.py   # Script to generate synthetic sensor data with anomalies
├─ preprocess\_data.py                 # Preprocessing pipeline: imputation, scaling, train/test split
├─ train\_isolation\_forest.py          # Train Isolation Forest model on preprocessed data
├─ smartfactory\_alert\_agent.py        # Real-time alert agent: rule-based and ML-based detection
└─ README.md                          # Project documentation

```

**Description of each script/file:**

- **generate_smartfactory_dataset.py**: Generates synthetic sensor data (`temp`, `pressure`, `vibration`) with normal, abnormal, and unknown labels. Can inject missing values.

- **preprocess_data.py**: Handles missing value imputation, scaling (`StandardScaler` or `MinMaxScaler`), and splits data into train/test sets. Saves processed datasets and scaler object.

- **train_isolation_forest.py**: Trains an Isolation Forest on preprocessed data, optionally performs grid search, evaluates performance, and saves the trained model and threshold info.

- **smartfactory_alert_agent.py**: Simulates real-time anomaly detection. Supports:
  - Rule-based detection using JSON-configured thresholds.
  - ML-based detection using a pre-trained Isolation Forest.
  - Alerts with console messages and human-friendly suggestions.
  - Plots each feature with anomalies marked.

- **config/rules.json**: JSON file defining thresholds, operators, severity levels for rule-based detection.

- **out/**: Directory for all artifacts, including processed datasets, trained models, alerts logs, plots, and recommended thresholds.

- **README.md**: Documentation describing usage, features, configuration, and project workflow.


Here’s the **Configuration** section in Markdown, standalone:

## Configuration

### Rule-based Alerts JSON

The rule-based detection system uses a JSON file to define thresholds, operators, and alert severity for each feature. Example structure:

```json
{
  "thresholds": {
    "temp": {"anomaly_low": 43.0, "anomaly_high": 52.0},
    "pressure": {"anomaly_low": 0.97, "anomaly_high": 1.08},
    "vibration": {"anomaly_high": 0.07}
  },
  "operators": {
    "temp": "outside_range",
    "pressure": "outside_range",
    "vibration": "above"
  },
  "combine": "any",
  "alert_meta": {
    "severity": {
      "temp": "high",
      "pressure": "high",
      "vibration": "medium"
    }
  }
}
````

**Field explanations:**

* `thresholds`: Defines numeric thresholds for each feature.

  * Use `anomaly_low` and/or `anomaly_high` for ranges.
  * Use only `anomaly_high` or `anomaly_low` for one-sided thresholds.
* `operators`: Specifies how to evaluate a feature against its thresholds.

  * Supported values: `above`, `below`, `outside_range`, `>`, `<`, `>=`, `<=`, `==`, `!=`.
* `combine`: How per-feature alerts are aggregated into a row-level alert.

  * `"any"`: triggers if any feature exceeds threshold.
  * `"all"`: triggers only if all features exceed thresholds.
* `alert_meta`: Optional metadata, including `severity` for each feature. Used in console messages and alert logs.

### Adjusting Thresholds and Severity

1. Open `config/rules.json` in a text editor.
2. Modify numeric values for `anomaly_low` and `anomaly_high`.
3. Change `operators` if a feature requires a different rule logic.
4. Adjust `severity` levels to reflect operational importance (`low`, `medium`, `high`).
5. Save the file and provide its path to `smartfactory_alert_agent.py` using the `--rules` argument.

> **Note:** If a feature specified in the rules is missing in the dataset, the agent will log a warning and ignore that feature.


## Output / Artifacts

The project generates several artifacts during dataset generation, preprocessing, model training, and real-time simulation. The default output directory is `./out`, but it can be changed via CLI arguments.

| Artifact | Location / File | Description |
|----------|----------------|-------------|
|Generated dataset|./data/smart_factory_data.csv|Generated data|
| Cleaned dataset (train) | `./out/train.csv` | Preprocessed training data after missing value imputation and scaling. |
| Cleaned dataset (test) | `./out/test.csv` | Preprocessed test data for model evaluation. |
| Scaler object | `./out/scaler.joblib` | Joblib object storing the fitted scaler used for numeric features. |
| Trained Isolation Forest model | `./out/isolation_forest.joblib` | Joblib object of trained Isolation Forest for anomaly detection. |
| Threshold recommendations | `./out/threshold.json` | Optional JSON containing recommended threshold for model-based alerts. |
| Alert logs | `./out/alerts.csv` | CSV of detected anomalies with timestamp, features triggered, raw & scaled values, rule & model flags, severity, and suggestions. |
| Alert summary | `./out/alerts_summary.json` | Aggregated counts, first/last alert timestamps, and per-feature alert statistics. |
| Feature anomaly plots | `./out/plots/<feature>_anomalies.png` | Time-series plots for each numeric feature marking rule-based, ML-based, and combined anomalies. |

**Notes:**

- All CSVs are comma-delimited with headers.
- Joblib objects (`.joblib`) can be loaded with `joblib.load()` for inference or further processing.
- Plots are saved as PNG files and can be viewed in any image viewer.
- The alert logs and summary are updated in real-time during simulation.
