#!/usr/bin/env python3
"""
Smart Factory Alert Agent - Real-Time Simulation with per-row CLI alerts

Usage example:
python smartfactory_alert_agent.py \
  --input ./data/smart_factory_data.csv \
  --scaler ./out/scaler.joblib \
  --model ./out/isolation_forest.joblib \
  --rules ./config/rules.json \
  --out_dir ./out \
  --report
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("smartfactory_alert_agent")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -------------------------
# Utilities
# -------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        logger.error(f"Input CSV not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    required = ["timestamp", "temp", "pressure", "vibration"]
    for c in required:
        if c not in df.columns:
            logger.error(f"Missing column '{c}' in CSV")
            sys.exit(1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        logger.error("Invalid timestamp detected")
        sys.exit(1)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        logger.error(f"Rules file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        rules = json.load(f)
    rules.setdefault("combine", "any")
    rules.setdefault("alert_meta", {})
    return rules


def load_scaler(path: str):
    if not os.path.isfile(path):
        logger.error(f"Scaler file not found: {path}")
        sys.exit(1)
    return joblib.load(path)


def load_model(path: Optional[str]) -> Optional[BaseEstimator]:
    if not path:
        return None
    if not os.path.isfile(path):
        logger.error(f"Model file not found: {path}")
        sys.exit(1)
    return joblib.load(path)


# -------------------------
# Threshold scaling
# -------------------------
def transform_threshold_value(idx: int, val: float, scaler, feature_order: List[str]) -> float:
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return float((val - scaler.mean_[idx]) / scaler.scale_[idx])
    sample = pd.DataFrame([[0.0]*len(feature_order)], columns=feature_order)
    sample.iloc[0, idx] = val
    return float(scaler.transform(sample)[0, idx])


def transform_thresholds(rules: Dict[str, Any], scaler, feature_order: List[str]) -> Dict[str, Any]:
    thresholds = rules.get("thresholds", {})
    scaled = {"thresholds": {}, "operators": rules.get("operators", {}), "combine": rules.get("combine"), "alert_meta": rules.get("alert_meta")}
    feature_to_idx = {f:i for i,f in enumerate(feature_order)}
    for feat, thr in thresholds.items():
        if feat not in feature_to_idx:
            continue
        idx = feature_to_idx[feat]
        scaled_thr = {}
        if "anomaly_low" in thr:
            scaled_thr["anomaly_low"] = transform_threshold_value(idx, thr["anomaly_low"], scaler, feature_order)
        if "anomaly_high" in thr:
            scaled_thr["anomaly_high"] = transform_threshold_value(idx, thr["anomaly_high"], scaler, feature_order)
        if "value" in thr:
            scaled_thr["value"] = transform_threshold_value(idx, thr["value"], scaler, feature_order)
        scaled["thresholds"][feat] = scaled_thr
    return scaled

# -------------------------
# Rule evaluation
# -------------------------
def eval_operator(val: float, operator: str, thr: Dict[str, float]) -> Tuple[bool, str]:
    op = operator.lower()
    detail = ""
    triggered = False
    if op in (">", "above"):
        if "anomaly_high" in thr: triggered = val > thr["anomaly_high"]; detail=f"> {thr['anomaly_high']:.4f}"
    elif op in ("<", "below"):
        if "anomaly_low" in thr: triggered = val < thr["anomaly_low"]; detail=f"< {thr['anomaly_low']:.4f}"
    elif op=="outside_range":
        low, high = thr.get("anomaly_low"), thr.get("anomaly_high")
        if low is not None and high is not None:
            triggered = val < low or val > high
            detail=f"outside [{low:.4f},{high:.4f}]"
    return triggered, detail

# -------------------------
# Suggestion mapping
# -------------------------
SUGGESTION_MAP = {
    "temp": "Check cooling system.",
    "pressure": "Inspect pumps and valves.",
    "vibration": "Check bearings and mounting.",
}

def generate_suggestion(features: List[str], alert_meta: Dict[str, Any]) -> str:
    parts = [SUGGESTION_MAP.get(f) for f in features if SUGGESTION_MAP.get(f)]
    return " ".join(parts) if parts else "Investigate sensors."

# -------------------------
# Alert CSV
# -------------------------
def ensure_alerts_csv(path: str):
    header = ["timestamp","row_index","feature","raw_values","scaled_values","rule_triggered","rule_details","model_triggered","model_score","model_prediction","severity","suggestion"]
    if not os.path.exists(path):
        with open(path,"w") as f:
            f.write(",".join(header)+"\n")

def append_alert_csv(path: str, record: Dict[str,Any]):
    ensure_alerts_csv(path)
    row = [
        str(record.get("timestamp")),
        str(record.get("row_index")),
        str(record.get("feature")),
        json.dumps(record.get("raw_values")),
        json.dumps(record.get("scaled_values")),
        str(record.get("rule_triggered")),
        json.dumps(record.get("rule_details")),
        str(record.get("model_triggered")),
        "" if record.get("model_score") is None else f"{record.get('model_score'):.6f}",
        "" if record.get("model_prediction") is None else str(record.get("model_prediction")),
        str(record.get("severity") or ""),
        json.dumps(record.get("suggestion") or ""),
    ]
    line=",".join('"' + s.replace('"','""') + '"' for s in row)
    with open(path,"a") as f:
        f.write(line+"\n"); f.flush()

# -------------------------
# Plotting
# -------------------------
def plot_feature_anomalies(timestamps, raw_vals, rule_flags, ml_flags, out_path, feature_name, thresholds=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.plot(timestamps, raw_vals,"-",label=f"{feature_name} raw")
    if thresholds:
        if "low" in thresholds: plt.axhline(thresholds["low"], color="blue", linestyle="--", label="low threshold")
        if "high" in thresholds: plt.axhline(thresholds["high"], color="green", linestyle="--", label="high threshold")
    rule_idx=[i for i,v in enumerate(rule_flags) if v and not ml_flags[i]]
    ml_idx=[i for i,v in enumerate(ml_flags) if v and not rule_flags[i]]
    both_idx=[i for i,v in enumerate(rule_flags) if v and ml_flags[i]]
    if rule_idx: plt.scatter([timestamps[i] for i in rule_idx], [raw_vals[i] for i in rule_idx], color="red", label="rule")
    if ml_idx: plt.scatter([timestamps[i] for i in ml_idx], [raw_vals[i] for i in ml_idx], color="orange", label="ml")
    if both_idx: plt.scatter([timestamps[i] for i in both_idx], [raw_vals[i] for i in both_idx], color="purple", label="both")
    plt.xlabel("timestamp"); plt.ylabel(feature_name)
    plt.title(f"{feature_name} anomalies")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

# -------------------------
# Row processing
# -------------------------
def process_row(idx, row, scaled_rules, operators, combine, alert_meta, model, feature_order, scaler):
    numeric={f: float(row[f]) for f in feature_order}

    # Scale row
    row_arr = np.array([numeric[f] for f in feature_order]).reshape(1, -1)
    scaled_arr = scaler.transform(row_arr)
    scaled_vals={f: float(scaled_arr[0,i]) for i,f in enumerate(feature_order)}

    rule_flags, rule_details={},{}
    for f in feature_order:
        if f in scaled_rules["thresholds"]:
            thr=scaled_rules["thresholds"][f]; op=operators.get(f,"outside_range")
            trig, det=eval_operator(scaled_vals[f], op, thr)
            rule_flags[f]=trig; rule_details[f]=det
        else:
            rule_flags[f]=False; rule_details[f]=""
    rule_triggered = any(rule_flags.values()) if combine=="any" else all(rule_flags.values())

    # ML detection
    ml_flags = {f:False for f in feature_order}
    model_triggered=False
    model_score=None
    model_pred=None
    if model:
        model_pred_arr = model.predict(scaled_arr)[0]
        model_triggered = model_pred_arr == -1
        if hasattr(model, "decision_function"):
            model_score = float(-model.decision_function(scaled_arr)[0])
        else:
            model_score = None
        # approximate attribution
        z=(scaled_arr[0]-scaler.mean_)/scaler.scale_
        for i,f in enumerate(feature_order): ml_flags[f]=abs(z[i])>1.5

    # Suggestion
    suggestion=None
    if rule_triggered:
        triggered_feats=[f for f,v in rule_flags.items() if v]
        suggestion=generate_suggestion(triggered_feats, alert_meta)

    # Build alert record
    alert_record={
        "timestamp":str(row["timestamp"]),
        "row_index":int(idx),
        "feature":",".join([f for f,v in rule_flags.items() if v]) if rule_triggered else "model",
        "raw_values":numeric,
        "scaled_values":scaled_vals,
        "rule_triggered":rule_triggered,
        "rule_details":rule_details,
        "model_triggered":model_triggered,
        "model_score":model_score,
        "model_prediction":-1 if model_triggered else 1,
        "severity":None,
        "suggestion":suggestion
    }
    return alert_record, rule_flags, ml_flags

# -------------------------
# Simulation
# -------------------------
def simulate_realtime(
    df: pd.DataFrame,
    scaler,
    rules: Dict[str, Any],
    model,
    out_dir: str,
    delay: float,
    use_sleep: bool,
    limit: Optional[int],
    seed: int,
    verbose: bool,
) -> Dict[str, Any]:

    np.random.seed(seed)
    feature_order = ["temp", "pressure", "vibration"]
    scaled_rules = transform_thresholds(rules, scaler, feature_order)
    operators = rules.get("operators", {})
    combine = rules.get("combine", "any")
    alert_meta = rules.get("alert_meta", {})

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    alerts_csv = os.path.join(out_dir, "alerts.csv")
    ensure_alerts_csv(alerts_csv)

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    df_proc = df.copy()
    df_proc[feature_order] = imputer.fit_transform(df_proc[feature_order])

    # Scaling
    df_proc_scaled = pd.DataFrame(scaler.transform(df_proc[feature_order]), columns=feature_order)

    total_rows=0; rule_alert_count=0; ml_only_count=0; combined_count=0
    per_feature_counts={f:0 for f in feature_order}
    plot_data={f:{"timestamps":[],"raw":[],"rule_flags":[],"ml_flags":[]} for f in feature_order}

    limit_rows = min(int(limit) if limit else len(df_proc), len(df_proc))
    for idx in range(limit_rows):
        row=df_proc.iloc[idx]
        total_rows+=1

        alert_record, rule_flags, ml_flags = process_row(
            idx, row, scaled_rules, operators, combine, alert_meta, model, feature_order, scaler
        )

        is_rule = alert_record["rule_triggered"]
        is_model = alert_record["model_triggered"]

        if is_rule and is_model:
            combined_count+=1; alert_type="RULE+ML"
        elif is_rule:
            rule_alert_count+=1; alert_type="RULE"
        elif is_model:
            ml_only_count+=1; alert_type="ML"
        else:
            alert_type="NORMAL"

        for f in feature_order:
            if rule_flags.get(f): per_feature_counts[f]+=1

        append_alert_csv(alerts_csv, alert_record)

        for f in feature_order:
            plot_data[f]["timestamps"].append(row["timestamp"])
            plot_data[f]["raw"].append(float(row[f]))
            # Use rule_triggered and row-level model_triggered for plotting
            plot_data[f]["rule_flags"].append(rule_flags.get(f, False))
            plot_data[f]["ml_flags"].append(alert_record['model_triggered'])  # <- row-level ML


        score_str = f"{alert_record['model_score']:.6f}" if alert_record['model_score'] is not None else "n/a"
        sug_str = alert_record['suggestion'] if alert_record['suggestion'] else "None"
        print(f"[{alert_type}] {alert_record['timestamp']} "
            f"temp={row['temp']:.3f} pressure={row['pressure']:.3f} vibration={row['vibration']:.3f} "
            f"ML_score={score_str} Suggestion: {sug_str}")

        if use_sleep and delay>0: time.sleep(delay)

    # Plotting
    feature_thresholds={}
    for f,thresh in rules.get("thresholds", {}).items():
        ft={}
        if "anomaly_low" in thresh: ft["low"]=thresh["anomaly_low"]
        if "anomaly_high" in thresh: ft["high"]=thresh["anomaly_high"]
        feature_thresholds[f]=ft
    plot_paths={}
    for f in feature_order:
        ppath=os.path.join(plots_dir,f"{f}_anomalies.png")
        plot_feature_anomalies(
            plot_data[f]["timestamps"], plot_data[f]["raw"], plot_data[f]["rule_flags"], plot_data[f]["ml_flags"], ppath, f, thresholds=feature_thresholds.get(f)
        )
        plot_paths[f]=ppath

    summary={
        "total_rows_processed": total_rows,
        "rule_alerts": rule_alert_count,
        "ml_only_alerts": ml_only_count,
        "combined_alerts": combined_count,
        "per_feature_counts": per_feature_counts,
        "plots": plot_paths,
        "alerts_csv": alerts_csv,
    }

    summary_path=os.path.join(out_dir,"alerts_summary.json")
    with open(summary_path,"w",encoding="utf-8") as sf:
        json.dump(summary, sf, default=str, indent=2)

    return summary

# -------------------------
# CLI
# -------------------------
def parse_args():
    p=argparse.ArgumentParser(description="Smart Factory Alert Agent - realtime simulation")
    p.add_argument("--input",type=str,required=True)
    p.add_argument("--scaler",type=str,required=True)
    p.add_argument("--model",type=str,default=None)
    p.add_argument("--rules",type=str,required=True)
    p.add_argument("--out_dir",type=str,default="./out")
    p.add_argument("--report",action="store_true")
    p.add_argument("--delay",type=float,default=0.0)
    p.add_argument("--use_sleep",action="store_true")
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--verbose",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    df=load_data(args.input)
    rules=load_rules(args.rules)
    scaler=load_scaler(args.scaler)
    model=load_model(args.model)
    limit=args.limit if args.limit and args.limit>0 else None
    summary=simulate_realtime(df, scaler, rules, model, args.out_dir,
                              args.delay, args.use_sleep, limit, args.seed, args.verbose)
    if args.report:
        logger.info(json.dumps(summary, indent=2))

if __name__=="__main__":
    main()
