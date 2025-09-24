#!/usr/bin/env python3
"""
generate_smartfactory_dataset.py

Generate a synthetic Smart Factory CSV dataset with exact fraction of anomalies
based on strict anomaly thresholds.

Usage:
python generate_smartfactory_dataset.py --rows 500 --freq 1min --anomaly_pct 0.1 --out ./data/smart_factory_data.csv
"""

from __future__ import annotations
import argparse, os
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
import pandas as pd

# Normal ranges (for base values)
TEMP_MIN, TEMP_MAX = 45.0, 50.0
PRESSURE_MIN, PRESSURE_MAX = 1.00, 1.05
VIBRATION_MIN, VIBRATION_MAX = 0.02, 0.04

# Strict anomaly thresholds
TEMP_LOW, TEMP_HIGH = 43.0, 52.0
PRESSURE_LOW, PRESSURE_HIGH = 0.97, 1.08
VIBRATION_HIGH = 0.07

# Noise
TEMP_SIGMA, PRESSURE_SIGMA, VIBRATION_SIGMA = 0.1, 0.005, 0.003

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--freq", choices=["1min","5min"], default="1min")
    parser.add_argument("--anomaly_pct", type=float, default=0.08)
    parser.add_argument("--missing_pct", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="./data/smartfactory_dataset_exact.csv")
    return parser.parse_args()

def ensure_outdir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def generate_timestamps(start: Optional[datetime], count: int, freq: str) -> List[str]:
    if start is None:
        start = datetime.now().replace(second=0, microsecond=0)
    delta = timedelta(minutes=1 if freq=="1min" else 5)
    return [(start + i*delta).strftime("%Y-%m-%d %H:%M:%S") for i in range(count)]

def is_abnormal(temp, pres, vib) -> bool:
    """Check strict anomaly thresholds."""
    if temp is not None and not np.isnan(temp):
        if temp < TEMP_LOW or temp > TEMP_HIGH:
            return True
    if pres is not None and not np.isnan(pres):
        if pres < PRESSURE_LOW or pres > PRESSURE_HIGH:
            return True
    if vib is not None and not np.isnan(vib):
        if vib > VIBRATION_HIGH:
            return True
    return False

def main():
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows = args.rows
    anomaly_target = int(round(rows*args.anomaly_pct))
    missing_rows = int(round(rows*args.missing_pct))
    ensure_outdir(args.out)

    # Generate baseline normal values
    temps = rng.uniform(TEMP_MIN, TEMP_MAX, size=rows) + rng.normal(0,TEMP_SIGMA,size=rows)
    pressures = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, size=rows) + rng.normal(0,PRESSURE_SIGMA,size=rows)
    vibrations = rng.uniform(VIBRATION_MIN, VIBRATION_MAX, size=rows) + rng.normal(0,VIBRATION_SIGMA,size=rows)

    # Track anomalous rows
    anomaly_rows = set()

    # Inject anomalies until target count
    while len(anomaly_rows) < anomaly_target:
        idx = rng.integers(0, rows)
        if idx in anomaly_rows:
            continue
        sensor = rng.integers(0,3)
        if sensor==0:
            # Temp anomaly
            if rng.random() < 0.5:
                temps[idx] = rng.uniform(30, TEMP_LOW-0.01)  # low
            else:
                temps[idx] = rng.uniform(TEMP_HIGH+0.01, TEMP_HIGH+8)  # high
        elif sensor==1:
            # Pressure anomaly
            if rng.random() < 0.5:
                pressures[idx] = rng.uniform(0.80, PRESSURE_LOW-0.01)
            else:
                pressures[idx] = rng.uniform(PRESSURE_HIGH+0.01, PRESSURE_HIGH+0.2)
        else:
            # Vibration anomaly
            vibrations[idx] = rng.uniform(VIBRATION_HIGH+0.01, VIBRATION_HIGH+0.2)
        anomaly_rows.add(idx)

    # Inject missing values
    all_idx = list(range(rows))
    rng.shuffle(all_idx)
    for r in all_idx[:missing_rows]:
        num_sensors = rng.integers(1,4)
        missing_sensors = rng.choice([0,1,2], size=num_sensors, replace=False)
        for s in missing_sensors:
            if s==0: temps[r]=np.nan
            elif s==1: pressures[r]=np.nan
            else: vibrations[r]=np.nan

    # Assign labels based on strict anomaly thresholds
    labels = []
    for i in range(rows):
        t,p,v = temps[i], pressures[i], vibrations[i]
        if all(np.isnan([t,p,v])):
            labels.append("unknown")
        elif is_abnormal(t,p,v):
            labels.append("abnormal")
        else:
            labels.append("normal")

    # Save DataFrame
    df = pd.DataFrame({
        "timestamp": generate_timestamps(None, rows, args.freq),
        "temp": temps,
        "pressure": pressures,
        "vibration": vibrations,
        "label": labels
    })
    df.to_csv(args.out, index=False)

    print(f"Saved CSV to {args.out}")
    print(f"Counts -> normal: {sum(l=='normal' for l in labels)}, abnormal: {sum(l=='abnormal' for l in labels)}, unknown: {sum(l=='unknown' for l in labels)}")

    print('Sample data:')
    print(df.head(5))

    print('Summary statistics:')
    print(df.describe())

if __name__=="__main__":
    main()
