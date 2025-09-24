#!/usr/bin/env python3
"""
generate_smartfactory_dataset.py

Generate a synthetic Smart Factory CSV dataset with timestamped sensor readings and labeled anomalies.

Usage example:
    python generate_smartfactory_dataset.py --rows 500 --freq 1min --anomaly_pct 0.1 --out ./data/smart_factory_data.csv

Creates a CSV with header: timestamp,temp,pressure,vibration,label
Default output path: ./data/smartfactory_dataset.csv
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Sensor normal ranges and anomaly thresholds (exact numeric values required)
TEMP_NORMAL_MIN = 45.0
TEMP_NORMAL_MAX = 50.0
TEMP_ANOMALY_LOW = 43.0  # anomaly if < 43.0
TEMP_ANOMALY_HIGH = 52.0  # anomaly if > 52.0

PRESSURE_NORMAL_MIN = 1.00
PRESSURE_NORMAL_MAX = 1.05
PRESSURE_ANOMALY_LOW = 0.97
PRESSURE_ANOMALY_HIGH = 1.08

VIBRATION_NORMAL_MIN = 0.02
VIBRATION_NORMAL_MAX = 0.04
VIBRATION_ANOMALY_HIGH = 0.07  # anomaly if > 0.07

# Measurement noise sigmas
TEMP_NOISE_SIGMA = 0.1
PRESSURE_NOISE_SIGMA = 0.005
VIBRATION_NOISE_SIGMA = 0.003


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic Smart Factory dataset CSV.")
    parser.add_argument("--rows", type=int, default=200,
                        help="Number of rows to generate (allowed range 100–500). Default: 200")
    parser.add_argument("--freq", type=str, choices=("1min", "5min"), default="1min",
                        help="Interval between timestamps. Choices: 1min or 5min. Default: 1min")
    parser.add_argument("--anomaly_pct", type=float, default=0.08,
                        help="Percent of rows that contain at least one injected anomaly. Default: 0.08")
    parser.add_argument("--missing_pct", type=float, default=0.02,
                        help="Percent of rows with at least one missing value. Default: 0.02")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default: 42")
    parser.add_argument("--out", type=str, default="./data/smartfactory_dataset.csv",
                        help="Output CSV path. Default: ./data/smartfactory_dataset.csv")
    return parser.parse_args()


def ensure_outdir(path: str) -> None:
    """Create directories for the output path if they don't exist."""
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def generate_timestamps(start: Optional[datetime], count: int, freq: str) -> List[str]:
    """Generate a list of timestamp strings in 'YYYY-MM-DD HH:MM:SS' format."""
    if start is None:
        start = datetime.now().replace(second=0, microsecond=0)
    delta = timedelta(minutes=1) if freq == "1min" else timedelta(minutes=5)
    timestamps = [(start + i * delta).strftime("%Y-%m-%d %H:%M:%S") for i in range(count)]
    return timestamps


def in_normal_ranges(temp: float, pressure: float, vibration: float) -> bool:
    """Return True if all three sensors are within their normal ranges."""
    t_ok = TEMP_NORMAL_MIN <= temp <= TEMP_NORMAL_MAX
    p_ok = PRESSURE_NORMAL_MIN <= pressure <= PRESSURE_NORMAL_MAX
    v_ok = VIBRATION_NORMAL_MIN <= vibration <= VIBRATION_NORMAL_MAX
    return t_ok and p_ok and v_ok


def is_out_of_range_for_row(temp: Optional[float], pressure: Optional[float], vibration: Optional[float]) -> bool:
    """
    Determine whether any non-NaN sensor is outside its normal range.
    Returns True if any non-NaN sensor is abnormal.
    """
    # temp
    if temp is not None and not np.isnan(temp):
        if temp < TEMP_NORMAL_MIN or temp > TEMP_NORMAL_MAX:
            return True
    # pressure
    if pressure is not None and not np.isnan(pressure):
        if pressure < PRESSURE_NORMAL_MIN or pressure > PRESSURE_NORMAL_MAX:
            return True
    # vibration
    if vibration is not None and not np.isnan(vibration):
        if vibration < VIBRATION_NORMAL_MIN or vibration > VIBRATION_NORMAL_MAX:
            return True
    return False


def clamp_array(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clamp numpy array values to a min/max range (used rarely; mostly clarity)."""
    return np.minimum(np.maximum(arr, min_val), max_val)


def inject_anomalies(
    temp: np.ndarray,
    pressure: np.ndarray,
    vibration: np.ndarray,
    target_anomaly_rows: int,
    rng: np.random.Generator,
) -> None:
    """
    Inject a diverse mixture of anomalies into the provided arrays in-place.

    Types:
    - Single-point spike (sudden high/low)
    - Gradual drift across a short sequence
    - Stuck sensor (same out-of-range value repeated for several rows)
    - Random outliers (single-row anomalies)

    Strategy:
    - Pick event types with weighted probabilities and apply until target number of unique anomalous rows is reached.
    - Keep track of which row indices have been marked anomalous to meet the target.
    """
    n = len(temp)
    anomalous_rows = set()
    max_attempts = target_anomaly_rows * 20 + 1000
    attempts = 0

    # Weights for diversity (decide sensible proportions)
    # spike: 35%, drift: 30%, stuck: 25%, random outlier: 10%
    event_types = ["spike", "drift", "stuck", "outlier"]
    event_weights = [0.35, 0.30, 0.25, 0.10]

    while len(anomalous_rows) < target_anomaly_rows and attempts < max_attempts:
        attempts += 1
        ev = rng.choice(event_types, p=event_weights)
        # pick a sensor: 0=temp,1=pressure,2=vibration
        sensor = rng.integers(0, 3)

        if ev == "spike":
            # single index spike
            idx = int(rng.integers(0, n))
            # avoid choosing same idx too many times
            if idx in anomalous_rows and rng.random() < 0.5:
                continue
            if sensor == 0:
                # choose high or low spike
                if rng.random() < 0.5:
                    temp[idx] = rng.uniform(30.0, TEMP_ANOMALY_LOW - 0.5)  # low spike
                else:
                    temp[idx] = rng.uniform(TEMP_ANOMALY_HIGH + 0.5, TEMP_ANOMALY_HIGH + 8.0)  # high spike
            elif sensor == 1:
                if rng.random() < 0.5:
                    pressure[idx] = rng.uniform(0.80, PRESSURE_ANOMALY_LOW - 0.01)
                else:
                    pressure[idx] = rng.uniform(PRESSURE_ANOMALY_HIGH + 0.01, PRESSURE_ANOMALY_HIGH + 0.2)
            else:
                vibration[idx] = rng.uniform(VIBRATION_ANOMALY_HIGH + 0.01, VIBRATION_ANOMALY_HIGH + 0.2)
            anomalous_rows.add(idx)

        elif ev == "drift":
            # gradual drift across a small sequence
            length = int(rng.integers(3, min(9, n)))  # 3-8
            start = int(rng.integers(0, max(1, n - length + 1)))
            idxs = list(range(start, start + length))
            if len(anomalous_rows.union(idxs)) - len(anomalous_rows) == 0 and rng.random() < 0.5:
                # not adding new anomalous rows; try another start
                continue
            # For drift, gradually add an offset that crosses anomaly threshold by end
            if sensor == 0:
                # temp drift upward or downward
                direction = 1 if rng.random() < 0.6 else -1
                end_offset = 4.0 * direction + (rng.random() - 0.5) * 1.0  # push beyond threshold
                offsets = np.linspace(0.0, end_offset, length)
                temp[idxs] = temp[idxs] + offsets
            elif sensor == 1:
                direction = 1 if rng.random() < 0.6 else -1
                end_offset = 0.06 * direction + (rng.random() - 0.5) * 0.02
                offsets = np.linspace(0.0, end_offset, length)
                pressure[idxs] = pressure[idxs] + offsets
            else:
                # vibration drift up
                end_offset = 0.05 + rng.random() * 0.04  # ensure crossing >0.07 likely
                offsets = np.linspace(0.0, end_offset, length)
                vibration[idxs] = vibration[idxs] + offsets
            anomalous_rows.update(idxs)

        elif ev == "stuck":
            # stuck sensor: same out-of-range value repeated for several rows
            length = int(rng.integers(4, min(12, n)))  # 4-11
            start = int(rng.integers(0, max(1, n - length + 1)))
            idxs = list(range(start, start + length))
            if len(anomalous_rows.union(idxs)) - len(anomalous_rows) == 0 and rng.random() < 0.5:
                continue
            if sensor == 0:
                stuck_val = rng.uniform(TEMP_ANOMALY_HIGH + 0.5, TEMP_ANOMALY_HIGH + 6.0) if rng.random() < 0.8 else rng.uniform(30.0, TEMP_ANOMALY_LOW - 0.5)
                temp[idxs] = stuck_val
            elif sensor == 1:
                stuck_val = rng.uniform(PRESSURE_ANOMALY_HIGH + 0.01, PRESSURE_ANOMALY_HIGH + 0.15) if rng.random() < 0.8 else rng.uniform(0.80, PRESSURE_ANOMALY_LOW - 0.01)
                pressure[idxs] = stuck_val
            else:
                stuck_val = rng.uniform(VIBRATION_ANOMALY_HIGH + 0.01, VIBRATION_ANOMALY_HIGH + 0.2)
                vibration[idxs] = stuck_val
            anomalous_rows.update(idxs)

        elif ev == "outlier":
            # random outlier single-row; similar to spike but less frequent
            idx = int(rng.integers(0, n))
            if idx in anomalous_rows and rng.random() < 0.6:
                continue
            if sensor == 0:
                temp[idx] = rng.uniform(TEMP_ANOMALY_HIGH + 0.5, TEMP_ANOMALY_HIGH + 10.0)
            elif sensor == 1:
                pressure[idx] = rng.uniform(PRESSURE_ANOMALY_HIGH + 0.02, PRESSURE_ANOMALY_HIGH + 0.3)
            else:
                vibration[idx] = rng.uniform(VIBRATION_ANOMALY_HIGH + 0.02, VIBRATION_ANOMALY_HIGH + 0.3)
            anomalous_rows.add(idx)

    # End: ensure we have at least target_anomaly_rows marked; if not, force-add some spikes
    if len(anomalous_rows) < target_anomaly_rows:
        needed = target_anomaly_rows - len(anomalous_rows)
        available = [i for i in range(n) if i not in anomalous_rows]
        rng.shuffle(available)
        for idx in available[:needed]:
            # force a temperature spike high
            temp[idx] = TEMP_ANOMALY_HIGH + 3.0 + rng.random() * 3.0
            anomalous_rows.add(idx)


def assign_labels_after_missing(
    temp_arr: np.ndarray, pressure_arr: np.ndarray, vibration_arr: np.ndarray
) -> List[str]:
    """
    Assign label per row according to:
    - 'abnormal' if any of the non-NaN sensor values in that row is outside the normal ranges;
    - 'normal' otherwise;
    - 'unknown' if all three are NaN.
    """
    labels: List[str] = []
    n = len(temp_arr)
    for i in range(n):
        t = temp_arr[i]
        p = pressure_arr[i]
        v = vibration_arr[i]
        # if all three are NaN => unknown
        if (np.isnan(t) if t is not None else True) and (np.isnan(p) if p is not None else True) and (np.isnan(v) if v is not None else True):
            labels.append("unknown")
            continue
        # if any non-NaN is out of normal range => abnormal
        if is_out_of_range_for_row(t, p, v):
            labels.append("abnormal")
        else:
            labels.append("normal")
    return labels


def main() -> None:
    """Main entrypoint for script execution."""
    args = parse_args()

    # Basic assertions / checks
    assert 100 <= args.rows <= 500, "rows must be between 100 and 500"
    assert 0.0 <= args.anomaly_pct <= 0.5, "anomaly_pct must be between 0 and 0.5"
    assert 0.0 <= args.missing_pct <= 0.5, "missing_pct must be between 0 and 0.5"

    # Seed RNGs for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows = int(args.rows)
    freq = args.freq
    anomaly_pct = float(args.anomaly_pct)
    missing_pct = float(args.missing_pct)
    out_path = args.out

    ensure_outdir(out_path)

    # Generate timestamps
    timestamps = generate_timestamps(start=None, count=rows, freq=freq)

    # Baseline normal sensor values (uniform inside normal range) + Gaussian noise
    temps_base = rng.uniform(TEMP_NORMAL_MIN, TEMP_NORMAL_MAX, size=rows)
    pressures_base = rng.uniform(PRESSURE_NORMAL_MIN, PRESSURE_NORMAL_MAX, size=rows)
    vibrations_base = rng.uniform(VIBRATION_NORMAL_MIN, VIBRATION_NORMAL_MAX, size=rows)

    # Add small Gaussian measurement noise to normal readings
    temps = temps_base + rng.normal(loc=0.0, scale=TEMP_NOISE_SIGMA, size=rows)
    pressures = pressures_base + rng.normal(loc=0.0, scale=PRESSURE_NOISE_SIGMA, size=rows)
    vibrations = vibrations_base + rng.normal(loc=0.0, scale=VIBRATION_NOISE_SIGMA, size=rows)

    # Determine number of anomaly rows
    target_anomaly_rows = int(round(rows * anomaly_pct))
    # Ensure at least 1 anomaly row if anomaly_pct > 0 and rows >= 1
    if anomaly_pct > 0 and target_anomaly_rows == 0:
        target_anomaly_rows = 1

    # Inject anomalies (variety)
    inject_anomalies(temps, pressures, vibrations, target_anomaly_rows, rng)

    # Now inject missing values: choose number of rows that will have at least one missing sensor
    num_missing_rows = int(round(rows * missing_pct))
    if missing_pct > 0 and num_missing_rows == 0:
        num_missing_rows = 1  # ensure at least one if pct > 0

    all_indices = list(range(rows))
    rng.shuffle(all_indices)
    missing_rows_candidates = set(all_indices[:num_missing_rows])

    # To keep labels consistent with injected anomalies:
    # For each missing row, randomly set 1-3 sensors to NaN but ensure that if the row is currently abnormal
    # we do not set all out-of-range sensors to NaN (we must preserve at least one non-NaN out-of-range reading).
    for r in missing_rows_candidates:
        # determine which sensors are currently out-of-range (before applying missing)
        out_of_range_sensors = []
        if temps[r] < TEMP_NORMAL_MIN or temps[r] > TEMP_NORMAL_MAX:
            out_of_range_sensors.append(0)
        if pressures[r] < PRESSURE_NORMAL_MIN or pressures[r] > PRESSURE_NORMAL_MAX:
            out_of_range_sensors.append(1)
        if vibrations[r] < VIBRATION_NORMAL_MIN or vibrations[r] > VIBRATION_NORMAL_MAX:
            out_of_range_sensors.append(2)

        # choose how many sensors to set NaN on this row (at least 1)
        num_to_null = int(rng.integers(1, 4))  # 1-3
        sensors_choice = list(rng.choice([0, 1, 2], size=num_to_null, replace=False))

        # If this row currently has anomaly(s), ensure we do not nullify all out_of_range_sensors
        if len(out_of_range_sensors) > 0:
            # If sensors_choice covers all out_of_range_sensors, drop one of them from sensors_choice
            if set(out_of_range_sensors).issubset(set(sensors_choice)):
                # pick one out_of_range_sensor to keep
                keep = int(rng.choice(out_of_range_sensors))
                if keep in sensors_choice:
                    sensors_choice.remove(keep)
                # ensure at least one sensor remains nullable; if sensors_choice becomes empty, pick a different sensor to null
                if len(sensors_choice) == 0:
                    # pick a sensor that is not the kept one to null
                    other = [s for s in [0, 1, 2] if s != keep]
                    sensors_choice = [int(rng.choice(other))]

        # Apply NaNs
        for s in sensors_choice:
            if s == 0:
                temps[r] = np.nan
            elif s == 1:
                pressures[r] = np.nan
            else:
                vibrations[r] = np.nan

    # Final label assignment
    labels = assign_labels_after_missing(temps, pressures, vibrations)

    # Build DataFrame with required header order: timestamp,temp,pressure,vibration,label
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temp": temps,
        "pressure": pressures,
        "vibration": vibrations,
        "label": labels
    })

    # Save CSV
    df.to_csv(out_path, index=False, columns=["timestamp", "temp", "pressure", "vibration", "label"])

    # Output & verification printed to console
    total_rows = len(df)
    print(f"Saved CSV to: {out_path}")
    print(f"Total rows: {total_rows}")
    counts = df["label"].value_counts(dropna=False).to_dict()
    # Ensure keys for normal/abnormal/unknown exist
    normal_count = counts.get("normal", 0)
    abnormal_count = counts.get("abnormal", 0)
    unknown_count = counts.get("unknown", 0)
    print(f"Counts -> normal: {normal_count}, abnormal: {abnormal_count}, unknown: {unknown_count}")
    # Print first 5 rows
    print("\nSample (first 5 rows):")
    print(df.head(5).to_string(index=False))
    # Summary statistics (mean/std/min/max) for numeric columns
    numeric_stats = df[["temp", "pressure", "vibration"]].agg(["mean", "std", "min", "max"]).transpose()
    print("\nSummary statistics (numeric columns):")
    # Nicely format numeric_stats
    print(numeric_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nDone: saved {total_rows} rows to {out_path}")


if __name__ == "__main__":
    main()