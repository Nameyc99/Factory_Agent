#!/usr/bin/env python3
"""
Preprocess Smart Factory Dataset

Example:
python preprocess_data.py --input ./data/smart_factory_data.csv \
    --output ./data/smart_factory_preprocessed.csv \
    --impute ffill --scale standard --split 0.8 --report
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

REQUIRED_COLUMNS = ["timestamp", "temp", "pressure", "vibration", "label"]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and ensure required columns exist."""
    if not os.path.exists(path):
        sys.exit(f"Error: input file not found at {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        sys.exit(f"Error: failed to read CSV file: {e}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"Error: missing required columns: {missing_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        sys.exit("Error: invalid timestamps found")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def impute_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle missing values with chosen strategy."""
    numeric_cols = ["temp", "pressure", "vibration"]
    before_missing = df[numeric_cols].isna().sum()

    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif strategy == "bfill":
        df[numeric_cols] = df[numeric_cols].bfill().ffill()
    elif strategy == "drop":
        before = len(df)
        df = df.dropna(subset=numeric_cols)
        after = len(df)
        if after < 50:
            sys.exit(
                f"Error: drop strategy removed too many rows ({before}->{after}), aborting."
            )
    else:
        sys.exit(f"Error: unsupported imputation strategy '{strategy}'")

    after_missing = df[numeric_cols].isna().sum()
    return df, before_missing, after_missing


def scale_features(df: pd.DataFrame, method: str, out_dir: str):
    """Apply feature scaling to numeric columns."""
    scaler = None
    numeric_cols = ["temp", "pressure", "vibration"]
    before_stats = df[numeric_cols].describe().T

    if method == "standard":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif method == "minmax":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif method == "none":
        pass
    else:
        sys.exit(f"Error: invalid scaling method '{method}'")

    after_stats = df[numeric_cols].describe().T

    if scaler is not None:
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    return df, before_stats, after_stats


def split_data(
    df: pd.DataFrame, split: float, seed: int, out_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets with stratification if possible."""
    if not 0 < split < 1:
        sys.exit("Error: split ratio must be between 0 and 1")

    os.makedirs(out_dir, exist_ok=True)

    labels = df["label"].copy()
    stratify_labels = labels.where(labels.isin(["normal", "abnormal"]))

    # If stratification possible
    if stratify_labels.notna().sum() >= 2 and stratify_labels.nunique() > 1:
        try:
            train_df, test_df = train_test_split(
                df, train_size=split, random_state=seed, stratify=stratify_labels
            )
        except ValueError as e:
            print(f"Warning: stratified split failed ({e}), using random split.")
            train_df, test_df = train_test_split(
                df, train_size=split, random_state=seed, stratify=None
            )
    else:
        print("Warning: not enough class diversity, using random split.")
        train_df, test_df = train_test_split(
            df, train_size=split, random_state=seed, stratify=None
        )

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    return train_df, test_df


def save_artifacts(df: pd.DataFrame, output_path: str):
    """Save the cleaned full dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def print_report(
    before_missing, after_missing, before_stats, after_stats, train_df, test_df
):
    """Print preprocessing report to console."""
    print("\n=== Preprocessing Report ===")
    print("\nMissing values before:\n", before_missing)
    print("\nMissing values after:\n", after_missing)
    if before_stats is not None and after_stats is not None:
        print("\nNumeric stats before scaling:\n", before_stats)
        print("\nNumeric stats after scaling:\n", after_stats)
    print("\nTrain size:", len(train_df), "Test size:", len(test_df))
    print("\nLabel distribution (train):\n", train_df["label"].value_counts())
    print("\nLabel distribution (test):\n", test_df["label"].value_counts())
    print("============================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Smart Factory dataset"
    )
    parser.add_argument("--input", type=str, required=True, help="Input dataset CSV path")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/smartfactory_preprocessed.csv",
        help="Path to save cleaned dataset CSV",
    )
    parser.add_argument(
        "--impute",
        type=str,
        default="ffill",
        choices=["mean", "median", "ffill", "bfill", "drop"],
        help="Imputation strategy for missing values",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="standard",
        choices=["none", "standard", "minmax"],
        help="Scaling method",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/test split ratio (0-1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print summary statistics before and after preprocessing",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out",
        help="Output directory for artifacts",
    )
    args = parser.parse_args()

    df = load_data(args.input)
    df, before_missing, after_missing = impute_missing(df, args.impute)
    df, before_stats, after_stats = scale_features(df, args.scale, args.out_dir)
    train_df, test_df = split_data(df, args.split, args.seed, args.out_dir)
    save_artifacts(df, args.output)

    if args.report:
        print_report(before_missing, after_missing, before_stats, after_stats, train_df, test_df)

    print(f"\nDone: cleaned data saved to {args.output}, train/test in {args.out_dir}\n")


if __name__ == "__main__":
    main()
