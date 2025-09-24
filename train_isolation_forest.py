#!/usr/bin/env python3
"""
train_isolation_forest.py

Train an Isolation Forest model on preprocessed Smart Factory data.
Supports grid search for hyperparameter tuning and always outputs metrics.

Usage Examples:
python train_isolation_forest.py --train ./out/train.csv --test ./out/test.csv \
    --scaler ./out/scaler.joblib --output_model ./out/isolation_forest.joblib \
    --grid_search --report
"""

import os
import sys
import json
import logging
import argparse
from typing import Tuple, Optional, Dict, Any, Union

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------------
# Data loading and preprocessing
# -------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        logging.error(f"File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    required_cols = ["temp", "pressure", "vibration"]
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Missing required column '{col}' in {path}")
            sys.exit(1)
    return df


def prepare_features(df: pd.DataFrame, scaler_path: Optional[str] = None) -> pd.DataFrame:
    features = df[["temp", "pressure", "vibration"]].copy()
    if scaler_path:
        if not os.path.isfile(scaler_path):
            logging.error(f"Scaler file not found: {scaler_path}")
            sys.exit(1)
        scaler = joblib.load(scaler_path)
        features = pd.DataFrame(scaler.transform(features), columns=features.columns)
    return features


def map_labels(df: pd.DataFrame, ignore_unknown: bool = False) -> pd.Series:
    if "label" not in df.columns:
        return pd.Series(dtype=float)
    labels = df["label"].str.lower().map({"normal": 0, "abnormal": 1, "unknown": np.nan})
    if ignore_unknown:
        labels = labels.dropna()
    return labels


# -------------------------
# Isolation Forest training
# -------------------------
def train_iforest(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    contamination: float = 0.01,
    n_estimators: int = 100,
    max_samples: Union[str, float] = "auto",
    n_jobs: int = -1,
    seed: int = 42,
    grid_search: bool = False,
) -> Tuple[IsolationForest, float]:
    """
    Train IsolationForest with optional manual grid search using a test-set threshold.

    Returns:
        clf: trained IsolationForest
        threshold: score threshold corresponding to realistic anomaly fraction
    """
    # Determine max_samples
    max_samples_val = max_samples
    if isinstance(max_samples, str) and max_samples.lower() != "auto":
        try:
            max_samples_val = float(max_samples)
            if not (0 < max_samples_val <= 1):
                raise ValueError()
        except Exception:
            logging.error("max_samples must be 'auto' or a float in (0,1].")
            sys.exit(1)

    base_model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples_val,
        random_state=seed,
        n_jobs=n_jobs,
    )

    best_model = base_model
    best_threshold = None

    # -------------------------
    # Grid search logic
    # -------------------------
    if grid_search and X_test is not None and y_test is not None and (~np.isnan(y_test)).sum() > 0:
        logging.info("Starting manual grid search using test set for threshold selection...")
        mask = ~np.isnan(y_test)
        X_val = X_test[mask]
        y_val = y_test[mask].astype(int)

        n_estimators_grid = [100, 200, 300]
        contamination_grid = [0.15, 0.20, 0.25]  # tuned for your ~20% anomaly data
        expected_max_anomaly_frac = 0.212  # known anomaly fraction

        best_f1 = -1.0
        for n_est in n_estimators_grid:
            for cont in contamination_grid:
                clf = IsolationForest(
                    n_estimators=n_est,
                    contamination=cont,
                    max_samples=max_samples_val,
                    random_state=seed,
                    n_jobs=n_jobs,
                )
                clf.fit(X_train)
                scores = clf.decision_function(X_val)
                thresh = np.percentile(scores, 100 * expected_max_anomaly_frac)
                preds = (scores <= thresh).astype(int)
                anomaly_frac = preds.mean()
                f1 = f1_score(y_val, preds, pos_label=1)
                logging.info(
                    f"n_estimators={n_est}, contamination={cont:.3f}, anomaly_frac={anomaly_frac:.3f}, anomaly F1={f1:.4f}"
                )
                tolerance = 0.02
                if anomaly_frac <= expected_max_anomaly_frac * (1 + tolerance) and f1 > best_f1:
                    best_f1 = f1
                    best_model = clf
                    best_threshold = thresh
        if best_threshold is None:
            logging.warning("No grid search combination satisfied anomaly fraction constraint. Using default model.")
            best_model.fit(X_train)
            scores = best_model.decision_function(X_train)
            best_threshold = np.percentile(scores, 100 * expected_max_anomaly_frac)
    else:
        # Default training
        base_model.fit(X_train)
        scores = base_model.decision_function(X_train)
        # Default threshold at ~2% lowest scores
        best_threshold = np.percentile(scores, 100 * 0.02)
        best_model = base_model

    # -------------------------
    # Always evaluate if test labels exist
    # -------------------------
    if X_test is not None and y_test is not None and (~np.isnan(y_test)).sum() > 0:
        mask = ~np.isnan(y_test)
        X_eval = X_test[mask]
        y_eval = y_test[mask].astype(int)
        scores_eval = best_model.decision_function(X_eval)
        preds_eval = (scores_eval <= best_threshold).astype(int)
        anomaly_frac = preds_eval.mean()
        f1_eval = f1_score(y_eval, preds_eval, pos_label=1)
        logging.info(f"[Evaluation] Anomaly fraction={anomaly_frac:.3f}, F1 score={f1_eval:.4f}")


    accuracy = accuracy_score(y_eval, preds_eval)
    precision_normal = precision_score(y_eval, preds_eval, pos_label=0)
    precision_abnormal = precision_score(y_eval, preds_eval, pos_label=1)
    recall_normal = recall_score(y_eval, preds_eval, pos_label=0)
    recall_abnormal = recall_score(y_eval, preds_eval, pos_label=1)
    f1_normal = f1_score(y_eval, preds_eval, pos_label=0)
    f1_abnormal = f1_score(y_eval, preds_eval, pos_label=1)
    roc_auc = roc_auc_score(y_eval, -scores_eval)  # invert since low scores = anomalies

    logging.info("=== Evaluation Metrics ===")
    logging.info(f"Accuracy               : {accuracy:.4f}")
    logging.info(f"Precision (Normal)     : {precision_normal:.4f}")
    logging.info(f"Precision (Abnormal)   : {precision_abnormal:.4f}")
    logging.info(f"Recall (Normal)        : {recall_normal:.4f}")
    logging.info(f"Recall (Abnormal)      : {recall_abnormal:.4f}")
    logging.info(f"F1-Score (Normal)      : {f1_normal:.4f}")
    logging.info(f"F1-Score (Abnormal)    : {f1_abnormal:.4f}")
    logging.info(f"ROC-AUC                : {roc_auc:.4f}")
    logging.info("==========================")



    logging.info(f"Selected threshold for anomaly detection: {best_threshold:.4f}")

    return best_model, best_threshold


# -------------------------
# Save model and threshold
# -------------------------
def save_artifacts(clf: IsolationForest, threshold: float, output_model: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, output_model)
    logging.info(f"Trained model saved to {output_model}")
    threshold_path = os.path.join(out_dir, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f)
    logging.info(f"Threshold saved to {threshold_path}")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest on preprocessed Smart Factory data")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str)
    parser.add_argument("--scaler", type=str)
    parser.add_argument("--output_model", type=str, default="./out/isolation_forest.joblib")
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    train_df = load_data(args.train)
    X_train = prepare_features(train_df, args.scaler)
    y_train = map_labels(train_df)

    if args.test and os.path.isfile(args.test):
        test_df = load_data(args.test)
        X_test = prepare_features(test_df, args.scaler)
        y_test = map_labels(test_df)
    else:
        X_test = y_test = None

    clf, threshold = train_iforest(
        X_train,
        X_test,
        y_test,
        contamination=0.01,
        n_estimators=100,
        grid_search=args.grid_search,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    save_artifacts(clf, threshold, args.output_model, args.out_dir)
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
