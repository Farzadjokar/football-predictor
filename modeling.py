from __future__ import annotations
import json
import logging
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from feature_engineering import PREMATCH_FEATURE_COLS
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("modeling")
warnings.filterwarnings("ignore", category=UserWarning)
PROCESSED = "out_csv"
OUT = "out_csv"
RANDOM_SEED = 42
OUTCOME_CATEGORIES = ["A", "D", "H"]
def ranked_probability_score(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    n_classes = proba.shape[1]
    cum_pred = np.cumsum(proba, axis=1)
    y_onehot = np.eye(n_classes)[y_true_idx]
    cum_true = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1) / (n_classes - 1)))
def multiclass_brier(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    n_classes = proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true_idx]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))
def expected_calibration_error(y_true_idx: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_true_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_true_idx)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else \
               (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)
def classification_metrics(y_true_idx: np.ndarray, proba: np.ndarray) -> dict:
    proba = np.clip(proba, 1e-12, 1 - 1e-12)
    proba = proba / proba.sum(axis=1, keepdims=True)
    acc = float((proba.argmax(axis=1) == y_true_idx).mean())
    return {
        "accuracy": acc,
        "rps": ranked_probability_score(y_true_idx, proba),
        "log_loss": float(log_loss(y_true_idx, proba, labels=list(range(proba.shape[1])))),
        "brier": multiclass_brier(y_true_idx, proba),
        "ece": expected_calibration_error(y_true_idx, proba),
    }
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    corr = float(pearsonr(y_true, y_pred)[0]) if len(set(y_true)) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "pearson_r": corr}
def classifier_suite() -> dict[str, object]:
    return {
        "Dummy": DummyClassifier(strategy="prior", random_state=RANDOM_SEED),
        "KernelSVM": make_pipeline(StandardScaler(),
                                    SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_SEED,
                                  verbosity=0, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1),
    }
def regressor_suite() -> dict[str, object]:
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "KernelSVM": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        "KernelRidge": make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "XGBoost": XGBRegressor(random_state=RANDOM_SEED, verbosity=0, n_jobs=-1),
        "LightGBM": LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1),
    }
INPLAY_CLASSIFIER_NAMES = ["Dummy", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
INPLAY_REGRESSOR_NAMES = ["Dummy", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
def fit_and_time(model, X_train, y_train):
    t0 = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - t0
def run_classification_task(task_name: str, model_names: list[str],
                              X: dict, y_idx: dict, results: list) -> dict:
    suite = classifier_suite()
    fitted = {}
    for name in model_names:
        base_model = suite[name]
        base_model, fit_time = fit_and_time(base_model, X["train"], y_idx["train"])
        raw_proba_test = base_model.predict_proba(X["test"])
        raw_metrics = classification_metrics(y_idx["test"], raw_proba_test)
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        calibrated.fit(X["train"], y_idx["train"])
        cal_proba_test = calibrated.predict_proba(X["test"])
        cal_metrics = classification_metrics(y_idx["test"], cal_proba_test)
        row = {"task": task_name, "model": name, "fit_time_sec": round(fit_time, 3),
               "n_train": len(y_idx["train"]), "n_test": len(y_idx["test"])}
        row.update({f"raw_{k}": v for k, v in raw_metrics.items()})
        row.update({f"calibrated_{k}": v for k, v in cal_metrics.items()})
        results.append(row)
        fitted[name] = calibrated
        log.info("[%s] %-16s fit=%.2fs  raw ECE=%.3f -> calibrated ECE=%.3f  "
                  "raw RPS=%.3f -> calibrated RPS=%.3f",
                  task_name, name, fit_time, raw_metrics["ece"], cal_metrics["ece"],
                  raw_metrics["rps"], cal_metrics["rps"])
    return fitted
def run_regression_task(task_name: str, model_names: list[str],
                          X: dict, y: dict, results: list) -> dict:
    suite = regressor_suite()
    fitted = {}
    for name in model_names:
        model = suite[name]
        model, fit_time = fit_and_time(model, X["train"], y["train"])
        pred_test = model.predict(X["test"])
        metrics = regression_metrics(y["test"], pred_test)
        row = {"task": task_name, "model": name, "fit_time_sec": round(fit_time, 3),
               "n_train": len(y["train"]), "n_test": len(y["test"])}
        row.update(metrics)
        results.append(row)
        fitted[name] = model
        log.info("[%s] %-16s fit=%.2fs  MAE=%.3f  RMSE=%.3f  r=%.3f",
                  task_name, name, fit_time, metrics["mae"], metrics["rmse"], metrics["pearson_r"])
    return fitted
def assemble_prematch_data(features: pd.DataFrame):
    outcome_idx = features["outcome"].map({c: i for i, c in enumerate(OUTCOME_CATEGORIES)})
    X, y_idx, y_margin = {}, {}, {}
    for split in ["train", "val", "test"]:
        mask = features["split"] == split
        X[split] = features.loc[mask, PREMATCH_FEATURE_COLS].to_numpy()
        y_idx[split] = outcome_idx[mask].to_numpy()
        y_margin[split] = features.loc[mask, "margin"].to_numpy()
    return X, y_idx, y_margin
INPLAY_FEATURE_COLS = PREMATCH_FEATURE_COLS + [
    "snapshot_minute", "current_score_diff", "man_advantage",
    "events_so_far", "events_last_10min",
]
def assemble_inplay_data(snapshot_features: pd.DataFrame):
    outcome_idx = snapshot_features["outcome"].map({c: i for i, c in enumerate(OUTCOME_CATEGORIES)})
    X, y_idx, y_margin = {}, {}, {}
    for split in ["train", "val", "test"]:
        mask = snapshot_features["split"] == split
        X[split] = snapshot_features.loc[mask, INPLAY_FEATURE_COLS].to_numpy()
        y_idx[split] = outcome_idx[mask].to_numpy()
        y_margin[split] = snapshot_features.loc[mask, "margin"].to_numpy()
    return X, y_idx, y_margin
def inplay_metric_vs_minute(snapshot_features: pd.DataFrame, X_inplay: dict, y_idx: dict,
                              y_margin: dict, inplay_clf, inplay_reg,
                              frozen_clf, frozen_reg, out_dir: Path) -> pd.DataFrame:
    test_mask = snapshot_features["split"] == "test"
    if not test_mask.any():
        log.warning("in-play metric-vs-minute: TEST split is empty for Task L, skipping "
                     "(need snapshot data whose parent match falls in the chronological test "
                     "window -- see the run() docstring note about this demo's data coverage)")
        return pd.DataFrame()
    test_df = snapshot_features.loc[test_mask].reset_index(drop=True)
    X_test = X_inplay["test"]
    X_test_prematch_only = test_df[PREMATCH_FEATURE_COLS].to_numpy()
    inplay_proba = inplay_clf.predict_proba(X_test)
    frozen_proba = frozen_clf.predict_proba(X_test_prematch_only)
    inplay_margin_pred = inplay_reg.predict(X_test)
    frozen_margin_pred = frozen_reg.predict(X_test_prematch_only)
    rows = []
    for minute, idx in test_df.groupby("snapshot_minute").groups.items():
        idx = np.array(idx)
        y_i = y_idx["test"][idx]
        rows.append({
            "snapshot_minute": minute,
            "n_snapshots": len(idx),
            "model3_rps": ranked_probability_score(y_i, inplay_proba[idx]),
            "frozen_prematch_rps": ranked_probability_score(y_i, frozen_proba[idx]),
            "model3_margin_mae": mean_absolute_error(y_margin["test"][idx], inplay_margin_pred[idx]),
            "frozen_prematch_margin_mae": mean_absolute_error(y_margin["test"][idx],
                                                                frozen_margin_pred[idx]),
        })
    curve = pd.DataFrame(rows).sort_values("snapshot_minute")
    curve.to_csv(out_dir / "inplay_metric_vs_minute.csv", index=False, encoding="utf-8")
    log.info("wrote inplay_metric_vs_minute.csv (%d minute buckets) -> %s", len(curve), out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(curve["snapshot_minute"], curve["model3_rps"], marker="o", label="Model 3 (in-play)")
    axes[0].plot(curve["snapshot_minute"], curve["frozen_prematch_rps"], marker="o", linestyle="--",
                 label="Frozen pre-match prior")
    axes[0].set_xlabel("Match minute")
    axes[0].set_ylabel("RPS (lower is better)")
    axes[0].set_title("Outcome forecast: RPS vs. match minute")
    axes[0].legend()
    axes[1].plot(curve["snapshot_minute"], curve["model3_margin_mae"], marker="o", label="Model 3 (in-play)")
    axes[1].plot(curve["snapshot_minute"], curve["frozen_prematch_margin_mae"], marker="o",
                 linestyle="--", label="Frozen pre-match prior")
    axes[1].set_xlabel("Match minute")
    axes[1].set_ylabel("Margin MAE (lower is better)")
    axes[1].set_title("Goal margin forecast: MAE vs. match minute")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "inplay_metric_vs_minute.png", dpi=150)
    plt.close(fig)
    log.info("wrote inplay_metric_vs_minute.png -> %s", out_dir)
    return curve
def run(processed_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    features = pd.read_csv(processed_dir / "matches_features.csv",
                            parse_dates=["match_date"], encoding="utf-8")
    for split in ["train", "val", "test"]:
        n = (features["split"] == split).sum()
        if n == 0:
            log.warning("Task C/R: split '%s' has 0 matches -- that split's metrics will be "
                         "undefined/empty below", split)
    X_pm, y_idx_pm, y_margin_pm = assemble_prematch_data(features)
    log.info("=== Task C: pre-match outcome classification ===")
    fitted_clf = run_classification_task("C", list(classifier_suite().keys()),
                                           X_pm, y_idx_pm, results)
    log.info("=== Task R: pre-match margin regression ===")
    fitted_reg = run_regression_task("R", list(regressor_suite().keys()),
                                       X_pm, y_margin_pm, results)
    snap = pd.read_csv(processed_dir / "snapshot_index_split.csv",
                        parse_dates=["match_date"], encoding="utf-8")
    snap_features = snap.merge(
        features[["match_id"] + PREMATCH_FEATURE_COLS], on="match_id", how="inner",
        suffixes=("", "_prematch"),
    )
    if len(snap_features) != len(snap):
        log.warning("Task L: %d/%d snapshot rows dropped on merge with matches_features.csv "
                     "(their match_id wasn't found there -- should not happen if Stage 5A ran "
                     "on the same matches_split.csv)", len(snap) - len(snap_features), len(snap))
    split_counts = snap_features["split"].value_counts().to_dict()
    log.info("Task L snapshot rows by split: %s", split_counts)
    demo_fallback = split_counts.get("train", 0) == 0 or split_counts.get("test", 0) == 0
    if demo_fallback:
        log.warning("Task L: chronological train or test split is empty in THIS run (limited "
                     "events.csv coverage, not a code bug) -- running a SANITY-ONLY random "
                     "70/30 split to prove the training code works, written separately to "
                     "inplay_SANITY_ONLY_results.csv. Re-run with fuller events coverage for "
                     "the real chronological Task L results.")
        rng = np.random.RandomState(RANDOM_SEED)
        shuffled_idx = rng.permutation(len(snap_features))
        cut = int(0.7 * len(shuffled_idx))
        snap_features = snap_features.copy()
        snap_features["split"] = "val"
        snap_features.iloc[shuffled_idx[:cut], snap_features.columns.get_loc("split")] = "train"
        snap_features.iloc[shuffled_idx[cut:], snap_features.columns.get_loc("split")] = "test"
    X_ip, y_idx_ip, y_margin_ip = assemble_inplay_data(snap_features)
    sanity_results: list = []
    target_results = sanity_results if demo_fallback else results
    log.info("=== Task L: in-play classification (score/outcome state) ===")
    fitted_clf_L = run_classification_task("L", INPLAY_CLASSIFIER_NAMES,
                                             X_ip, y_idx_ip, target_results)
    log.info("=== Task L: in-play margin regression ===")
    fitted_reg_L = run_regression_task("L", INPLAY_REGRESSOR_NAMES,
                                         X_ip, y_margin_ip, target_results)
    if demo_fallback:
        pd.DataFrame(sanity_results).to_csv(out_dir / "inplay_SANITY_ONLY_results.csv",
                                              index=False, encoding="utf-8")
        log.info("wrote inplay_SANITY_ONLY_results.csv (%d rows, NOT real chronological "
                  "results) -> %s", len(sanity_results), out_dir)
    else:
        inplay_metric_vs_minute(snap_features, X_ip, y_idx_ip, y_margin_ip,
                                  fitted_clf_L["RandomForest"], fitted_reg_L["RandomForest"],
                                  fitted_clf["RandomForest"], fitted_reg["RandomForest"],
                                  out_dir)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "model_results.csv", index=False, encoding="utf-8")
    log.info("wrote model_results.csv (%d rows -- Task C + Task R%s) -> %s", len(results_df),
              "" if demo_fallback else " + Task L", out_dir)
    log.info("=== modeling.py: ALL STAGES COMPLETE ===")
run(Path(PROCESSED), Path(OUT))