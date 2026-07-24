from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("feature_engineering")
PROCESSED = "out_csv"
OUT = "out_csv"
ROLL_WINDOW = 5  # "recent form" window, per the brief's own wording
def _team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) instead of one row per match -- the shape
    rolling per-team aggregates need. Every match contributes two rows."""
    home = matches[["match_id", "match_date", "home_team_id", "away_team_id",
                     "home_score", "away_score"]].rename(columns={
        "home_team_id": "team_id", "away_team_id": "opponent_id",
        "home_score": "goals_for", "away_score": "goals_against",
    })
    home["is_home"] = 1
    away = matches[["match_id", "match_date", "away_team_id", "home_team_id",
                     "home_score", "away_score"]].rename(columns={
        "away_team_id": "team_id", "home_team_id": "opponent_id",
        "away_score": "goals_for", "home_score": "goals_against",
    })
    away["is_home"] = 0
    log_df = pd.concat([home, away], ignore_index=True)
    log_df["points"] = np.select(
        [log_df["goals_for"] > log_df["goals_against"],
         log_df["goals_for"] == log_df["goals_against"]],
        [3, 1], default=0,
    )
    return log_df.sort_values(["team_id", "match_date"]).reset_index(drop=True)
def _rolling_prior_form(team_log: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """For every (team, match) row, compute rolling-window averages using
    ONLY that team's matches strictly before this one. shift(1) is what
    excludes the current match's own result from its own feature -- remove
    it and this becomes leakage."""
    g = team_log.groupby("team_id", group_keys=False)
    def _prior_stats(group: pd.DataFrame) -> pd.DataFrame:
        shifted = group.shift(1)
        group["form_goals_for"] = shifted["goals_for"].rolling(window, min_periods=1).mean()
        group["form_goals_against"] = shifted["goals_against"].rolling(window, min_periods=1).mean()
        group["form_points"] = shifted["points"].rolling(window, min_periods=1).mean()
        group["matches_played_prior"] = np.arange(len(group))
        group["rest_days"] = (group["match_date"] - shifted["match_date"]).dt.days
        return group
    return g.apply(_prior_stats)
def build_prematch_features(matches: pd.DataFrame) -> pd.DataFrame:
    n0 = len(matches)
    team_log = _team_match_log(matches)
    team_log = _rolling_prior_form(team_log)
    feat_cols = ["form_goals_for", "form_goals_against", "form_points",
                 "matches_played_prior", "rest_days"]
    home_feats = (team_log[team_log["is_home"] == 1]
                  .set_index("match_id")[feat_cols]
                  .add_prefix("home_"))
    away_feats = (team_log[team_log["is_home"] == 0]
                  .set_index("match_id")[feat_cols]
                  .add_prefix("away_"))
    out = matches.set_index("match_id").join(home_feats).join(away_feats).reset_index()
    out["form_goals_for_diff"] = out["home_form_goals_for"] - out["away_form_goals_for"]
    out["form_goals_against_diff"] = out["home_form_goals_against"] - out["away_form_goals_against"]
    out["form_points_diff"] = out["home_form_points"] - out["away_form_points"]
    out["rest_days_diff"] = out["home_rest_days"] - out["away_rest_days"]
    out["min_matches_played_prior"] = out[["home_matches_played_prior",
                                            "away_matches_played_prior"]].min(axis=1)
    out["is_cold_start"] = out["min_matches_played_prior"] == 0
    n_cold = int(out["is_cold_start"].sum())
    if n_cold:
        log.warning("%d/%d matches have at least one team with ZERO prior matches in this "
                     "dataset (cold start -- rolling-form columns are NaN for that side, "
                     "imputed downstream from TRAIN-split medians only)", n_cold, n0)
    if len(out) != n0:
        raise ValueError("build_prematch_features must never change row count")
    log.info("build_prematch_features: %d matches -> %d feature columns added", n0,
              len([c for c in out.columns if c.startswith(("home_", "away_", "form_", "rest_",
                                                             "min_matches", "is_cold"))]))
    return out
def impute_from_train(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = features.copy()
    train_medians = df.loc[df["split"] == "train", feature_cols].median()
    n_filled = int(df[feature_cols].isna().sum().sum())
    df[feature_cols] = df[feature_cols].fillna(train_medians)
    if n_filled:
        log.info("imputed %d missing feature values using %d TRAIN-split medians", n_filled,
                  len(feature_cols))
    return df
PREMATCH_FEATURE_COLS = [
    "home_form_goals_for", "home_form_goals_against", "home_form_points",
    "home_matches_played_prior", "home_rest_days",
    "away_form_goals_for", "away_form_goals_against", "away_form_points",
    "away_matches_played_prior", "away_rest_days",
    "form_goals_for_diff", "form_goals_against_diff", "form_points_diff",
    "rest_days_diff", "min_matches_played_prior",
]
def run(processed_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(processed_dir / "matches_split.csv", parse_dates=["match_date"],
                           encoding="utf-8")
    features = build_prematch_features(matches)
    features = impute_from_train(features, PREMATCH_FEATURE_COLS)
    features.to_csv(out_dir / "matches_features.csv", index=False, encoding="utf-8")
    log.info("wrote matches_features.csv (%d rows, %d cols) -> %s", len(features),
              features.shape[1], out_dir)
run(Path(PROCESSED), Path(OUT))