from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO,format="%(levelname)s:%(message)s");
log=logging.getLogger("temporal_split");
PROCESSED="out_csv";
OUT="out_csv";

# ---------------------------------------------------------------------------
# Stage 4 deliverable: temporal (chronological) train/val/test splitting.
# Input : matches_clean.csv, snapshot_index_clean.csv (from preprocess_clean.py)
# Output: split assignment columns added back onto both tables, plus a
#         standalone split manifest for auditability at Mid Defence 1.
#
# Non-negotiable rules enforced here (Section 3 of the brief):
#   - Split is chronological BY MATCH (never a random shuffle) so that no
#     model is ever trained on the future to predict the past.
#   - Every snapshot row of a given match_id lives entirely on one side of
#     the split -- Task L's within-match correlation would otherwise make
#     the test score fiction (Section 3.1).
#   - The split boundary is a DATE, not a row count, so it is reusable
#     identically across Task C, Task R, and Task L.
# ---------------------------------------------------------------------------

TRAIN_FRAC=0.70;
VAL_FRAC=0.15;
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC

def compute_split_dates(matches:pd.DataFrame,
                          train_frac:float=TRAIN_FRAC,
                          val_frac:float=VAL_FRAC)->dict[str,pd.Timestamp]:
    """Choose the train/val and val/test boundary dates from the quantiles of
    match_date, NOT from a fixed calendar date -- this keeps the split
    proportionate regardless of which competitions/seasons were selected."""
    dates=matches["match_date"].sort_values().reset_index(drop=True);
    n=len(dates);
    train_end_idx=int(np.floor(n*train_frac))-1;
    val_end_idx=int(np.floor(n*(train_frac+val_frac)))-1;
    train_end_idx=max(0,min(train_end_idx,n-1));
    val_end_idx=max(train_end_idx,min(val_end_idx,n-1));
    boundaries={
        "train_end_date": dates.iloc[train_end_idx],
        "val_end_date": dates.iloc[val_end_idx],
    };
    log.info("split boundaries chosen from match_date quantiles: train_end=%s val_end=%s "
              "(n_matches=%d, train_frac=%.2f, val_frac=%.2f)",
              boundaries["train_end_date"],boundaries["val_end_date"],n,train_frac,val_frac);
    return boundaries;

def assign_match_splits(matches:pd.DataFrame,
                          train_end_date:pd.Timestamp,
                          val_end_date:pd.Timestamp)->pd.DataFrame:
    """Assign each match to train/val/test purely from match_date. This single
    per-match_id assignment is what both the pre-match tasks (C, R) and the
    in-play task (L) inherit from -- one source of truth, so a match can
    never end up split differently across tasks."""
    df=matches.copy();
    conditions=[
        df["match_date"]<=train_end_date,
        (df["match_date"]>train_end_date)&(df["match_date"]<=val_end_date),
        df["match_date"]>val_end_date,
    ];
    choices=["train","val","test"];
    df["split"]=np.select(conditions,choices,default="unassigned");

    n_unassigned=int((df["split"]=="unassigned").sum());
    if n_unassigned:
        raise ValueError(f"{n_unassigned} matches failed split assignment -- "
                          f"check for NaT match_date values upstream");

    counts=df["split"].value_counts().to_dict();
    log.info("match-level split assignment: %s",counts);

    overlap_check=df.groupby("split")["match_date"].agg(["min","max"]);
    log.info("date ranges per split:\n%s",overlap_check.to_string());
    train_max=df.loc[df["split"]=="train","match_date"].max();
    val_min=df.loc[df["split"]=="val","match_date"].min();
    val_max=df.loc[df["split"]=="val","match_date"].max();
    test_min=df.loc[df["split"]=="test","match_date"].min();
    if pd.notna(train_max) and pd.notna(val_min) and train_max>val_min:
        raise ValueError("chronology violated: a train match is dated after a val match");
    if pd.notna(val_max) and pd.notna(test_min) and val_max>test_min:
        raise ValueError("chronology violated: a val match is dated after a test match");
    log.info("chronology check passed: train_end(%s) <= val_start(%s), "
              "val_end(%s) <= test_start(%s)",train_max,val_min,val_max,test_min);
    return df;

def propagate_split_to_snapshots(snapshot_index:pd.DataFrame,
                                   match_splits:pd.DataFrame)->pd.DataFrame:
    """Every snapshot row inherits its parent match's split label via a join
    on match_id -- snapshots are NEVER independently assigned, which is the
    mechanism that guarantees Task L respects match-level integrity."""
    lookup=match_splits.set_index("match_id")["split"];
    df=snapshot_index.copy();
    df["split"]=df["match_id"].map(lookup);

    n_missing=int(df["split"].isna().sum());
    if n_missing:
        log.warning("DROPPED %d snapshot rows whose match_id has no split assignment "
                    "(orphaned after match cleaning)",n_missing);
        df=df.dropna(subset=["split"]);

    per_match_split_counts=df.groupby("match_id")["split"].nunique();
    violating=per_match_split_counts[per_match_split_counts>1];
    if len(violating):
        raise ValueError(
            f"LEAKAGE: {len(violating)} match_id(s) have snapshots spread across "
            f"more than one split -- match-level split integrity violated: "
            f"{violating.index.tolist()[:10]}"
        );
    log.info("propagate_split_to_snapshots: verified every match_id's snapshots sit "
              "entirely on one split side (0 violations)");

    counts=df["split"].value_counts().to_dict();
    log.info("snapshot-level split distribution: %s",counts);
    return df;

def check_class_balance_across_splits(matches:pd.DataFrame)->None:
    """Report (do not rebalance) the outcome distribution per split -- a
    required honesty check, since chronological splitting can shift the
    class mix (e.g. a competition realignment or era effect), which is a
    real form of distribution shift the report must acknowledge rather
    than silently correct."""
    if "outcome" not in matches.columns:
        log.warning("check_class_balance_across_splits: no 'outcome' column found, skipping");
        return;
    table=(matches.groupby("split")["outcome"]
           .value_counts(normalize=True)
           .round(3)
           .unstack(fill_value=0.0));
    log.info("outcome distribution by split (rows sum to 1.0):\n%s",table.to_string());

def write_split_manifest(out_dir:Path,
                           boundaries:dict[str,pd.Timestamp],
                           match_splits:pd.DataFrame,
                           snapshot_splits:pd.DataFrame)->None:
    manifest={
        "train_end_date": str(boundaries["train_end_date"].date()),
        "val_end_date": str(boundaries["val_end_date"].date()),
        "n_matches": {
            k: int(v) for k,v in match_splits["split"].value_counts().to_dict().items()
        },
        "n_snapshots": {
            k: int(v) for k,v in snapshot_splits["split"].value_counts().to_dict().items()
        },
        "rule": (
            "Split is chronological by match_date; every snapshot inherits its "
            "parent match's split. No match_id appears on more than one side."
        ),
    };
    (out_dir/"split_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");
    log.info("wrote split_manifest.json->%s",out_dir);

def run(processed_dir:Path,out_dir:Path)->None:
    out_dir.mkdir(parents=True,exist_ok=True);

    matches=pd.read_csv(processed_dir/"matches_clean.csv",parse_dates=["match_date"],
                         encoding="utf-8");
    snapshot_index=pd.read_csv(processed_dir/"snapshot_index_clean.csv",
                                 parse_dates=["match_date"],encoding="utf-8");

    boundaries=compute_split_dates(matches);
    match_splits=assign_match_splits(matches,boundaries["train_end_date"],
                                       boundaries["val_end_date"]);
    check_class_balance_across_splits(match_splits);

    snapshot_splits=propagate_split_to_snapshots(snapshot_index,match_splits);

    match_splits.to_csv(out_dir/"matches_split.csv",index=False,encoding="utf-8");
    log.info("wrote matches_split.csv (%d rows)->%s",len(match_splits),out_dir);

    snapshot_splits.to_csv(out_dir/"snapshot_index_split.csv",index=False,encoding="utf-8");
    log.info("wrote snapshot_index_split.csv (%d rows)->%s",len(snapshot_splits),out_dir);

    write_split_manifest(out_dir,boundaries,match_splits,snapshot_splits);

run(Path(PROCESSED),Path(OUT));
