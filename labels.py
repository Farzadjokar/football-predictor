from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO,format="%(levelname)s:%(message)s");
log=logging.getLogger("label_and_define");
PROCESSED="out_csv";
OUT="out_csv";
SNAPSHOT_INTERVAL_MIN=5;
REGULATION_END_MIN=90;
MARGIN_CLIP=5;
def label_matches(matches: pd.DataFrame)->pd.DataFrame:
    df=matches.copy();
    diff=df["home_score"]-df["away_score"]
    df["outcome"]=np.select(
        [diff>0,diff == 0,diff<0],
        ["H","D","A"],
        default="?",
    ).astype(str);
    df["margin"]=diff.clip(-MARGIN_CLIP,MARGIN_CLIP);
    df["margin_clipped"]=diff.ne(df["margin"]);
    n_clipped=int(df["margin_clipped"].sum());
    if n_clipped:
        log.warning("%d/%d matches had |margin| > %d and were clipped for Task R",
                    n_clipped,len(df),MARGIN_CLIP);
    counts=df["outcome"].value_counts(normalize=True).round(3).to_dict();
    log.info("outcome label distribution:%s",counts);
    return df;
def build_snapshot_index(labeled_matches:pd.DataFrame,
                          interval:int=SNAPSHOT_INTERVAL_MIN,
                          end_min:int=REGULATION_END_MIN)->pd.DataFrame:
    rows=[];
    for _, m in labeled_matches.iterrows():
        for minute in range(interval,end_min+1,interval):
            rows.append({
                "match_id": m["match_id"],
                "snapshot_minute": minute,
                "outcome": m["outcome"],
                "margin": m["margin"],
                "competition_id": m["competition_id"],
                "season_id": m["season_id"],
                "match_date": m["match_date"],
            });
    snap=pd.DataFrame(rows);
    log.info("snapshot index->%d snapshot rows from %d matches (%d snapshots/match)",
              len(snap),labeled_matches["match_id"].nunique(),end_min//interval);
    return snap;
PROBLEM_DEFINITION = {
    "task_C": {
        "name": "Model 1 -- Pre-Match Outcome Classification",
        "one_example_is": "one row per match_id, indexed only by match_id",
        "input_x": (
            "a feature vector built EXCLUSIVELY from information available strictly "
            "before this match's kick_off timestamp: aggregates of each team's prior "
            "matches (rolling form, xG-proxy shot rates, possession share, etc.), "
            "venue, rest days, head-to-head history. No event from this match may "
            "contribute to the vector."
        ),
        "label_y": "outcome in {H, D, A}, derived from this match's home_score/away_score",
        "granularity": "1 row = 1 match",
        "leakage_rule": "only matches with match_date < this match's match_date may feed the aggregates",
    },
    "task_R": {
        "name": "Model 2 -- Pre-Match Goal-Margin Regression",
        "one_example_is": "one row per match_id (same rows/features as Task C)",
        "input_x": "identical pre-match feature vector as Task C",
        "label_y": "margin = clip(home_score - away_score, -5, +5)",
        "granularity": "1 row = 1 match",
        "leakage_rule": "same as Task C",
    },
    "task_L": {
        "name": "Model 3 -- In-Play (Live) Prediction",
        "one_example_is": "one row per (match_id, snapshot_minute) pair",
        "input_x": (
            "the match's pre-match feature vector (as in Task C/R) CONCATENATED with "
            "in-play features computed only from this match's own events with "
            "timestamp <= snapshot_minute: current score, man-advantage (red cards), "
            "event counts/rates in a trailing window, momentum indicators."
        ),
        "label_y": "outcome AND margin of THIS match (identical value at every snapshot_minute of the match)",
        "granularity": f"1 match = many rows, one every {SNAPSHOT_INTERVAL_MIN} minutes from "
                        f"{SNAPSHOT_INTERVAL_MIN} to {REGULATION_END_MIN}",
        "leakage_rule":(
            "for the row at snapshot_minute=t, zero events with timestamp > t may "
            "contribute to any feature (STRICT time-t cut). All snapshot rows of one "
            "match_id must be placed entirely on one side of the train/val/test split "
            "-- never split within a match."
        ),
    },
};
def write_problem_definition(out_dir:Path)->None:
    (out_dir/"problem_definition.json").write_text(json.dumps(PROBLEM_DEFINITION,indent=2),encoding="utf-8");
    md=["#Problem Definition (Stage 2 deliverable)\n"];
    for task_id, spec in PROBLEM_DEFINITION.items():
        md.append(f"##{task_id}:{spec['name']}\n");
        md.append(f"-**One example is:**{spec['one_example_is']}");
        md.append(f"-**Input (X):**{spec['input_x']}");
        md.append(f"-**Label (y):**{spec['label_y']}");
        md.append(f"-**Granularity:**{spec['granularity']}");
        md.append(f"-**Leakage rule:**{spec['leakage_rule']}\n");
    (out_dir/"problem_definition.md").write_text("\n".join(md),encoding="utf-8");
    log.info("wrote problem_definition.json/.md->%s",out_dir);
def run(processed_dir:Path,out_dir:Path)->None:
    out_dir.mkdir(parents=True,exist_ok=True);
    matches = pd.read_csv(processed_dir/"matches.csv",parse_dates=["match_date"],encoding="utf-8");
    labeled=label_matches(matches);
    labeled.to_csv(out_dir/"labeled_matches.csv",index=False,encoding="utf-8");
    log.info("wrote labeled_matches.csv (%d rows)->%s",len(labeled),out_dir);
    snap_index=build_snapshot_index(labeled);
    snap_index.to_csv(out_dir/"snapshot_index.csv",index=False,encoding="utf-8");
    log.info("wrote snapshot_index.csv(%d rows)->%s",len(snap_index),out_dir);
    write_problem_definition(out_dir);
run(Path(PROCESSED),Path(OUT));