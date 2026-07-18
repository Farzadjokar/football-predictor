from __future__ import annotations
import ast
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("label_and_define")
PROCESSED = "out_csv"
OUT = "out_csv"
SNAPSHOT_INTERVAL_MIN = 5
REGULATION_END_MIN = 90
MARGIN_CLIP = 5
RECENT_WINDOW_MIN = 10
def label_matches(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    diff = df["home_score"] - df["away_score"]
    df["outcome"] = np.select(
        [diff > 0, diff == 0, diff < 0],
        ["H", "D", "A"],
        default="?",
    ).astype(str)
    df["margin"] = diff.clip(-MARGIN_CLIP, MARGIN_CLIP)
    df["margin_clipped"] = diff.ne(df["margin"])
    n_clipped = int(df["margin_clipped"].sum())
    if n_clipped:
        log.warning("%d/%d matches had |margin| > %d and were clipped for Task R",
                    n_clipped, len(df), MARGIN_CLIP)

    counts = df["outcome"].value_counts(normalize=True).round(3).to_dict()
    log.info("outcome label distribution: %s", counts)
    return df
def _safe_literal_eval(val):
    if not isinstance(val, str) or not val.strip():
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None
def _extract_goal_events(events: pd.DataFrame) -> pd.DataFrame:
    frames = []
    shots = events[events["type"] == "Shot"]
    if len(shots):
        detail = shots["shot"].apply(_safe_literal_eval)
        is_goal = detail.apply(
            lambda d: isinstance(d, dict) and d.get("outcome", {}).get("name") == "Goal"
        )
        frames.append(shots[is_goal][["match_id", "minute", "team"]])
    own_goals = events[events["type"] == "Own Goal For"]
    if len(own_goals):
        frames.append(own_goals[["match_id", "minute", "team"]])
    if not frames:
        return pd.DataFrame(columns=["match_id", "minute", "scoring_team"])
    return pd.concat(frames, ignore_index=True).rename(columns={"team": "scoring_team"})
def _extract_card_events(events: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col, ev_type in [("foul_committed", "Foul Committed"), ("bad_behaviour", "Bad Behaviour")]:
        if col not in events.columns:
            log.warning("events table has no '%s' column -- %s card events cannot be "
                        "captured; re-run ingest_integrate.py with that column included",
                        col, ev_type)
            continue
        sub = events[events["type"] == ev_type]
        if not len(sub):
            continue
        detail = sub[col].apply(_safe_literal_eval)
        card_name = detail.apply(lambda d: d.get("card", {}).get("name") if isinstance(d, dict) else None)
        sub = sub[card_name.isin(["Red Card", "Second Yellow"])]
        if len(sub):
            frames.append(sub[["match_id", "minute", "team"]])
    if not frames:
        return pd.DataFrame(columns=["match_id", "minute", "offending_team"])
    return pd.concat(frames, ignore_index=True).rename(columns={"team": "offending_team"})
def build_snapshot_index(labeled_matches: pd.DataFrame, events: pd.DataFrame,
                          interval: int = SNAPSHOT_INTERVAL_MIN,
                          end_min: int = REGULATION_END_MIN) -> pd.DataFrame:
    goals = _extract_goal_events(events)
    cards = _extract_card_events(events)
    normal_time = events[events["period"].isin([1, 2])] if "period" in events.columns else events
    last_minute_by_match = normal_time.groupby("match_id")["minute"].max()
    events_by_match = {mid: g["minute"].to_numpy() for mid, g in events.groupby("match_id")}
    goals_by_match = {mid: g for mid, g in goals.groupby("match_id")}
    cards_by_match = {mid: g for mid, g in cards.groupby("match_id")}
    empty_goals = goals.iloc[0:0]
    empty_cards = cards.iloc[0:0]
    have_events = set(events_by_match.keys())
    total = len(labeled_matches)
    matches_with_events = labeled_matches[labeled_matches["match_id"].isin(have_events)]
    dropped = total - len(matches_with_events)
    if dropped:
        log.warning("DROPPED %d/%d matches from the snapshot index: no event stream in "
                     "events.csv (they would otherwise get an all-zero phantom snapshot grid)",
                     dropped, total)
    rows = []
    short_matches = 0
    for _, m in matches_with_events.iterrows():
        mid = m["match_id"]
        home, away = m["home_team_name"], m["away_team_name"]
        ev_minutes = events_by_match[mid]
        gm = goals_by_match.get(mid, empty_goals)
        cm = cards_by_match.get(mid, empty_cards)
        match_last_minute = int(last_minute_by_match.get(mid, end_min))
        match_end = min(match_last_minute, end_min)
        if match_last_minute < end_min:
            short_matches += 1
        for minute in range(interval, match_end + 1, interval):
            g_upto = gm[gm["minute"] <= minute]
            c_upto = cm[cm["minute"] <= minute]
            home_score = int((g_upto["scoring_team"] == home).sum())
            away_score = int((g_upto["scoring_team"] == away).sum())
            home_reds = int((c_upto["offending_team"] == home).sum())
            away_reds = int((c_upto["offending_team"] == away).sum())
            events_so_far = int(np.searchsorted(ev_minutes, minute, side="right"))
            window_start = max(0, minute - RECENT_WINDOW_MIN)
            events_recent = events_so_far - int(np.searchsorted(ev_minutes, window_start, side="right"))
            rows.append({
                "match_id": mid,
                "snapshot_minute": minute,
                "current_home_score": home_score,
                "current_away_score": away_score,
                "current_score_diff": home_score - away_score,
                "home_red_cards": home_reds,
                "away_red_cards": away_reds,
                "man_advantage": away_reds - home_reds,
                "events_so_far": events_so_far,
                f"events_last_{RECENT_WINDOW_MIN}min": events_recent,
                "outcome": m["outcome"],
                "margin": m["margin"],
                "competition_id": m["competition_id"],
                "season_id": m["season_id"],
                "match_date": m["match_date"],
            })
    if short_matches:
        log.warning("%d/%d matches had fewer than %d minutes of recorded events; their "
                     "snapshot grid was capped at that match's own last event minute instead "
                     "of extending to %d with fabricated data",
                     short_matches, len(matches_with_events), end_min, end_min)
    snap = pd.DataFrame(rows)
    log.info("snapshot index -> %d snapshot rows from %d matches, grid capped per-match by "
              "real event data (not a blind fixed 0-%d grid for every match)",
              len(snap), matches_with_events["match_id"].nunique(), end_min)
    return snap
#Formal problem definition (written, not just implied by code)
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
                        f"{SNAPSHOT_INTERVAL_MIN} up to min({REGULATION_END_MIN}, that match's own "
                        f"last recorded event minute) -- matches with no event stream get no rows",
        "leakage_rule": (
            "for the row at snapshot_minute=t, zero events with timestamp > t may "
            "contribute to any feature (STRICT time-t cut). All snapshot rows of one "
            "match_id must be placed entirely on one side of the train/val/test split "
            "-- never split within a match."
        ),
    },
}
def write_problem_definition(out_dir: Path) -> None:
    (out_dir / "problem_definition.json").write_text(json.dumps(PROBLEM_DEFINITION, indent=2), encoding="utf-8")
    md = ["# Problem Definition (Stage 2 deliverable)\n"]
    for task_id, spec in PROBLEM_DEFINITION.items():
        md.append(f"## {task_id}: {spec['name']}\n")
        md.append(f"- **One example is:** {spec['one_example_is']}")
        md.append(f"- **Input (X):** {spec['input_x']}")
        md.append(f"- **Label (y):** {spec['label_y']}")
        md.append(f"- **Granularity:** {spec['granularity']}")
        md.append(f"- **Leakage rule:** {spec['leakage_rule']}\n")
    (out_dir / "problem_definition.md").write_text("\n".join(md), encoding="utf-8")
    log.info("wrote problem_definition.json / .md -> %s", out_dir)
def run(processed_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(processed_dir / "matches.csv", parse_dates=["match_date"], encoding="utf-8")
    events = pd.read_csv(processed_dir / "events.csv", encoding="utf-8")
    labeled = label_matches(matches)
    labeled.to_csv(out_dir / "labeled_matches.csv", index=False, encoding="utf-8")
    log.info("wrote labeled_matches.csv (%d rows) -> %s", len(labeled), out_dir)
    snap_index = build_snapshot_index(labeled, events)
    snap_index.to_csv(out_dir / "snapshot_index.csv", index=False, encoding="utf-8")
    log.info("wrote snapshot_index.csv (%d rows) -> %s", len(snap_index), out_dir)
    write_problem_definition(out_dir)
run(Path(PROCESSED), Path(OUT))