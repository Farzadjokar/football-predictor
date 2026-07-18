from __future__ import annotations
import ast
import csv
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO,format="%(levelname)s:%(message)s");
log=logging.getLogger("preprocess_clean");
PROCESSED="out_csv";
OUT="out_csv";
CHUNK_SIZE=50_000;  # rows per chunk; tune down if RAM is still tight

# ---------------------------------------------------------------------------
# Stage 3 deliverable: preprocessing & cleaning -- CHUNKED VERSION.
#
# Why this rewrite exists: the previous version read the entire events.csv
# into one DataFrame and ran 5 separate .apply(ast.literal_eval) passes over
# it. At full StatsBomb scale (millions of rows) that is both extremely slow
# (literal_eval is ~20x slower than json.loads) and memory-heavy enough to
# freeze the machine, since pandas keeps the original + several intermediate
# Series alive at once with nothing printed to show progress.
#
# This version:
#   1. Uses json.loads (via a light string-repair step) instead of
#      ast.literal_eval -- meaningfully faster for this volume.
#   2. Processes events.csv in fixed-size chunks via pd.read_csv(chunksize=..),
#      writing each cleaned chunk straight to disk and discarding it before
#      reading the next -- peak memory becomes O(chunk), not O(whole file).
#   3. Logs progress every chunk, with elapsed time and running row counts,
#      so a long run is visibly alive instead of silent.
# ---------------------------------------------------------------------------

def _pyrepr_to_json(s:str)->str:
    """Convert a Python-repr'd dict/list string (single quotes, None/True/False)
    into valid JSON text, so json.loads can parse it. This is a plain string
    substitution, not code execution -- much cheaper than ast.literal_eval
    for this volume, and safe because we only ever call json.loads on the
    result (json.loads never executes arbitrary code, unlike eval)."""
    if s.count("'")==0 and s.count('"')>0:
        return s;  # already looks like JSON
    out=s.replace('"','\\"');       # escape any literal double-quotes first
    out=out.replace("'",'"');        # python single-quotes -> JSON double-quotes
    out=out.replace("None","null").replace("True","true").replace("False","false");
    return out;

def _parse_maybe_dict(val):
    if isinstance(val,dict):
        return val;
    if isinstance(val,str) and val.startswith("{") and val.endswith("}"):
        try:
            return json.loads(_pyrepr_to_json(val));
        except (json.JSONDecodeError,ValueError):
            try:
                return ast.literal_eval(val);  # fallback for edge cases the quick path misses
            except (ValueError,SyntaxError):
                return None;
    return None;

def _parse_maybe_list(val):
    if isinstance(val,list):
        return val;
    if isinstance(val,str) and val.startswith("[") and val.endswith("]"):
        try:
            return json.loads(val);
        except (json.JSONDecodeError,ValueError):
            try:
                return ast.literal_eval(val);
            except (ValueError,SyntaxError):
                return None;
    return None;

def _get_name(d,*keys):
    cur=d;
    for k in keys:
        if not isinstance(cur,dict):
            return None;
        cur=cur.get(k);
    if isinstance(cur,dict):
        return cur.get("name");
    return cur;

def flatten_events_chunk(df:pd.DataFrame)->pd.DataFrame:
    """Same logic as before, applied to ONE chunk instead of the whole file."""
    n_before=len(df);

    if "location" in df.columns:
        loc=df["location"].apply(_parse_maybe_list);
        df["loc_x"]=loc.apply(lambda v: v[0] if isinstance(v,list) and len(v)==2 else np.nan);
        df["loc_y"]=loc.apply(lambda v: v[1] if isinstance(v,list) and len(v)==2 else np.nan);

    if "pass" in df.columns:
        pass_obj=df["pass"].apply(_parse_maybe_dict);
        df["pass_recipient"]=pass_obj.apply(lambda d: _get_name(d,"recipient"));
        df["pass_height"]=pass_obj.apply(lambda d: _get_name(d,"height"));
        df["pass_body_part"]=pass_obj.apply(lambda d: _get_name(d,"body_part"));
        df["pass_outcome"]=pass_obj.apply(lambda d: _get_name(d,"outcome"));
        df["pass_length"]=pass_obj.apply(lambda d: d.get("length") if isinstance(d,dict) else np.nan);
        df["pass_completed"]=df["pass_outcome"].isna() & df["type"].eq("Pass");

    if "shot" in df.columns:
        shot_obj=df["shot"].apply(_parse_maybe_dict);
        df["shot_outcome"]=shot_obj.apply(lambda d: _get_name(d,"outcome"));
        df["shot_body_part"]=shot_obj.apply(lambda d: _get_name(d,"body_part"));
        df["shot_statsbomb_xg"]=shot_obj.apply(lambda d: d.get("statsbomb_xg") if isinstance(d,dict) else np.nan);
        df["shot_is_goal"]=df["shot_outcome"].eq("Goal");

    if "carry" in df.columns:
        carry_obj=df["carry"].apply(_parse_maybe_dict);
        carry_end=carry_obj.apply(lambda d: d.get("end_location") if isinstance(d,dict) else None);
        df["carry_end_x"]=carry_end.apply(lambda v: v[0] if isinstance(v,list) and len(v)==2 else np.nan);
        df["carry_end_y"]=carry_end.apply(lambda v: v[1] if isinstance(v,list) and len(v)==2 else np.nan);

    if "duel" in df.columns:
        duel_obj=df["duel"].apply(_parse_maybe_dict);
        df["duel_type"]=duel_obj.apply(lambda d: _get_name(d,"type"));
        df["duel_outcome"]=duel_obj.apply(lambda d: _get_name(d,"outcome"));

    if "foul_committed" in df.columns:
        foul_obj=df["foul_committed"].apply(_parse_maybe_dict);
        df["foul_card"]=foul_obj.apply(lambda d: _get_name(d,"card"));
        df["is_red_card_event"]=df["foul_card"].isin(["Red Card","Second Yellow"]);

    raw_nested_cols=[c for c in ["pass","shot","carry","duel","pressure","dribble",
                                  "foul_committed","location"] if c in df.columns];
    df=df.drop(columns=raw_nested_cols);
    if len(df)!=n_before:
        raise ValueError("flatten_events_chunk must never change row count");
    return df;

def clean_events_chunk(df:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    """Same cleaning rules as before, applied per chunk. Returns the cleaned
    chunk plus a small stats dict so the caller can accumulate totals across
    chunks (duplicates need a cross-chunk key check done separately, see
    run_events_pipeline)."""
    n0=len(df);
    stats={"n_bad_time":0,"n_oob":0,"n_shootout":0};

    for col in ["minute","second","index"]:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors="coerce");
    bad_time=df["minute"].isna() if "minute" in df.columns else pd.Series(False,index=df.index);
    stats["n_bad_time"]=int(bad_time.sum());
    df=df[~bad_time];

    if "period" in df.columns:
        df["period"]=pd.to_numeric(df["period"],errors="coerce");
        df["is_shootout_event"]=df["period"].eq(5);
        stats["n_shootout"]=int(df["is_shootout_event"].sum());

    if "loc_x" in df.columns and "loc_y" in df.columns:
        out_of_bounds=(
            df["loc_x"].notna() & ((df["loc_x"]<0) | (df["loc_x"]>120))
        ) | (
            df["loc_y"].notna() & ((df["loc_y"]<0) | (df["loc_y"]>80))
        );
        stats["n_oob"]=int(out_of_bounds.sum());
        if stats["n_oob"]:
            df.loc[out_of_bounds,"loc_x"]=df.loc[out_of_bounds,"loc_x"].clip(0,120);
            df.loc[out_of_bounds,"loc_y"]=df.loc[out_of_bounds,"loc_y"].clip(0,80);

    return df,stats;

def run_events_pipeline(src_path:Path,dst_path:Path,chunk_size:int=CHUNK_SIZE)->None:
    """Stream events.csv -> events_clean.csv in chunks, with progress logging.
    Duplicate (match_id, id) detection is done via a running hash-set of keys
    seen so far, so it still works correctly across chunk boundaries without
    ever loading the whole file at once."""
    t_start=time.time();
    seen_keys=set();
    n_rows_in=0;
    n_rows_out=0;
    n_dup=0;
    total_stats={"n_bad_time":0,"n_oob":0,"n_shootout":0};
    first_chunk=True;

    reader=pd.read_csv(src_path,chunksize=chunk_size,encoding="utf-8");
    for chunk_idx,raw_chunk in enumerate(reader):
        n_rows_in+=len(raw_chunk);

        flat_chunk=flatten_events_chunk(raw_chunk);

        if "match_id" in flat_chunk.columns and "id" in flat_chunk.columns:
            key_series=list(zip(flat_chunk["match_id"],flat_chunk["id"]));
            dup_mask=pd.Series([k in seen_keys for k in key_series],index=flat_chunk.index);
            n_dup_here=int(dup_mask.sum());
            n_dup+=n_dup_here;
            flat_chunk=flat_chunk[~dup_mask];
            seen_keys.update(k for k in key_series if k not in seen_keys);

        clean_chunk,stats=clean_events_chunk(flat_chunk);
        for k,v in stats.items():
            total_stats[k]+=v;

        clean_chunk.to_csv(dst_path,mode="w" if first_chunk else "a",
                            header=first_chunk,index=False,encoding="utf-8");
        first_chunk=False;
        n_rows_out+=len(clean_chunk);

        elapsed=time.time()-t_start;
        log.info("events chunk %d: %d rows in (running total %d) -> %d rows kept "
                  "(running total %d) | elapsed %.1fs",
                  chunk_idx+1,len(raw_chunk),n_rows_in,len(clean_chunk),n_rows_out,elapsed);

    elapsed=time.time()-t_start;
    log.info("run_events_pipeline COMPLETE: %d rows in -> %d rows out "
              "(dropped: %d dup, %d bad-time; flagged %d shootout, clamped %d oob) "
              "in %.1fs -> %s",
              n_rows_in,n_rows_out,n_dup,total_stats["n_bad_time"],
              total_stats["n_shootout"],total_stats["n_oob"],elapsed,dst_path);

def clean_lineups(lineups:pd.DataFrame)->pd.DataFrame:
    df=lineups.copy();
    n0=len(df);
    dup_mask=df.duplicated(subset=["match_id","team_id","player_id"],keep="first");
    n_dup=int(dup_mask.sum());
    if n_dup:
        log.warning("DROPPED %d duplicate lineup rows (same player listed twice for one match)",n_dup);
    df=df[~dup_mask];
    missing_name=df["player_name"].isna();
    n_missing=int(missing_name.sum());
    if n_missing:
        log.warning("%d lineup rows have missing player_name (kept -- player_id still usable as key)",
                    n_missing);
    log.info("clean_lineups: %d rows in -> %d rows out",n0,len(df));
    return df;

def clean_matches(labeled_matches:pd.DataFrame)->pd.DataFrame:
    df=labeled_matches.copy();
    n0=len(df);

    dup_mask=df.duplicated(subset=["match_id"],keep="first");
    n_dup=int(dup_mask.sum());
    if n_dup:
        log.warning("DROPPED %d duplicate match_id rows in matches table",n_dup);
    df=df[~dup_mask];

    same_team=df["home_team_id"]==df["away_team_id"];
    n_same=int(same_team.sum());
    if n_same:
        log.warning("DROPPED %d matches where home_team_id == away_team_id (data-quality error)",
                    n_same);
    df=df[~same_team];

    missing_date=df["match_date"].isna();
    n_missing_date=int(missing_date.sum());
    if n_missing_date:
        log.warning("DROPPED %d matches with missing match_date (cannot be temporally split)",
                    n_missing_date);
    df=df[~missing_date];

    log.info("clean_matches: %d rows in -> %d rows out (%d dropped total)",
              n0,len(df),n0-len(df));
    return df.sort_values("match_date").reset_index(drop=True);

def validate_no_orphan_snapshots(snapshot_index:pd.DataFrame,matches:pd.DataFrame)->pd.DataFrame:
    n0=len(snapshot_index);
    valid_ids=set(matches["match_id"]);
    orphan_mask=~snapshot_index["match_id"].isin(valid_ids);
    n_orphan=int(orphan_mask.sum());
    if n_orphan:
        log.warning("DROPPED %d snapshot rows whose match_id was removed during match cleaning",
                    n_orphan);
    out=snapshot_index[~orphan_mask].reset_index(drop=True);
    log.info("validate_no_orphan_snapshots: %d rows in -> %d rows out",n0,len(out));
    return out;

def run(processed_dir:Path,out_dir:Path)->None:
    out_dir.mkdir(parents=True,exist_ok=True);

    log.info("=== stage: events (chunked, %d rows/chunk) ===",CHUNK_SIZE);
    run_events_pipeline(processed_dir/"events.csv",out_dir/"events_clean.csv");

    log.info("=== stage: lineups ===");
    lineups_raw=pd.read_csv(processed_dir/"lineups.csv",encoding="utf-8");
    lineups_clean=clean_lineups(lineups_raw);
    lineups_clean.to_csv(out_dir/"lineups_clean.csv",index=False,encoding="utf-8");
    log.info("wrote lineups_clean.csv (%d rows)->%s",len(lineups_clean),out_dir);

    log.info("=== stage: matches ===");
    labeled_path=processed_dir/"labeled_matches.csv";
    matches_raw=pd.read_csv(labeled_path,parse_dates=["match_date"],encoding="utf-8");
    matches_clean=clean_matches(matches_raw);
    matches_clean.to_csv(out_dir/"matches_clean.csv",index=False,encoding="utf-8");
    log.info("wrote matches_clean.csv (%d rows)->%s",len(matches_clean),out_dir);

    log.info("=== stage: snapshot index ===");
    snap_path=processed_dir/"snapshot_index.csv";
    snap_raw=pd.read_csv(snap_path,parse_dates=["match_date"],encoding="utf-8");
    snap_clean=validate_no_orphan_snapshots(snap_raw,matches_clean);
    snap_clean.to_csv(out_dir/"snapshot_index_clean.csv",index=False,encoding="utf-8");
    log.info("wrote snapshot_index_clean.csv (%d rows)->%s",len(snap_clean),out_dir);

    log.info("=== preprocess_clean.py: ALL STAGES COMPLETE ===");

run(Path(PROCESSED),Path(OUT));
