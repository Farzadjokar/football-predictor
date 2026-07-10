from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
logging.basicConfig(level=logging.INFO,format="%(levelname)s:%(message)s");
log=logging.getLogger("ingest_integrate");
ROOT="";
OUT="out_csv";
COMP_SEASON_PAIRS=None;
def load_competitions(root:Path)->pd.DataFrame:
    path=root/"data"/"competitions.json";
    raw=json.loads(path.read_text(encoding="utf-8"));
    df=pd.DataFrame(raw);
    log.info("competitions.json->%d competition-season rows",len(df));
    return df;
def discover_comp_season_pairs(root:Path)->list[tuple[int,int]]:
    pairs=[];
    matches_dir=root/"data"/"matches";
    for comp_dir in sorted(matches_dir.iterdir()):
        if not comp_dir.is_dir():
            continue;
        for season_file in sorted(comp_dir.glob("*.json")):
            pairs.append((int(comp_dir.name), int(season_file.stem)));
    log.info("auto-discovered %d competition/season pairs under %s", len(pairs), matches_dir);
    return pairs;
def load_matches(root:Path,comp_season_pairs:list[tuple[int,int]])->pd.DataFrame:
    frames=[];
    for comp_id,season_id in comp_season_pairs:
        path=root/"data"/"matches"/str(comp_id)/f"{season_id}.json";
        if not path.exists():
            log.warning("DROPPED comp=%s season=%s:matches file not found at %s",
                        comp_id,season_id,path);
            continue;
        raw=json.loads(path.read_text(encoding="utf-8"));
        df=pd.json_normalize(raw,sep="_");
        df["competition_id"]=comp_id;
        df["season_id"]=season_id;
        frames.append(df);
        log.info("matches/%s/%s.json -> %d matches",comp_id, season_id,len(df));
    matches=pd.concat(frames,ignore_index=True);
    keep=[
        "match_id", "competition_id", "season_id", "match_date", "kick_off",
        "home_team_home_team_id", "home_team_home_team_name",
        "away_team_away_team_id", "away_team_away_team_name",
        "home_score", "away_score", "match_status", "match_week",
        "competition_stage_name", "stadium_name", "referee_name",
    ];
    keep=[c for c in keep if c in matches.columns];
    dropped_cols=set(matches.columns)-set(keep);
    log.info("matches: keeping %d columns, dropping %d nested/unused columns",
              len(keep),len(dropped_cols));
    matches=matches[keep].rename(columns={
        "home_team_home_team_id": "home_team_id",
        "home_team_home_team_name": "home_team_name",
        "away_team_away_team_id": "away_team_id",
        "away_team_away_team_name": "away_team_name",
    });
    before=len(matches);
    matches=matches.dropna(subset=["home_score","away_score"]);
    after=len(matches);
    if before != after:
        log.warning("DROPPED %d matches with missing final score (unplayed/void)",
                    before-after);
    matches["match_date"]=pd.to_datetime(matches["match_date"]);
    matches["home_score"]=matches["home_score"].astype(int);
    matches["away_score"]=matches["away_score"].astype(int);
    return matches.sort_values("match_date").reset_index(drop=True);
def load_lineups(root:Path,match_ids:list[int])->pd.DataFrame:
    rows=[];
    missing=0;
    for mid in match_ids:
        path=root/"data"/"lineups"/f"{mid}.json";
        if not path.exists():
            missing+=1;
            continue;
        raw=json.loads(path.read_text(encoding="utf-8"));
        for team in raw:
            team_id=team["team_id"];
            team_name=team["team_name"];
            for p in team["lineup"]:
                rows.append({
                    "match_id": mid,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "jersey_number": p.get("jersey_number"),
                    "country": p.get("country", {}).get("name") if p.get("country") else None,
                });
    if missing:
        log.warning("DROPPED %d/%d matches: no lineups file (not released for those matches)",
                    missing,len(match_ids));
    lineups=pd.DataFrame(rows);
    log.info("lineups -> %d (match, team, player) rows from %d matches",
              len(lineups),len(match_ids)-missing);
    return lineups;
_EVENT_COLS=[
    "id", "index", "period", "timestamp", "minute", "second",
    "type", "possession", "possession_team", "play_pattern",
    "team", "player", "position", "location",
    "duration", "under_pressure", "counterpress",
    "pass", "shot", "carry", "duel", "pressure", "dribble", "foul_committed",
];
def _flatten_name(val):
    """StatsBomb nests {'id':.., 'name':..} for many fields; keep the name."""
    if isinstance(val,dict):
        return val.get("name");
    return val;
def load_events(root:Path,match_ids:list[int])->pd.DataFrame:
    frames=[];
    missing=0;
    malformed=0;
    for mid in match_ids:
        path=root/"data"/"events"/f"{mid}.json";
        if not path.exists():
            missing+=1;
            continue;
        try:
            raw=json.loads(path.read_text(encoding="utf-8"));
        except json.JSONDecodeError:
            malformed+=1;
            log.warning("DROPPED match_id=%s: events file failed to parse (malformed JSON)",mid);
            continue;
        df=pd.DataFrame(raw);
        df["match_id"]=mid;
        for col in ["type","possession_team","team","player","position","play_pattern"]:
            if col in df.columns:
                df[col]=df[col].apply(_flatten_name);
        keep=[c for c in _EVENT_COLS if c in df.columns]+["match_id"];
        frames.append(df[keep]);
    if missing:
        log.warning("DROPPED %d/%d matches: no events file",missing,len(match_ids));
    if malformed:
        log.warning("DROPPED %d matches: malformed events JSON",malformed);
    events=pd.concat(frames,ignore_index=True);
    events=events.sort_values(["match_id","period","timestamp","index"]).reset_index(drop=True);
    log.info("events->%d rows from %d matches(%.0fevents/match on average)",
              len(events),len(match_ids)-missing,len(events)/max(1,len(match_ids)-missing));
    return events;
def build_relational_store(root:Path,out_dir:Path,
                            comp_season_pairs:list[tuple[int,int]])->dict[str,pd.DataFrame]:
    out_dir.mkdir(parents=True,exist_ok=True);
    competitions=load_competitions(root);
    matches=load_matches(root,comp_season_pairs);
    match_ids=matches["match_id"].tolist();
    lineups=load_lineups(root,match_ids);
    events=load_events(root,match_ids);
#every event/lineup match_id must exist in matches (referential integrity)
    bad_events=set(events["match_id"])-set(matches["match_id"]);
    if bad_events:
        log.warning("DROPPING %d events belonging to match_ids absent from matches table",
                    len(events[events["match_id"].isin(bad_events)]));
        events=events[~events["match_id"].isin(bad_events)];
    tables={
        "competitions": competitions,
        "matches": matches,
        "lineups": lineups,
        "events": events,
    };
    for name,df in tables.items():
        df.to_csv(out_dir/f"{name}.csv",index=False,encoding="utf-8");
        log.info("wrote%s(%drows,%dcols)->%s",name,len(df),df.shape[1],
                  out_dir/f"{name}.csv");
    return tables;
#path of data and path of result
root_path=Path(ROOT);
out_path=Path(OUT);
pairs=COMP_SEASON_PAIRS if COMP_SEASON_PAIRS is not None else discover_comp_season_pairs(root_path);
build_relational_store(root_path,out_path,pairs);