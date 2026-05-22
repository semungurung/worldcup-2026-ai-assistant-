from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "derived"
CLEAN_DIR = DATA_DIR / "cleaned"

TEAM_ALIASES = {
    "United States": "USA",
    "US": "USA",
    "U.S.": "USA",
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Cape Verde": "Cabo Verde",
    "Curaçao": "Curacao",
    "Iran": "IR Iran",
    "Türkiye": "Turkiye",
    "Turkey": "Turkiye",
}

VIDEO_EXCLUSIONS = [
    "women",
    "women's",
    "womens",
    "lionesses",
    "female",
    "girls",
    "uswnt",
    "canwnt",
    "wnt",
    "u17",
    "u-17",
    "u20",
    "u-20",
    "u21",
    "u-21",
    "u23",
    "u-23",
    "under-17",
    "under 17",
    "under-20",
    "under 20",
    "under-21",
    "under 21",
    "under-23",
    "under 23",
    "futsal",
    "beach soccer",
    "efootball",
    "nfl",
    "football 🏈",
]

CORE_VIDEO_TERMS = [
    "world cup",
    "fifa world cup",
    "2026",
    "qualifier",
    "training",
    "press conference",
    "tactical",
    "analysis",
    "squad",
    "friendly",
    "usmnt",
    "canmnt",
    "three lions",
    "men's national",
    "mens national",
]

SOURCE_REQUIRED_VIDEO_TERMS = {
    "USA": ["usmnt", "men's national", "mens national", "world cup", "qualifier", "friendly"],
    "Canada": ["canmnt", "men's national", "mens national", "world cup", "qualifier", "friendly"],
    "England": ["england men", "men's", "mens", "three lions", "world cup", "qualifier", "friendly"],
    "Media": ["soccer", "fifa", "world cup", "champions league", "premier league", "laliga", "serie a", "bundesliga"],
}


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def normalize_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    return cleaned


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


def clean_teams() -> pd.DataFrame:
    teams = normalize_text_columns(read_csv(DATA_DIR / "teams.csv"))
    teams["team"] = teams["team"].map(normalize_team)
    teams["group"] = teams["group"].str.upper()
    teams["host"] = pd.to_numeric(teams["host"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    for column in ["power_rating", "form_index", "attack", "midfield", "defense", "depth"]:
        teams[column] = pd.to_numeric(teams[column], errors="coerce").clip(0, 100)
        teams[column] = teams[column].fillna(teams[column].median()).round(2)
    teams = teams.drop_duplicates(subset=["team"]).sort_values(["group", "team"])
    write_csv(teams, CLEAN_DIR / "teams_clean.csv")
    return teams


def clean_fixtures(valid_teams: set[str]) -> pd.DataFrame:
    fixtures = normalize_text_columns(read_csv(DATA_DIR / "fixtures.csv"))
    fixtures["team_a"] = fixtures["team_a"].map(normalize_team)
    fixtures["team_b"] = fixtures["team_b"].map(normalize_team)
    fixtures["group"] = fixtures["group"].str.upper()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.date.astype(str).replace("NaT", "")
    fixtures = fixtures[
        fixtures["team_a"].isin(valid_teams) & fixtures["team_b"].isin(valid_teams) & (fixtures["team_a"] != fixtures["team_b"])
    ].copy()
    fixtures = fixtures.drop_duplicates(subset=["match_id"])
    write_csv(fixtures, CLEAN_DIR / "fixtures_clean.csv")
    return fixtures


def clean_players(valid_teams: set[str]) -> pd.DataFrame:
    players = normalize_text_columns(read_csv(DATA_DIR / "players.csv"))
    players["team"] = players["team"].map(normalize_team)
    players = players[players["team"].isin(valid_teams)].copy()
    for column in ["role_rating", "expected_minutes", "injury_risk", "goals_xg"]:
        players[column] = pd.to_numeric(players[column], errors="coerce").fillna(0)
    players["role_rating"] = players["role_rating"].clip(0, 100)
    players["expected_minutes"] = players["expected_minutes"].clip(lower=0)
    players["injury_risk"] = players["injury_risk"].clip(0, 1)
    players["goals_xg"] = players["goals_xg"].clip(lower=0)
    players = players.drop_duplicates(subset=["team", "player_name", "position"])
    write_csv(players, CLEAN_DIR / "players_clean.csv")
    return players


def clean_injuries(valid_teams: set[str]) -> pd.DataFrame:
    injuries = normalize_text_columns(read_csv(DATA_DIR / "injuries.csv"))
    injuries["team"] = injuries["team"].map(normalize_team)
    injuries = injuries[injuries["team"].isin(valid_teams)].copy()
    injuries["status"] = injuries["status"].str.lower()
    injuries["severity"] = injuries["severity"].str.lower()
    injuries["expected_return"] = pd.to_datetime(injuries["expected_return"], errors="coerce").dt.date.astype(str).replace("NaT", "")
    injuries = injuries.drop_duplicates(subset=["team", "player_name", "status", "severity"])
    write_csv(injuries, CLEAN_DIR / "injuries_clean.csv")
    return injuries


def clean_match_events(valid_teams: set[str]) -> pd.DataFrame:
    events = normalize_text_columns(read_csv(DATA_DIR / "match_events.csv"))
    if events.empty:
        return events
    events["team"] = events["team"].map(normalize_team)
    events = events[events["team"].isin(valid_teams)].copy()
    events["minute"] = pd.to_numeric(events["minute"], errors="coerce").fillna(0).clip(0, 130).astype(int)
    events["xg"] = pd.to_numeric(events["xg"], errors="coerce").fillna(0).clip(0, 1)
    events = events.drop_duplicates(subset=["match_id", "minute", "team", "player", "event_type", "description"])
    write_csv(events, CLEAN_DIR / "match_events_clean.csv")
    return events


def clean_historical_results(valid_teams: set[str]) -> pd.DataFrame:
    results = normalize_text_columns(read_csv(RAW_DIR / "international_results.csv"))
    if results.empty:
        return results
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_score", "away_score"]).copy()
    results = results[results["date"] <= pd.Timestamp(date.today())].copy()
    results["home_team_norm"] = results["home_team"].map(normalize_team)
    results["away_team_norm"] = results["away_team"].map(normalize_team)
    results = results[results["home_team_norm"].isin(valid_teams) | results["away_team_norm"].isin(valid_teams)].copy()
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce").fillna(0).astype(int)
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce").fillna(0).astype(int)
    results = results.drop_duplicates(subset=["date", "home_team_norm", "away_team_norm", "home_score", "away_score", "tournament"])
    write_csv(results, CLEAN_DIR / "worldcup_team_historical_results_clean.csv")
    return results


def clean_recent_form(valid_teams: set[str]) -> pd.DataFrame:
    form = normalize_text_columns(read_csv(DERIVED_DIR / "recent_team_form.csv"))
    if form.empty:
        return form
    form["team"] = form["team"].map(normalize_team)
    form = form[form["team"].isin(valid_teams)].drop_duplicates(subset=["team"]).copy()
    numeric_columns = [column for column in form.columns if column not in {"team", "last_match_date"}]
    for column in numeric_columns:
        form[column] = pd.to_numeric(form[column], errors="coerce").fillna(0)
    write_csv(form, CLEAN_DIR / "recent_team_form_clean.csv")
    return form


def clean_scorers(valid_teams: set[str]) -> pd.DataFrame:
    scorers = normalize_text_columns(read_csv(DERIVED_DIR / "recent_scorer_form.csv"))
    if scorers.empty:
        return scorers
    scorers["team"] = scorers["team"].map(normalize_team)
    scorers = scorers[scorers["team"].isin(valid_teams)].copy()
    for column in ["goals", "penalty_goals", "own_goals"]:
        scorers[column] = pd.to_numeric(scorers[column], errors="coerce").fillna(0).astype(int)
    scorers = scorers[scorers["own_goals"] == 0].copy()
    scorers = scorers.drop_duplicates(subset=["team", "scorer"])
    write_csv(scorers, CLEAN_DIR / "recent_scorer_form_clean.csv")
    return scorers


def clean_video_metadata() -> pd.DataFrame:
    videos = normalize_text_columns(read_csv(DERIVED_DIR / "video_metadata.csv"))
    if videos.empty:
        return videos
    text = (videos["title"] + " " + videos.get("matched_keywords", "")).str.lower()
    exclusion_mask = text.apply(lambda value: any(keyword in value for keyword in VIDEO_EXCLUSIONS))
    core_mask = text.apply(lambda value: any(keyword in value for keyword in CORE_VIDEO_TERMS))
    source_mask = pd.Series([True] * len(videos), index=videos.index)
    for team_or_scope, required_terms in SOURCE_REQUIRED_VIDEO_TERMS.items():
        row_mask = videos["team_or_scope"] == team_or_scope
        source_mask.loc[row_mask] = text.loc[row_mask].apply(lambda value: any(term in value for term in required_terms))
    if "is_mens_world_cup_relevant" in videos:
        source_relevance = videos["is_mens_world_cup_relevant"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        source_relevance = pd.Series([True] * len(videos), index=videos.index)
    videos["is_relevant_mens_worldcup_2026"] = (
        source_relevance
        & ~exclusion_mask
        & core_mask
        & source_mask
        & (pd.to_numeric(videos["relevance_score"], errors="coerce").fillna(0) > 0)
    )
    videos["cleaning_exclusion_reason"] = exclusion_mask.map(lambda excluded: "excluded_non_mens_or_non_senior" if excluded else "")
    clean_relevant = videos[videos["is_relevant_mens_worldcup_2026"]].drop_duplicates(subset=["video_id"]).copy()
    write_csv(videos, CLEAN_DIR / "video_metadata_audited.csv")
    write_csv(clean_relevant, CLEAN_DIR / "video_metadata_relevant_clean.csv")
    return clean_relevant


def clean_rankings(valid_teams: set[str]) -> pd.DataFrame:
    rankings = normalize_text_columns(read_csv(DERIVED_DIR / "fifa_rankings_wc_teams.csv"))
    if rankings.empty:
        return rankings
    rankings["team"] = rankings["team"].map(normalize_team)
    rankings = rankings[rankings["team"].isin(valid_teams)].drop_duplicates(subset=["team"]).copy()
    rankings["fifa_rank_april_2026"] = pd.to_numeric(rankings["fifa_rank_april_2026"], errors="coerce")
    write_csv(rankings, CLEAN_DIR / "fifa_rankings_wc_teams_clean.csv")
    return rankings


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    teams = clean_teams()
    valid_teams = set(teams["team"])
    fixtures = clean_fixtures(valid_teams)
    players = clean_players(valid_teams)
    injuries = clean_injuries(valid_teams)
    events = clean_match_events(valid_teams)
    historical = clean_historical_results(valid_teams)
    form = clean_recent_form(valid_teams)
    scorers = clean_scorers(valid_teams)
    videos = clean_video_metadata()
    rankings = clean_rankings(valid_teams)

    summary = pd.DataFrame(
        [
            {"dataset": "teams", "clean_rows": len(teams), "file": "cleaned/teams_clean.csv"},
            {"dataset": "fixtures", "clean_rows": len(fixtures), "file": "cleaned/fixtures_clean.csv"},
            {"dataset": "players", "clean_rows": len(players), "file": "cleaned/players_clean.csv"},
            {"dataset": "injuries", "clean_rows": len(injuries), "file": "cleaned/injuries_clean.csv"},
            {"dataset": "match_events", "clean_rows": len(events), "file": "cleaned/match_events_clean.csv"},
            {"dataset": "historical_results", "clean_rows": len(historical), "file": "cleaned/worldcup_team_historical_results_clean.csv"},
            {"dataset": "recent_form", "clean_rows": len(form), "file": "cleaned/recent_team_form_clean.csv"},
            {"dataset": "recent_scorers", "clean_rows": len(scorers), "file": "cleaned/recent_scorer_form_clean.csv"},
            {"dataset": "video_metadata_relevant", "clean_rows": len(videos), "file": "cleaned/video_metadata_relevant_clean.csv"},
            {"dataset": "rankings", "clean_rows": len(rankings), "file": "cleaned/fifa_rankings_wc_teams_clean.csv"},
        ]
    )
    write_csv(summary, DATA_DIR / "cleaning_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
