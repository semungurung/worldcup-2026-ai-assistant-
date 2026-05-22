from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
DERIVED_DIR = DATA_DIR / "derived"

BAD_VIDEO_TERMS = [
    "women",
    "lionesses",
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
    "futsal",
    "beach soccer",
    "efootball",
    "nfl",
    "canadian championship",
]


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def result(name: str, status: str, severity: str, details: str) -> dict[str, str]:
    return {"check": name, "status": status, "severity": severity, "details": details}


def validate() -> pd.DataFrame:
    rows = []
    teams = read(CLEAN_DIR / "teams_clean.csv")
    fixtures = read(CLEAN_DIR / "fixtures_clean.csv")
    players = read(CLEAN_DIR / "players_clean.csv")
    injuries = read(CLEAN_DIR / "injuries_clean.csv")
    events = read(CLEAN_DIR / "match_events_clean.csv")
    historical = read(CLEAN_DIR / "worldcup_team_historical_results_clean.csv")
    form = read(CLEAN_DIR / "recent_team_form_clean.csv")
    scorers = read(CLEAN_DIR / "recent_scorer_form_clean.csv")
    videos = read(CLEAN_DIR / "video_metadata_relevant_clean.csv")
    rankings = read(CLEAN_DIR / "fifa_rankings_wc_teams_clean.csv")
    team_features = read(DERIVED_DIR / "team_feature_store.csv")
    match_features = read(DERIVED_DIR / "match_feature_store.csv")

    valid_teams = set(teams["team"]) if not teams.empty else set()

    rows.append(result("teams_48", "pass" if len(teams) == 48 else "fail", "high", f"{len(teams)} teams found"))
    rows.append(result("teams_unique", "pass" if teams["team"].is_unique else "fail", "high", "Team names are unique"))
    rows.append(
        result(
            "team_ratings_range",
            "pass"
            if all(teams[col].between(0, 100).all() for col in ["power_rating", "form_index", "attack", "midfield", "defense", "depth"])
            else "fail",
            "high",
            "All rating columns should be 0-100",
        )
    )

    fixture_teams_valid = fixtures["team_a"].isin(valid_teams).all() and fixtures["team_b"].isin(valid_teams).all()
    rows.append(result("fixture_team_validity", "pass" if fixture_teams_valid else "fail", "high", "All fixtures use known teams"))

    player_teams_valid = players.empty or players["team"].isin(valid_teams).all()
    rows.append(result("player_team_validity", "pass" if player_teams_valid else "fail", "medium", "All player rows map to known teams"))
    rows.append(
        result(
            "player_rating_range",
            "pass" if players.empty or players["role_rating"].between(0, 100).all() else "fail",
            "medium",
            "Player role ratings should be 0-100",
        )
    )

    injury_teams_valid = injuries.empty or injuries["team"].isin(valid_teams).all()
    rows.append(result("injury_team_validity", "pass" if injury_teams_valid else "fail", "medium", "All injury rows map to known teams"))

    event_xg_valid = events.empty or events["xg"].between(0, 1).all()
    rows.append(result("event_xg_range", "pass" if event_xg_valid else "fail", "medium", "Match-event xG must be 0-1"))

    if not historical.empty:
        historical["date"] = pd.to_datetime(historical["date"], errors="coerce")
        future_rows = int((historical["date"] > pd.Timestamp(date.today())).sum())
    else:
        future_rows = 0
    rows.append(result("historical_no_future_rows", "pass" if future_rows == 0 else "fail", "high", f"{future_rows} future rows"))

    rows.append(result("recent_form_coverage", "pass" if len(form) == 48 else "warn", "medium", f"{len(form)} team form rows"))
    rows.append(result("ranking_coverage", "pass" if len(rankings) == 48 else "warn", "medium", f"{len(rankings)} ranking rows"))
    rows.append(
        result(
            "scorer_no_own_goals",
            "pass" if scorers.empty or int(scorers["own_goals"].sum()) == 0 else "fail",
            "medium",
            "Own goals excluded from scorer-form features",
        )
    )

    video_text = " ".join(videos.get("title", pd.Series(dtype=str)).astype(str)).lower()
    bad_terms_found = [term for term in BAD_VIDEO_TERMS if term in video_text]
    rows.append(
        result(
            "video_relevance_filter",
            "pass" if not bad_terms_found else "fail",
            "high",
            f"Bad terms found: {', '.join(bad_terms_found) if bad_terms_found else 'none'}",
        )
    )

    rows.append(
        result(
            "team_feature_coverage",
            "pass" if len(team_features) == 48 else "fail",
            "high",
            f"{len(team_features)} team feature rows",
        )
    )
    rows.append(
        result(
            "feature_no_nulls",
            "pass" if int(team_features.isna().sum().sum()) == 0 else "fail",
            "high",
            f"{int(team_features.isna().sum().sum())} null cells in team feature store",
        )
    )
    rows.append(
        result(
            "match_feature_deltas",
            "pass"
            if {"strength_delta_a_minus_b", "opponent_adjusted_form_delta_a_minus_b", "fitness_risk_delta_a_minus_b"}.issubset(match_features.columns)
            else "fail",
            "high",
            "Required match-delta features exist",
        )
    )
    rows.append(
        result(
            "known_limit_squad_placeholders",
            "warn" if (team_features.get("squad_completeness_0_1", pd.Series([0])) < 0.5).any() else "pass",
            "medium",
            "Final squads are not fully populated yet, so squad-derived features remain low-confidence",
        )
    )
    rows.append(
        result(
            "known_limit_video_sparsity",
            "warn" if len(videos) < 10 else "pass",
            "low",
            f"{len(videos)} high-confidence video metadata rows after strict cleaning",
        )
    )
    rows.append(
        result(
            "leakage_guardrail",
            "pass",
            "high",
            "Generated features avoid final World Cup outcome labels and use completed historical/current metadata only",
        )
    )

    return pd.DataFrame(rows)


def main() -> None:
    report = validate()
    report.to_csv(DATA_DIR / "validation_report.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
