from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DERIVED_DIR = DATA_DIR / "derived"
CLEAN_DIR = DATA_DIR / "cleaned"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_derived_csv(name: str) -> pd.DataFrame:
    path = DERIVED_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_clean_or_default(clean_name: str, default_path: Path) -> pd.DataFrame:
    clean_path = CLEAN_DIR / clean_name
    if clean_path.exists():
        return pd.read_csv(clean_path)
    if default_path.exists():
        return pd.read_csv(default_path)
    return pd.DataFrame()


def load_teams() -> pd.DataFrame:
    teams = load_clean_or_default("teams_clean.csv", DATA_DIR / "teams.csv")
    required = {"team", "group", "power_rating", "form_index", "attack", "midfield", "defense", "depth"}
    missing = required - set(teams.columns)
    if missing:
        raise ValueError(f"teams.csv is missing required columns: {sorted(missing)}")
    return teams


def load_players() -> pd.DataFrame:
    return load_clean_or_default("players_clean.csv", DATA_DIR / "players.csv")


def load_injuries() -> pd.DataFrame:
    return load_clean_or_default("injuries_clean.csv", DATA_DIR / "injuries.csv")


def load_events() -> pd.DataFrame:
    return load_clean_or_default("match_events_clean.csv", DATA_DIR / "match_events.csv")


def load_recent_form() -> pd.DataFrame:
    return load_clean_or_default("recent_team_form_clean.csv", DERIVED_DIR / "recent_team_form.csv")


def load_rankings() -> pd.DataFrame:
    return load_clean_or_default("fifa_rankings_wc_teams_clean.csv", DERIVED_DIR / "fifa_rankings_wc_teams.csv")


def load_scorer_form() -> pd.DataFrame:
    return load_clean_or_default("recent_scorer_form_clean.csv", DERIVED_DIR / "recent_scorer_form.csv")


def load_ingestion_summary() -> pd.DataFrame:
    return load_csv("ingestion_summary.csv")


def load_video_metadata() -> pd.DataFrame:
    return load_clean_or_default("video_metadata_relevant_clean.csv", DERIVED_DIR / "video_metadata_relevant.csv")


def load_video_channels() -> pd.DataFrame:
    return load_derived_csv("video_channel_index.csv")


def load_video_policy() -> pd.DataFrame:
    return load_csv("video_platform_policy.csv")


def load_kaggle_video_catalog() -> pd.DataFrame:
    return load_csv("kaggle_video_dataset_catalog.csv")


def load_video_analysis_design() -> pd.DataFrame:
    return load_csv("video_analysis_design.csv")


def load_video_ingestion_summary() -> pd.DataFrame:
    return load_csv("video_ingestion_summary.csv")


def load_team_feature_store() -> pd.DataFrame:
    return load_derived_csv("team_feature_store.csv")


def load_match_feature_store() -> pd.DataFrame:
    return load_derived_csv("match_feature_store.csv")


def load_feature_dictionary() -> pd.DataFrame:
    return load_derived_csv("feature_dictionary.csv")


def load_feature_engineering_summary() -> pd.DataFrame:
    return load_csv("feature_engineering_summary.csv")


def load_cleaning_summary() -> pd.DataFrame:
    return load_csv("cleaning_summary.csv")


def load_trust_report() -> pd.DataFrame:
    return load_csv("trust_report.csv")


def load_validation_report() -> pd.DataFrame:
    return load_csv("validation_report.csv")


def load_tournament_round_probabilities() -> pd.DataFrame:
    return load_derived_csv("tournament_round_probabilities.csv")


def load_tournament_scenarios() -> pd.DataFrame:
    return load_derived_csv("tournament_scenarios.csv")


def load_prediction_summary() -> pd.DataFrame:
    return load_derived_csv("prediction_summary.csv")


def load_fixtures(teams: pd.DataFrame | None = None) -> pd.DataFrame:
    fixtures = load_csv("fixtures.csv")
    if teams is None:
        teams = load_teams()

    generated = []
    for group, group_teams in teams.groupby("group"):
        names = group_teams["team"].tolist()
        for index, (team_a, team_b) in enumerate(combinations(names, 2), start=1):
            generated.append(
                {
                    "match_id": f"{group}{index}",
                    "date": "",
                    "group": group,
                    "team_a": team_a,
                    "team_b": team_b,
                    "venue": "",
                }
            )

    full = pd.DataFrame(generated)
    if fixtures.empty:
        return full

    keyed = full.set_index(["group", "team_a", "team_b"])
    known = fixtures.set_index(["group", "team_a", "team_b"])
    keyed.update(known)
    return keyed.reset_index()
