from __future__ import annotations

import pandas as pd


WEIGHTS = {
    "power_rating": 0.30,
    "form_index": 0.18,
    "attack": 0.16,
    "midfield": 0.13,
    "defense": 0.15,
    "depth": 0.08,
}


def team_strength(row: pd.Series) -> float:
    return sum(float(row[column]) * weight for column, weight in WEIGHTS.items())


def add_team_features(teams: pd.DataFrame, injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    featured = teams.copy()
    featured["base_strength"] = featured.apply(team_strength, axis=1)
    featured["host_bonus"] = featured.get("host", 0).fillna(0).astype(float) * 1.8

    injury_penalties = {}
    if injuries is not None and not injuries.empty:
        severity_map = {"low": 0.4, "medium": 1.2, "high": 2.5}
        active = injuries[injuries["status"].isin(["monitor", "doubtful", "out"])]
        for team, rows in active.groupby("team"):
            injury_penalties[team] = rows["severity"].map(severity_map).fillna(0.8).sum()

    featured["injury_penalty"] = featured["team"].map(injury_penalties).fillna(0.0)
    featured["strength"] = featured["base_strength"] + featured["host_bonus"] - featured["injury_penalty"]
    return featured


def squad_summary(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame(columns=["team", "squad_rating", "available_players", "fitness_flags"])

    grouped = players.groupby("team").agg(
        squad_rating=("role_rating", "mean"),
        available_players=("fitness_status", lambda s: int((s == "available").sum())),
        fitness_flags=("fitness_status", lambda s: int((s != "available").sum())),
    )
    return grouped.reset_index()
