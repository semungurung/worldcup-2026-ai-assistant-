from __future__ import annotations

import random
from collections import Counter, defaultdict

import pandas as pd

from .prediction_model import predict_match


def _sample_result(probabilities: dict[str, float | str], rng: random.Random) -> tuple[int, int]:
    roll = rng.random()
    if roll < float(probabilities["win_a"]):
        return 3, 0
    if roll < float(probabilities["win_a"]) + float(probabilities["draw"]):
        return 1, 1
    return 0, 3


def simulate_group_stage(
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    iterations: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    advance_counts = Counter()
    group_win_counts = Counter()

    for _ in range(iterations):
        points = defaultdict(int)
        rating_tiebreak = teams.set_index("team")["strength"].to_dict()

        for _, match in fixtures.iterrows():
            prediction = predict_match(match["team_a"], match["team_b"], teams)
            pts_a, pts_b = _sample_result(prediction, rng)
            points[match["team_a"]] += pts_a
            points[match["team_b"]] += pts_b

        third_places = []
        for group, group_teams in teams.groupby("group"):
            ranked = sorted(
                group_teams["team"],
                key=lambda name: (points[name], rating_tiebreak[name], rng.random()),
                reverse=True,
            )
            group_win_counts[ranked[0]] += 1
            for name in ranked[:2]:
                advance_counts[name] += 1
            third_places.append(ranked[2])

        best_thirds = sorted(
            third_places,
            key=lambda name: (points[name], rating_tiebreak[name], rng.random()),
            reverse=True,
        )[:8]
        for name in best_thirds:
            advance_counts[name] += 1

    rows = []
    for _, row in teams.iterrows():
        rows.append(
            {
                "team": row["team"],
                "group": row["group"],
                "strength": round(row["strength"], 1),
                "group_win_probability": group_win_counts[row["team"]] / iterations,
                "advance_probability": advance_counts[row["team"]] / iterations,
            }
        )
    return pd.DataFrame(rows).sort_values(["advance_probability", "strength"], ascending=False)


def tournament_winner_projection(group_results: pd.DataFrame) -> pd.DataFrame:
    projection = group_results.copy()
    projection["winner_score"] = projection["advance_probability"] * (projection["strength"] ** 2)
    total = projection["winner_score"].sum()
    projection["winner_probability"] = projection["winner_score"] / total
    return projection.sort_values("winner_probability", ascending=False)[
        ["team", "group", "winner_probability", "advance_probability", "strength"]
    ]
