from __future__ import annotations

import math

import pandas as pd


def _draw_probability(diff: float) -> float:
    return max(0.16, 0.30 - min(abs(diff), 24) * 0.006)


def predict_match(team_a: str, team_b: str, teams: pd.DataFrame) -> dict[str, float | str]:
    lookup = teams.set_index("team")
    if team_a not in lookup.index or team_b not in lookup.index:
        raise ValueError("Both teams must exist in teams.csv")

    strength_a = float(lookup.loc[team_a, "strength"])
    strength_b = float(lookup.loc[team_b, "strength"])
    diff = strength_a - strength_b
    non_draw = 1 - _draw_probability(diff)
    win_a = non_draw / (1 + math.exp(-diff / 7.5))
    draw = _draw_probability(diff)
    win_b = 1 - win_a - draw

    return {
        "team_a": team_a,
        "team_b": team_b,
        "win_a": round(win_a, 3),
        "draw": round(draw, 3),
        "win_b": round(win_b, 3),
        "expected_goal_diff": round(diff / 12, 2),
        "explanation": explain_match(team_a, team_b, strength_a, strength_b),
    }


def explain_match(team_a: str, team_b: str, strength_a: float, strength_b: float) -> str:
    diff = strength_a - strength_b
    if abs(diff) < 2:
        edge = "profiles as a narrow, low-margin matchup"
    elif diff > 0:
        edge = f"leans toward {team_a} because of the stronger blended rating"
    else:
        edge = f"leans toward {team_b} because of the stronger blended rating"
    return f"{team_a} vs {team_b} {edge}. Current model gap: {diff:.1f} rating points."
