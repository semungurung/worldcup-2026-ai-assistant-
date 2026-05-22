from __future__ import annotations

import pandas as pd


EVENT_WEIGHTS = {
    "goal": 5,
    "big_chance": 4,
    "red_card": 4,
    "penalty": 4,
    "shot": 2,
    "yellow_card": 1,
    "substitution": 1,
}


def top_moments(events: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["minute", "team", "event_type", "description", "importance"])
    scored = events.copy()
    scored["importance"] = scored["event_type"].map(EVENT_WEIGHTS).fillna(1) + scored.get("xg", 0).fillna(0) * 3
    return scored.sort_values(["importance", "minute"], ascending=[False, True]).head(limit)


def momentum_timeline(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["minute", "team", "momentum"])
    timeline = events.copy()
    timeline["momentum"] = timeline["event_type"].map(EVENT_WEIGHTS).fillna(1) + timeline.get("xg", 0).fillna(0)
    return timeline[["minute", "team", "momentum"]].sort_values("minute")


def generate_match_report(events: pd.DataFrame) -> str:
    if events.empty:
        return "No match events are loaded yet. Add rows to data/match_events.csv to generate analysis."

    goals = events[events["event_type"] == "goal"]
    chances = events[events["event_type"].isin(["big_chance", "shot", "penalty"])]
    cards = events[events["event_type"].isin(["yellow_card", "red_card"])]
    top = top_moments(events, 3)

    leader = events.groupby("team")["xg"].sum().sort_values(ascending=False)
    xg_line = ", ".join(f"{team}: {value:.2f} xG" for team, value in leader.items())
    moments = " ".join(f"{int(row.minute)}' {row.team} {row.event_type}: {row.description}" for row in top.itertuples())

    return (
        f"The event feed shows {len(goals)} goal(s), {len(chances)} attacking event(s), "
        f"and {len(cards)} card event(s). Expected-goal pressure: {xg_line}. "
        f"Key moments: {moments}"
    )
