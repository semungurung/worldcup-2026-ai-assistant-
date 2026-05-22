from __future__ import annotations

import pandas as pd


def answer_team_question(team: str, teams: pd.DataFrame, players: pd.DataFrame) -> str:
    lookup = teams.set_index("team")
    if team not in lookup.index:
        return f"I do not have {team} in the current team dataset."

    row = lookup.loc[team]
    squad = players[players["team"] == team] if not players.empty else pd.DataFrame()
    fitness_flags = 0 if squad.empty else int((squad["fitness_status"] != "available").sum())
    player_note = "No squad rows loaded yet." if squad.empty else f"{len(squad)} player row(s), {fitness_flags} fitness flag(s)."

    return (
        f"{team} are in Group {row['group']} with a current strength score of {row['strength']:.1f}. "
        f"Their tactical tag is {row['tactical_style']}. {player_note}"
    )
