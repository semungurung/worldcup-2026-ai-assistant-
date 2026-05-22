from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
CLEAN_DIR = DATA_DIR / "cleaned"

ROUNDS = [
    "group_win",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
    "winner",
]


def read_best(clean_name: str, fallback: Path) -> pd.DataFrame:
    clean_path = CLEAN_DIR / clean_name
    if clean_path.exists():
        return pd.read_csv(clean_path)
    if fallback.exists():
        return pd.read_csv(fallback)
    return pd.DataFrame()


def logistic_win_probability(strength_a: float, strength_b: float) -> float:
    return 1 / (1 + math.exp(-(strength_a - strength_b) / 8.5))


def sample_knockout_winner(team_a: str, team_b: str, strength: dict[str, float], rng: random.Random) -> str:
    probability_a = logistic_win_probability(strength[team_a], strength[team_b])
    return team_a if rng.random() < probability_a else team_b


def sample_group_match(team_a: str, team_b: str, strength: dict[str, float], rng: random.Random) -> tuple[int, int, int, int]:
    diff = strength[team_a] - strength[team_b]
    draw_probability = max(0.16, 0.30 - min(abs(diff), 24) * 0.006)
    win_a_probability = (1 - draw_probability) * logistic_win_probability(strength[team_a], strength[team_b])
    roll = rng.random()
    if roll < win_a_probability:
        return 3, 0, max(1, round(1 + abs(diff) / 18)), 0
    if roll < win_a_probability + draw_probability:
        goals = rng.choice([0, 1, 1, 2])
        return 1, 1, goals, goals
    return 0, 3, 0, max(1, round(1 + abs(diff) / 18))


def build_group_fixtures(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, group_teams in teams.groupby("group"):
        names = group_teams["team"].tolist()
        for i, team_a in enumerate(names):
            for team_b in names[i + 1 :]:
                rows.append({"group": group, "team_a": team_a, "team_b": team_b})
    return pd.DataFrame(rows)


def rank_group(points: dict[str, int], goal_diff: dict[str, int], goals_for: dict[str, int], teams: list[str], strength: dict[str, float], rng: random.Random) -> list[str]:
    return sorted(
        teams,
        key=lambda team: (points[team], goal_diff[team], goals_for[team], strength[team], rng.random()),
        reverse=True,
    )


def build_knockout_seed_pool(group_rankings: dict[str, list[str]], points: dict[str, int], goal_diff: dict[str, int], strength: dict[str, float], rng: random.Random) -> list[str]:
    qualifiers = []
    thirds = []
    for group in sorted(group_rankings):
        ranked = group_rankings[group]
        qualifiers.extend(ranked[:2])
        thirds.append(ranked[2])
    best_thirds = sorted(thirds, key=lambda team: (points[team], goal_diff[team], strength[team], rng.random()), reverse=True)[:8]
    qualifiers.extend(best_thirds)
    return sorted(qualifiers, key=lambda team: (points[team], goal_diff[team], strength[team], rng.random()), reverse=True)


def pair_knockout_round(seed_pool: list[str]) -> list[tuple[str, str]]:
    return [(seed_pool[i], seed_pool[-(i + 1)]) for i in range(len(seed_pool) // 2)]


def simulate_tournament(iterations: int = 3000, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    teams = read_best("teams_clean.csv", DATA_DIR / "teams.csv")
    team_features = pd.read_csv(DERIVED_DIR / "team_feature_store.csv")
    strength_column = "model_strength_score" if "model_strength_score" in team_features.columns else "strength"
    strength = team_features.set_index("team")[strength_column].astype(float).to_dict()
    confidence = team_features.set_index("team")["data_confidence_0_1"].astype(float).to_dict()
    fixtures = build_group_fixtures(teams)

    counts = {round_name: Counter() for round_name in ROUNDS}
    semi_sets = Counter()
    final_sets = Counter()

    for _ in range(iterations):
        points = defaultdict(int)
        goal_diff = defaultdict(int)
        goals_for = defaultdict(int)

        for match in fixtures.itertuples():
            pts_a, pts_b, gf_a, gf_b = sample_group_match(match.team_a, match.team_b, strength, rng)
            points[match.team_a] += pts_a
            points[match.team_b] += pts_b
            goals_for[match.team_a] += gf_a
            goals_for[match.team_b] += gf_b
            goal_diff[match.team_a] += gf_a - gf_b
            goal_diff[match.team_b] += gf_b - gf_a

        group_rankings = {}
        for group, group_teams in teams.groupby("group"):
            ranked = rank_group(points, goal_diff, goals_for, group_teams["team"].tolist(), strength, rng)
            group_rankings[group] = ranked
            counts["group_win"][ranked[0]] += 1

        round_32 = build_knockout_seed_pool(group_rankings, points, goal_diff, strength, rng)
        for team in round_32:
            counts["round_of_32"][team] += 1

        current = round_32
        round_sequence = ["round_of_16", "quarter_final", "semi_final", "final", "winner"]
        for round_name in round_sequence:
            winners = []
            for team_a, team_b in pair_knockout_round(current):
                winners.append(sample_knockout_winner(team_a, team_b, strength, rng))
            for team in winners:
                counts[round_name][team] += 1
            if round_name == "semi_final":
                semi_sets[tuple(sorted(winners))] += 1
            if round_name == "final":
                final_sets[tuple(sorted(winners))] += 1
            current = winners

    rows = []
    for team in teams["team"]:
        row = {
            "team": team,
            "group": teams.set_index("team").loc[team, "group"],
            "model_strength_score": round(strength[team], 3),
            "data_confidence_0_1": round(confidence.get(team, 0), 3),
        }
        for round_name in ROUNDS:
            row[f"{round_name}_probability"] = counts[round_name][team] / iterations
        rows.append(row)

    probabilities = pd.DataFrame(rows).sort_values(["winner_probability", "semi_final_probability"], ascending=False)

    scenario_rows = []
    for scenario, count in semi_sets.most_common(12):
        scenario_rows.append({"scenario_type": "semi_finalists", "teams": " | ".join(scenario), "probability": count / iterations})
    for scenario, count in final_sets.most_common(12):
        scenario_rows.append({"scenario_type": "finalists", "teams": " | ".join(scenario), "probability": count / iterations})
    scenarios = pd.DataFrame(scenario_rows)
    return probabilities, scenarios


def build_prediction_summary(probabilities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column, count in [
        ("Most likely semi-finalists", "semi_final_probability", 4),
        ("Most likely finalists", "final_probability", 2),
        ("Most likely winner", "winner_probability", 1),
        ("Dark horses", "semi_final_probability", 8),
    ]:
        frame = probabilities.sort_values(column, ascending=False)
        if label == "Dark horses":
            frame = frame[(frame["model_strength_score"] < frame["model_strength_score"].quantile(0.72)) & (frame[column] > 0)]
        selected = frame.head(count)
        rows.append(
            {
                "prediction": label,
                "teams": " | ".join(selected["team"].tolist()),
                "basis": column,
                "top_probability": selected[column].max() if not selected.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    probabilities, scenarios = simulate_tournament()
    summary = build_prediction_summary(probabilities)
    probabilities.to_csv(DERIVED_DIR / "tournament_round_probabilities.csv", index=False)
    scenarios.to_csv(DERIVED_DIR / "tournament_scenarios.csv", index=False)
    summary.to_csv(DERIVED_DIR / "prediction_summary.csv", index=False)
    print(
        pd.DataFrame(
            [
                {"file": "derived/tournament_round_probabilities.csv", "rows": len(probabilities), "columns": len(probabilities.columns)},
                {"file": "derived/tournament_scenarios.csv", "rows": len(scenarios), "columns": len(scenarios.columns)},
                {"file": "derived/prediction_summary.csv", "rows": len(summary), "columns": len(summary.columns)},
            ]
        ).to_string(index=False)
    )
    print(probabilities[["team", "semi_final_probability", "final_probability", "winner_probability"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
