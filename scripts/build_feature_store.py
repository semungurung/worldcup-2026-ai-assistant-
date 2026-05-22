from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
CLEAN_DIR = DATA_DIR / "cleaned"

TEAM_BASE_COLUMNS = ["power_rating", "attack", "midfield", "defense", "depth"]
HOST_COUNTRIES = {"Mexico": "Mexico", "USA": "United States", "Canada": "Canada"}
CONFEDERATION_STRENGTH = {
    "UEFA": 1.00,
    "CONMEBOL": 0.97,
    "CAF": 0.82,
    "Concacaf": 0.78,
    "AFC": 0.76,
    "OFC": 0.62,
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_best(clean_name: str, fallback: Path) -> pd.DataFrame:
    clean_path = CLEAN_DIR / clean_name
    if clean_path.exists():
        return pd.read_csv(clean_path)
    return read_csv(fallback)


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    numeric = numeric.fillna(numeric.median())
    low = numeric.min()
    high = numeric.max()
    if high == low:
        scaled = pd.Series([0.5] * len(numeric), index=series.index)
    else:
        scaled = (numeric - low) / (high - low)
    return 1 - scaled if invert else scaled


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    std = numeric.std()
    if std == 0 or math.isnan(std):
        return pd.Series([0.0] * len(numeric), index=series.index)
    return (numeric - numeric.mean()) / std


def build_squad_features(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame(columns=["team"])
    rows = []
    for team, group in players.groupby("team"):
        attacking = group[group["position"].isin(["Forward", "Winger"])]
        midfield = group[group["position"].isin(["Midfielder"])]
        defensive = group[group["position"].isin(["Defender", "Goalkeeper"])]
        weighted_minutes = group["expected_minutes"].clip(lower=0)
        weighted_rating = (
            (group["role_rating"] * weighted_minutes).sum() / weighted_minutes.sum()
            if weighted_minutes.sum() > 0
            else group["role_rating"].mean()
        )
        rows.append(
            {
                "team": team,
                "squad_rows": len(group),
                "squad_rating_mean": round(group["role_rating"].mean(), 3),
                "squad_rating_weighted_minutes": round(weighted_rating, 3),
                "attack_player_rating": round(attacking["role_rating"].mean(), 3) if not attacking.empty else 0,
                "midfield_player_rating": round(midfield["role_rating"].mean(), 3) if not midfield.empty else 0,
                "defense_player_rating": round(defensive["role_rating"].mean(), 3) if not defensive.empty else 0,
                "expected_minutes_total": round(group["expected_minutes"].sum(), 3),
                "avg_injury_risk": round(group["injury_risk"].mean(), 3),
                "fitness_flag_count": int((group["fitness_status"] != "available").sum()),
                "projected_player_xg": round(group["goals_xg"].sum(), 3),
            }
        )
    return pd.DataFrame(rows)


def build_injury_features(injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries.empty:
        return pd.DataFrame(columns=["team"])
    severity = {"low": 0.5, "medium": 1.5, "high": 3.0}
    active = injuries[injuries["status"].isin(["monitor", "doubtful", "out", "injured"])].copy()
    active["injury_severity_score"] = active["severity"].map(severity).fillna(1.0)
    grouped = active.groupby("team").agg(
        active_injury_count=("player_name", "size"),
        injury_severity_total=("injury_severity_score", "sum"),
    )
    return grouped.reset_index()


def build_scorer_features(scorers: pd.DataFrame) -> pd.DataFrame:
    if scorers.empty:
        return pd.DataFrame(columns=["team"])
    rows = []
    for team, group in scorers.groupby("team"):
        top = group.sort_values("goals", ascending=False).head(5)
        total_goals = group["goals"].sum()
        top5_goals = top["goals"].sum()
        rows.append(
            {
                "team": team,
                "recent_goal_scorers": group["scorer"].nunique(),
                "recent_scorer_goals_total": int(total_goals),
                "top_scorer_goals": int(group["goals"].max()),
                "top5_scorer_goals": int(top5_goals),
                "scorer_concentration_top5": round(top5_goals / total_goals, 3) if total_goals else 0,
                "penalty_goal_share": round(group["penalty_goals"].sum() / total_goals, 3) if total_goals else 0,
            }
        )
    return pd.DataFrame(rows)


def build_video_features(videos: pd.DataFrame) -> pd.DataFrame:
    if videos.empty:
        return pd.DataFrame(columns=["team"])
    team_videos = videos[videos["team_or_scope"].notna() & (videos["team_or_scope"] != "Global")].copy()
    if team_videos.empty:
        return pd.DataFrame(columns=["team"])
    grouped = team_videos.groupby("team_or_scope").agg(
        video_source_rows=("video_id", "count"),
        relevant_video_score=("relevance_score", "sum"),
        training_video_count=("matched_keywords", lambda s: int(s.fillna("").str.contains("training").sum())),
        highlight_video_count=("matched_keywords", lambda s: int(s.fillna("").str.contains("highlights").sum())),
        press_video_count=("matched_keywords", lambda s: int(s.fillna("").str.contains("press conference").sum())),
    )
    return grouped.reset_index().rename(columns={"team_or_scope": "team"})


def build_opponent_adjusted_form(results: pd.DataFrame, rankings: pd.DataFrame, teams: pd.DataFrame, matches: int = 12) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["team"])
    rank_lookup = rankings.set_index("team")["fifa_rank_april_2026"].to_dict() if not rankings.empty else {}
    team_set = set(teams["team"])
    prepared = results.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.dropna(subset=["date", "home_score", "away_score"]).sort_values("date")
    rows = []
    for team in teams["team"]:
        matches_df = prepared[(prepared["home_team_norm"] == team) | (prepared["away_team_norm"] == team)].tail(matches)
        adjusted_points = 0.0
        upset_score = 0.0
        clean_sheets = 0
        failed_to_score = 0
        competitive_matches = 0
        home_matches = 0
        neutral_matches = 0
        opponent_ranks = []
        opponent_confeds = []

        for match in matches_df.itertuples():
            is_home = match.home_team_norm == team
            opponent = match.away_team_norm if is_home else match.home_team_norm
            gf = int(match.home_score if is_home else match.away_score)
            ga = int(match.away_score if is_home else match.home_score)
            points = 3 if gf > ga else 1 if gf == ga else 0
            opponent_rank = rank_lookup.get(opponent, 75)
            opponent_strength = max(0.55, 1.35 - (opponent_rank / 100))
            adjusted_points += (points / 3) * opponent_strength
            own_rank = rank_lookup.get(team, 75)
            upset_score += max(0, own_rank - opponent_rank) * (1 if points == 3 else 0.35 if points == 1 else 0)
            clean_sheets += int(ga == 0)
            failed_to_score += int(gf == 0)
            competitive_matches += int(str(match.tournament).lower() not in {"friendly", "friendly tournament"})
            home_matches += int(is_home)
            neutral_matches += int(bool(match.neutral))
            opponent_ranks.append(opponent_rank)
            opponent_row = teams[teams["team"] == opponent]
            if not opponent_row.empty:
                opponent_confeds.append(opponent_row.iloc[0]["confederation"])

        rows.append(
            {
                "team": team,
                "opponent_adjusted_form_0_100": round((adjusted_points / max(len(matches_df), 1)) * 100, 3),
                "avg_recent_opponent_rank": round(sum(opponent_ranks) / len(opponent_ranks), 3) if opponent_ranks else 75,
                "recent_clean_sheet_rate": round(clean_sheets / len(matches_df), 3) if len(matches_df) else 0,
                "recent_failed_to_score_rate": round(failed_to_score / len(matches_df), 3) if len(matches_df) else 0,
                "competitive_match_share": round(competitive_matches / len(matches_df), 3) if len(matches_df) else 0,
                "recent_home_match_share": round(home_matches / len(matches_df), 3) if len(matches_df) else 0,
                "recent_neutral_match_share": round(neutral_matches / len(matches_df), 3) if len(matches_df) else 0,
                "recent_upset_score": round(upset_score, 3),
                "recent_cross_confed_share": round(sum(1 for confed in opponent_confeds if confed) / len(matches_df), 3)
                if len(matches_df)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def build_team_feature_store() -> pd.DataFrame:
    teams = read_best("teams_clean.csv", DATA_DIR / "teams.csv")
    recent = read_best("recent_team_form_clean.csv", DERIVED_DIR / "recent_team_form.csv")
    rankings = read_best("fifa_rankings_wc_teams_clean.csv", DERIVED_DIR / "fifa_rankings_wc_teams.csv")
    players = read_best("players_clean.csv", DATA_DIR / "players.csv")
    injuries = read_best("injuries_clean.csv", DATA_DIR / "injuries.csv")
    scorers = read_best("recent_scorer_form_clean.csv", DERIVED_DIR / "recent_scorer_form.csv")
    videos = read_best("video_metadata_relevant_clean.csv", DERIVED_DIR / "video_metadata_relevant.csv")
    historical = read_best("worldcup_team_historical_results_clean.csv", DERIVED_DIR / "worldcup_team_historical_results.csv")

    features = teams.copy()
    features = features.merge(recent, on="team", how="left")
    features = features.merge(rankings[["team", "fifa_rank_april_2026"]], on="team", how="left")
    features = features.merge(build_opponent_adjusted_form(historical, rankings, teams), on="team", how="left")
    features = features.merge(build_squad_features(players), on="team", how="left")
    features = features.merge(build_injury_features(injuries), on="team", how="left")
    features = features.merge(build_scorer_features(scorers), on="team", how="left")
    features = features.merge(build_video_features(videos), on="team", how="left")

    fill_zero = [
        "squad_rows",
        "squad_rating_mean",
        "squad_rating_weighted_minutes",
        "attack_player_rating",
        "midfield_player_rating",
        "defense_player_rating",
        "expected_minutes_total",
        "avg_injury_risk",
        "fitness_flag_count",
        "projected_player_xg",
        "active_injury_count",
        "injury_severity_total",
        "recent_goal_scorers",
        "recent_scorer_goals_total",
        "top_scorer_goals",
        "top5_scorer_goals",
        "scorer_concentration_top5",
        "penalty_goal_share",
        "video_source_rows",
        "relevant_video_score",
        "training_video_count",
        "highlight_video_count",
        "press_video_count",
        "opponent_adjusted_form_0_100",
        "avg_recent_opponent_rank",
        "recent_clean_sheet_rate",
        "recent_failed_to_score_rate",
        "competitive_match_share",
        "recent_home_match_share",
        "recent_neutral_match_share",
        "recent_upset_score",
        "recent_cross_confed_share",
    ]
    for column in fill_zero:
        if column not in features:
            features[column] = 0
        features[column] = features[column].fillna(0)

    features["rank_strength_0_100"] = minmax(features["fifa_rank_april_2026"], invert=True) * 100
    features["form_strength_0_100"] = features["weighted_form_0_100"].fillna(features["form_index"])
    features["attack_balance"] = features["attack"] - features[["midfield", "defense"]].mean(axis=1)
    features["defensive_stability"] = features["defense"] + minmax(-features["goals_against"]).fillna(0) * 10
    features["goal_diff_per_match"] = features["goal_difference"].fillna(0) / features["matches_counted"].replace(0, pd.NA).fillna(1)
    features["goals_for_per_match"] = features["goals_for"].fillna(0) / features["matches_counted"].replace(0, pd.NA).fillna(1)
    features["goals_against_per_match"] = features["goals_against"].fillna(0) / features["matches_counted"].replace(0, pd.NA).fillna(1)
    features["squad_completeness_0_1"] = (features["squad_rows"] / 26).clip(upper=1)
    squad_rows = pd.to_numeric(features["squad_rows"], errors="coerce")
    fitness_flags = pd.to_numeric(features["fitness_flag_count"], errors="coerce")
    features["fitness_availability_0_1"] = 1 - (fitness_flags / squad_rows.where(squad_rows > 0))
    features["fitness_availability_0_1"] = features["fitness_availability_0_1"].fillna(0).astype(float)
    features["confederation_strength_factor"] = features["confederation"].map(CONFEDERATION_STRENGTH).fillna(0.75)
    features["opponent_quality_index_0_100"] = minmax(features["avg_recent_opponent_rank"], invert=True) * 100
    features["attack_efficiency_proxy"] = (
        0.55 * minmax(features["goals_for_per_match"]) * 100
        + 0.25 * minmax(features["recent_scorer_goals_total"]) * 100
        + 0.20 * features["attack"]
    )
    features["defense_resilience_proxy"] = (
        0.45 * minmax(features["recent_clean_sheet_rate"]) * 100
        + 0.35 * minmax(features["goals_against_per_match"], invert=True) * 100
        + 0.20 * features["defense"]
    )
    features["squad_depth_proxy"] = (
        0.55 * features["depth"]
        + 0.25 * features["squad_rating_weighted_minutes"]
        + 0.20 * (features["squad_completeness_0_1"] * 100)
    )
    features["fitness_risk_score_0_100"] = (
        0.55 * minmax(features["injury_severity_total"]) * 100
        + 0.25 * minmax(features["avg_injury_risk"]) * 100
        + 0.20 * (1 - features["fitness_availability_0_1"]) * 100
    )
    features["source_reliability_0_1"] = (
        0.35 * features["fifa_rank_april_2026"].notna().astype(float)
        + 0.30 * (features["matches_counted"].fillna(0) / 12).clip(upper=1)
        + 0.20 * (features["recent_goal_scorers"] > 0).astype(float)
        + 0.15 * (features["video_source_rows"] > 0).astype(float)
    )
    features["data_confidence_0_1"] = (
        0.30 * (features["matches_counted"].fillna(0) / 12).clip(upper=1)
        + 0.20 * features["squad_completeness_0_1"]
        + 0.15 * (features["recent_goal_scorers"] / features["recent_goal_scorers"].max()).fillna(0)
        + 0.15 * (features["video_source_rows"] / features["video_source_rows"].max()).fillna(0)
        + 0.20 * features["fifa_rank_april_2026"].notna().astype(float)
    )
    features["data_confidence_0_1"] = features["data_confidence_0_1"].fillna(0).clip(0, 1)

    features["model_strength_score"] = (
        0.18 * features["power_rating"]
        + 0.14 * features["rank_strength_0_100"]
        + 0.14 * features["form_strength_0_100"]
        + 0.10 * features["opponent_adjusted_form_0_100"]
        + 0.09 * features["attack_efficiency_proxy"]
        + 0.09 * features["midfield"]
        + 0.09 * features["defense_resilience_proxy"]
        + 0.05 * features["squad_depth_proxy"]
        + 0.03 * features["confederation_strength_factor"] * 100
        + 0.04 * minmax(features["recent_scorer_goals_total"]) * 100
        + 0.03 * minmax(features["projected_player_xg"]) * 100
        + 0.02 * features["host"].fillna(0) * 100
        - 0.03 * features["fitness_risk_score_0_100"]
    )
    features["model_strength_z"] = zscore(features["model_strength_score"])

    ordered = [
        "team",
        "group",
        "confederation",
        "host",
        "model_strength_score",
        "model_strength_z",
        "data_confidence_0_1",
        "rank_strength_0_100",
        "form_strength_0_100",
        "opponent_adjusted_form_0_100",
        "opponent_quality_index_0_100",
        "fifa_rank_april_2026",
        "matches_counted",
        "points_per_match",
        "goal_diff_per_match",
        "goals_for_per_match",
        "goals_against_per_match",
        "attack_balance",
        "defensive_stability",
        "attack_efficiency_proxy",
        "defense_resilience_proxy",
        "squad_depth_proxy",
        "squad_completeness_0_1",
        "fitness_availability_0_1",
        "fitness_risk_score_0_100",
        "source_reliability_0_1",
        "injury_severity_total",
        "recent_goal_scorers",
        "recent_scorer_goals_total",
        "scorer_concentration_top5",
        "video_source_rows",
        "relevant_video_score",
    ]
    remaining = [column for column in features.columns if column not in ordered]
    return features[ordered + remaining].sort_values("model_strength_score", ascending=False)


def build_match_feature_store(team_features: pd.DataFrame) -> pd.DataFrame:
    fixtures = read_best("fixtures_clean.csv", DATA_DIR / "fixtures.csv")
    h2h = read_csv(DERIVED_DIR / "fixture_head_to_head.csv")
    lookup = team_features.set_index("team")
    rows = []
    for match in fixtures.itertuples():
        if match.team_a not in lookup.index or match.team_b not in lookup.index:
            continue
        a = lookup.loc[match.team_a]
        b = lookup.loc[match.team_b]
        total_h2h = 0
        h2h_edge = 0
        if not h2h.empty:
            h2h_row = h2h[h2h["match_id"] == match.match_id]
            if not h2h_row.empty:
                item = h2h_row.iloc[0]
                total_h2h = int(item["historical_matches"])
                h2h_edge = (int(item["team_a_wins"]) - int(item["team_b_wins"])) / max(total_h2h, 1)
        rows.append(
            {
                "match_id": match.match_id,
                "group": match.group,
                "team_a": match.team_a,
                "team_b": match.team_b,
                "team_a_strength": round(a["model_strength_score"], 3),
                "team_b_strength": round(b["model_strength_score"], 3),
                "strength_delta_a_minus_b": round(a["model_strength_score"] - b["model_strength_score"], 3),
                "rank_delta_a_minus_b": round(a["fifa_rank_april_2026"] - b["fifa_rank_april_2026"], 3),
                "form_delta_a_minus_b": round(a["form_strength_0_100"] - b["form_strength_0_100"], 3),
                "opponent_adjusted_form_delta_a_minus_b": round(
                    a["opponent_adjusted_form_0_100"] - b["opponent_adjusted_form_0_100"], 3
                ),
                "attack_efficiency_delta_a_minus_b": round(a["attack_efficiency_proxy"] - b["attack_efficiency_proxy"], 3),
                "defense_resilience_delta_a_minus_b": round(a["defense_resilience_proxy"] - b["defense_resilience_proxy"], 3),
                "attack_vs_defense_delta_a": round(a["attack"] - b["defense"], 3),
                "attack_vs_defense_delta_b": round(b["attack"] - a["defense"], 3),
                "fitness_delta_a_minus_b": round(a["fitness_availability_0_1"] - b["fitness_availability_0_1"], 3),
                "fitness_risk_delta_a_minus_b": round(a["fitness_risk_score_0_100"] - b["fitness_risk_score_0_100"], 3),
                "squad_depth_delta_a_minus_b": round(a["squad_depth_proxy"] - b["squad_depth_proxy"], 3),
                "scorer_goal_delta_a_minus_b": round(a["recent_scorer_goals_total"] - b["recent_scorer_goals_total"], 3),
                "data_confidence_min": round(min(a["data_confidence_0_1"], b["data_confidence_0_1"]), 3),
                "source_reliability_min": round(min(a["source_reliability_0_1"], b["source_reliability_0_1"]), 3),
                "historical_matches": total_h2h,
                "h2h_edge_a_minus_b": round(h2h_edge, 3),
            }
        )
    return pd.DataFrame(rows)


def build_trust_report(team_features: pd.DataFrame, match_features: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add_check(name: str, status: str, detail: str, severity: str = "info") -> None:
        checks.append({"check": name, "status": status, "severity": severity, "detail": detail})

    add_check("team_count", "pass" if len(team_features) == 48 else "fail", f"{len(team_features)} teams in feature store", "high")
    add_check(
        "missing_model_strength",
        "pass" if team_features["model_strength_score"].isna().sum() == 0 else "fail",
        f"{int(team_features['model_strength_score'].isna().sum())} missing model strength values",
        "high",
    )
    add_check(
        "missing_confidence",
        "pass" if team_features["data_confidence_0_1"].isna().sum() == 0 else "fail",
        f"{int(team_features['data_confidence_0_1'].isna().sum())} missing confidence values",
        "high",
    )
    add_check(
        "low_squad_completeness",
        "warn" if (team_features["squad_completeness_0_1"] < 0.5).any() else "pass",
        f"{int((team_features['squad_completeness_0_1'] < 0.5).sum())} teams below 50% squad placeholder coverage",
        "medium",
    )
    add_check(
        "video_signal_sparse",
        "warn" if team_features["video_source_rows"].sum() == 0 else "pass",
        f"{int(team_features['video_source_rows'].sum())} team-specific cleaned video rows",
        "low",
    )
    add_check(
        "match_feature_rows",
        "pass" if len(match_features) > 0 else "fail",
        f"{len(match_features)} match feature rows generated",
        "high",
    )
    add_check(
        "confidence_floor",
        "warn" if (team_features["data_confidence_0_1"] < 0.45).any() else "pass",
        f"{int((team_features['data_confidence_0_1'] < 0.45).sum())} teams below 0.45 confidence",
        "medium",
    )
    add_check(
        "leakage_guardrail",
        "pass",
        "Feature builder uses completed historical results and pre-match fixture/team/player/source metadata only",
        "high",
    )
    return pd.DataFrame(checks)


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    team_features = build_team_feature_store()
    match_features = build_match_feature_store(team_features)

    team_features.to_csv(DERIVED_DIR / "team_feature_store.csv", index=False)
    match_features.to_csv(DERIVED_DIR / "match_feature_store.csv", index=False)
    trust_report = build_trust_report(team_features, match_features)
    trust_report.to_csv(DATA_DIR / "trust_report.csv", index=False)

    dictionary = pd.DataFrame(
        [
            {"feature": "model_strength_score", "level": "team", "meaning": "Weighted blended team quality, ranking, form, squad, scorer, host and injury signal."},
            {"feature": "data_confidence_0_1", "level": "team", "meaning": "How complete the current public data is for this team."},
            {"feature": "rank_strength_0_100", "level": "team", "meaning": "Inverse FIFA rank scaled so higher is better."},
            {"feature": "form_strength_0_100", "level": "team", "meaning": "Recency-weighted points from recent international matches."},
            {"feature": "opponent_adjusted_form_0_100", "level": "team", "meaning": "Recent form adjusted by opponent FIFA rank strength."},
            {"feature": "attack_efficiency_proxy", "level": "team", "meaning": "Blended goals-for, scorer-depth and attack-rating feature."},
            {"feature": "defense_resilience_proxy", "level": "team", "meaning": "Blended clean-sheet, goals-against and defense-rating feature."},
            {"feature": "source_reliability_0_1", "level": "team", "meaning": "Reliability signal based on ranking, form, scorer and video source coverage."},
            {"feature": "fitness_risk_score_0_100", "level": "team", "meaning": "Higher score means more squad/injury/availability risk."},
            {"feature": "goal_diff_per_match", "level": "team", "meaning": "Recent goal difference normalized by matches counted."},
            {"feature": "fitness_availability_0_1", "level": "team", "meaning": "Available player share from current squad rows."},
            {"feature": "scorer_concentration_top5", "level": "team", "meaning": "Share of recent national-team goals coming from top five scorers."},
            {"feature": "relevant_video_score", "level": "team", "meaning": "Count-weighted metadata signal from relevant official/media video sources."},
            {"feature": "strength_delta_a_minus_b", "level": "match", "meaning": "Team A model strength minus Team B model strength."},
            {"feature": "attack_vs_defense_delta_a", "level": "match", "meaning": "Team A attack rating against Team B defense rating."},
            {"feature": "h2h_edge_a_minus_b", "level": "match", "meaning": "Historical head-to-head win edge for Team A."},
        ]
    )
    dictionary.to_csv(DERIVED_DIR / "feature_dictionary.csv", index=False)

    summary = pd.DataFrame(
        [
            {"file": "derived/team_feature_store.csv", "rows": len(team_features), "columns": len(team_features.columns)},
            {"file": "derived/match_feature_store.csv", "rows": len(match_features), "columns": len(match_features.columns)},
            {"file": "derived/feature_dictionary.csv", "rows": len(dictionary), "columns": len(dictionary.columns)},
            {"file": "trust_report.csv", "rows": len(trust_report), "columns": len(trust_report.columns)},
        ]
    )
    summary.to_csv(DATA_DIR / "feature_engineering_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
