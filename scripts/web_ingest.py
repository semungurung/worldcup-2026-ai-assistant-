from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "derived"

USER_AGENT = "worldcup-2026-ai-assistant/0.1 (+local research prototype)"

MARTJ42_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"

SOURCES = [
    {
        "dataset": "historical_results",
        "source_name": "martj42 international_results results.csv",
        "url": f"{MARTJ42_BASE}/results.csv",
        "license_or_terms": "CC0-1.0 / public-domain style dataset",
        "notes": "Strictly men's full international results; used for recent form, head-to-head and Elo-style signals.",
    },
    {
        "dataset": "historical_goalscorers",
        "source_name": "martj42 international_results goalscorers.csv",
        "url": f"{MARTJ42_BASE}/goalscorers.csv",
        "license_or_terms": "CC0-1.0 / public-domain style dataset",
        "notes": "Goal-scorer rows used for national-team scorer trend features where available.",
    },
    {
        "dataset": "historical_shootouts",
        "source_name": "martj42 international_results shootouts.csv",
        "url": f"{MARTJ42_BASE}/shootouts.csv",
        "license_or_terms": "CC0-1.0 / public-domain style dataset",
        "notes": "Penalty-shootout outcomes for knockout-model feature engineering.",
    },
    {
        "dataset": "fifa_rankings_reference",
        "source_name": "FIFA Men's World Ranking / ESPN April 2026 summary",
        "url": "https://www.espn.com/soccer/story/_/id/46664763/fifa-mens-top-50-world-rankings",
        "license_or_terms": "Reference only; ranks are facts, page text is not redistributed.",
        "notes": "April 1 2026 rank positions for World Cup-qualified teams; points pulled where public table provides them.",
    },
]


TEAM_ALIASES = {
    "United States": "USA",
    "US": "USA",
    "U.S.": "USA",
    "South Korea": "Korea Republic",
    "Korea, Republic of": "Korea Republic",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Cape Verde": "Cabo Verde",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "Iran": "IR Iran",
    "Türkiye": "Turkiye",
    "Turkey": "Turkiye",
}

FALLBACK_RANKS_APRIL_2026 = {
    "France": 1,
    "Spain": 2,
    "Argentina": 3,
    "England": 4,
    "Portugal": 5,
    "Brazil": 6,
    "Netherlands": 7,
    "Morocco": 8,
    "Belgium": 9,
    "Germany": 10,
    "Croatia": 11,
    "Colombia": 13,
    "Senegal": 14,
    "Mexico": 15,
    "USA": 16,
    "Uruguay": 17,
    "Japan": 18,
    "Switzerland": 19,
    "IR Iran": 21,
    "Turkiye": 22,
    "Ecuador": 23,
    "Austria": 24,
    "Korea Republic": 25,
    "Australia": 27,
    "Algeria": 28,
    "Egypt": 29,
    "Canada": 30,
    "Norway": 31,
    "Panama": 33,
    "Cote d'Ivoire": 34,
    "Sweden": 38,
    "Paraguay": 40,
    "Czechia": 41,
    "Scotland": 43,
    "Tunisia": 44,
    "Congo DR": 46,
    "Uzbekistan": 50,
    "Qatar": 55,
    "Iraq": 57,
    "South Africa": 60,
    "Saudi Arabia": 61,
    "Jordan": 63,
    "Bosnia and Herzegovina": 65,
    "Cabo Verde": 69,
    "Ghana": 74,
    "Curacao": 82,
    "Haiti": 83,
    "New Zealand": 85,
}


@dataclass
class DownloadedCsv:
    dataset: str
    path: Path
    rows: int


def normalize_team(name: str) -> str:
    clean = str(name).strip()
    return TEAM_ALIASES.get(clean, clean)


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def download_csv(url: str, output: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    text = fetch_text(url)
    frame = pd.read_csv(StringIO(text), parse_dates=parse_dates)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def load_or_download_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[DownloadedCsv]]:
    downloaded = []
    results = download_csv(f"{MARTJ42_BASE}/results.csv", RAW_DIR / "international_results.csv", ["date"])
    downloaded.append(DownloadedCsv("historical_results", RAW_DIR / "international_results.csv", len(results)))

    goalscorers = download_csv(f"{MARTJ42_BASE}/goalscorers.csv", RAW_DIR / "international_goalscorers.csv", ["date"])
    downloaded.append(DownloadedCsv("historical_goalscorers", RAW_DIR / "international_goalscorers.csv", len(goalscorers)))

    shootouts = download_csv(f"{MARTJ42_BASE}/shootouts.csv", RAW_DIR / "international_shootouts.csv", ["date"])
    downloaded.append(DownloadedCsv("historical_shootouts", RAW_DIR / "international_shootouts.csv", len(shootouts)))
    return results, goalscorers, shootouts, downloaded


def get_qualified_teams() -> pd.DataFrame:
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    teams["team"] = teams["team"].map(normalize_team)
    return teams


def add_normalized_names(results: pd.DataFrame) -> pd.DataFrame:
    prepared = results.copy()
    prepared = prepared.dropna(subset=["home_score", "away_score"]).copy()
    prepared = prepared[prepared["date"] <= pd.Timestamp(date.today())].copy()
    prepared["home_score"] = prepared["home_score"].astype(int)
    prepared["away_score"] = prepared["away_score"].astype(int)
    prepared["home_team_norm"] = prepared["home_team"].map(normalize_team)
    prepared["away_team_norm"] = prepared["away_team"].map(normalize_team)
    prepared["home_points"] = prepared.apply(
        lambda row: 3 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 0,
        axis=1,
    )
    prepared["away_points"] = prepared.apply(
        lambda row: 3 if row.away_score > row.home_score else 1 if row.home_score == row.away_score else 0,
        axis=1,
    )
    prepared["home_goal_diff"] = prepared["home_score"] - prepared["away_score"]
    prepared["away_goal_diff"] = prepared["away_score"] - prepared["home_score"]
    return prepared


def build_recent_form(results: pd.DataFrame, teams: pd.DataFrame, matches: int = 12) -> pd.DataFrame:
    rows = []
    results = results.sort_values("date")
    for team in teams["team"]:
        team_matches = results[(results["home_team_norm"] == team) | (results["away_team_norm"] == team)].tail(matches)
        points = 0
        goals_for = 0
        goals_against = 0
        wins = 0
        draws = 0
        losses = 0
        weighted_score = 0.0
        weight_total = 0.0

        for age, match in enumerate(reversed(list(team_matches.itertuples())), start=1):
            is_home = match.home_team_norm == team
            earned = match.home_points if is_home else match.away_points
            gf = match.home_score if is_home else match.away_score
            ga = match.away_score if is_home else match.home_score
            weight = math.exp(-(age - 1) / 8)
            points += earned
            goals_for += gf
            goals_against += ga
            wins += int(earned == 3)
            draws += int(earned == 1)
            losses += int(earned == 0)
            weighted_score += (earned / 3) * weight
            weight_total += weight

        rows.append(
            {
                "team": team,
                "matches_counted": len(team_matches),
                "last_match_date": team_matches["date"].max().date().isoformat() if not team_matches.empty else "",
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "points": points,
                "points_per_match": round(points / len(team_matches), 3) if len(team_matches) else 0,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_difference": goals_for - goals_against,
                "weighted_form_0_100": round((weighted_score / weight_total) * 100, 2) if weight_total else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_form_0_100", ascending=False)


def build_head_to_head(results: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match in fixtures.itertuples():
        team_a = normalize_team(match.team_a)
        team_b = normalize_team(match.team_b)
        h2h = results[
            ((results["home_team_norm"] == team_a) & (results["away_team_norm"] == team_b))
            | ((results["home_team_norm"] == team_b) & (results["away_team_norm"] == team_a))
        ].sort_values("date")
        team_a_wins = 0
        team_b_wins = 0
        draws = 0
        for row in h2h.itertuples():
            if row.home_score == row.away_score:
                draws += 1
            elif (row.home_team_norm == team_a and row.home_score > row.away_score) or (
                row.away_team_norm == team_a and row.away_score > row.home_score
            ):
                team_a_wins += 1
            else:
                team_b_wins += 1
        rows.append(
            {
                "match_id": match.match_id,
                "group": match.group,
                "team_a": team_a,
                "team_b": team_b,
                "historical_matches": len(h2h),
                "team_a_wins": team_a_wins,
                "draws": draws,
                "team_b_wins": team_b_wins,
                "last_played": h2h["date"].max().date().isoformat() if not h2h.empty else "",
            }
        )
    return pd.DataFrame(rows)


def build_scorer_form(goalscorers: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if goalscorers.empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp(date(2022, 1, 1))
    prepared = goalscorers[goalscorers["date"] >= cutoff].copy()
    if "own_goal" in prepared:
        prepared = prepared[~prepared["own_goal"].fillna(False).astype(bool)].copy()
    prepared["team_norm"] = prepared["team"].map(normalize_team)
    prepared = prepared[prepared["team_norm"].isin(set(teams["team"]))]
    if prepared.empty:
        return pd.DataFrame()
    grouped = (
        prepared.groupby(["team_norm", "scorer"])
        .agg(
            goals=("scorer", "size"),
            penalty_goals=("penalty", "sum"),
            own_goals=("own_goal", "sum"),
            first_goal_date=("date", "min"),
            last_goal_date=("date", "max"),
        )
        .reset_index()
        .rename(columns={"team_norm": "team"})
    )
    grouped["first_goal_date"] = grouped["first_goal_date"].dt.date.astype(str)
    grouped["last_goal_date"] = grouped["last_goal_date"].dt.date.astype(str)
    return grouped.sort_values(["goals", "last_goal_date"], ascending=[False, False])


def build_rankings(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team in teams["team"]:
        rank = FALLBACK_RANKS_APRIL_2026.get(team)
        rows.append(
            {
                "team": team,
                "fifa_rank_april_2026": rank,
                "ranking_source": "FIFA/ESPN April 2026 reference",
                "ranking_note": "Use official FIFA points when API/table access is added" if rank else "Rank missing; verify manually",
            }
        )
    return pd.DataFrame(rows).sort_values("fifa_rank_april_2026")


def build_model_resource_index(downloaded: list[DownloadedCsv]) -> pd.DataFrame:
    source_rows = []
    by_dataset = {item.dataset: item for item in downloaded}
    for source in SOURCES:
        item = by_dataset.get(source["dataset"])
        source_rows.append(
            {
                **source,
                "local_path": str(item.path.relative_to(ROOT)) if item else "",
                "rows_saved": item.rows if item else "",
                "retrieved_on": date.today().isoformat(),
            }
        )
    return pd.DataFrame(source_rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    teams = get_qualified_teams()
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    results, goalscorers, shootouts, downloaded = load_or_download_raw()
    normalized_results = add_normalized_names(results)

    wc_team_results = normalized_results[
        normalized_results["home_team_norm"].isin(set(teams["team"]))
        | normalized_results["away_team_norm"].isin(set(teams["team"]))
    ].copy()
    wc_team_results.to_csv(DERIVED_DIR / "worldcup_team_historical_results.csv", index=False)

    recent_form = build_recent_form(normalized_results, teams)
    recent_form.to_csv(DERIVED_DIR / "recent_team_form.csv", index=False)

    h2h = build_head_to_head(normalized_results, fixtures)
    h2h.to_csv(DERIVED_DIR / "fixture_head_to_head.csv", index=False)

    scorer_form = build_scorer_form(goalscorers, teams)
    scorer_form.to_csv(DERIVED_DIR / "recent_scorer_form.csv", index=False)

    rankings = build_rankings(teams)
    rankings.to_csv(DERIVED_DIR / "fifa_rankings_wc_teams.csv", index=False)

    enriched = teams.merge(recent_form[["team", "weighted_form_0_100", "points_per_match"]], on="team", how="left").merge(
        rankings[["team", "fifa_rank_april_2026"]], on="team", how="left"
    )
    enriched.to_csv(DERIVED_DIR / "teams_enriched.csv", index=False)

    source_index = build_model_resource_index(downloaded)
    source_index.to_csv(DATA_DIR / "data_sources.csv", index=False)

    summary = pd.DataFrame(
        [
            {"file": "raw/international_results.csv", "rows": len(results)},
            {"file": "raw/international_goalscorers.csv", "rows": len(goalscorers)},
            {"file": "raw/international_shootouts.csv", "rows": len(shootouts)},
            {"file": "derived/worldcup_team_historical_results.csv", "rows": len(wc_team_results)},
            {"file": "derived/recent_team_form.csv", "rows": len(recent_form)},
            {"file": "derived/fixture_head_to_head.csv", "rows": len(h2h)},
            {"file": "derived/recent_scorer_form.csv", "rows": len(scorer_form)},
            {"file": "derived/fifa_rankings_wc_teams.csv", "rows": len(rankings)},
            {"file": "derived/teams_enriched.csv", "rows": len(enriched)},
            {"file": "data_sources.csv", "rows": len(source_index)},
        ]
    )
    summary.to_csv(DATA_DIR / "ingestion_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except URLError as exc:
        raise SystemExit(f"Network error while retrieving public data: {exc}") from exc
