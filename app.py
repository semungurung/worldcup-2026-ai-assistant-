from __future__ import annotations

import html
import sys
import base64
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
ASSETS_DIR = ROOT / "assets"

from src.assistant import answer_team_question
from src.data_pipeline import (
    load_cleaning_summary,
    load_events,
    load_feature_dictionary,
    load_feature_engineering_summary,
    load_fixtures,
    load_ingestion_summary,
    load_injuries,
    load_kaggle_video_catalog,
    load_match_feature_store,
    load_players,
    load_prediction_summary,
    load_rankings,
    load_recent_form,
    load_scorer_form,
    load_teams,
    load_team_feature_store,
    load_tournament_round_probabilities,
    load_tournament_scenarios,
    load_trust_report,
    load_validation_report,
    load_video_analysis_design,
    load_video_channels,
    load_video_ingestion_summary,
    load_video_metadata,
    load_video_policy,
)
from src.features import add_team_features, squad_summary
from src.prediction_model import predict_match
from src.simulator import simulate_group_stage, tournament_winner_projection
from src.tactical_analyzer import generate_match_report, momentum_timeline, top_moments


st.set_page_config(page_title="World Cup 2026 AI Predictor", layout="wide")


def image_as_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


HERO_IMAGE_URI = image_as_data_uri(ASSETS_DIR / "worldcup-analytics-hero.png")


st.markdown(
    """
    <style>
    :root {
        --wc-bg: #08111d;
        --wc-panel: #ffffff;
        --wc-ink: #101827;
        --wc-muted: #536579;
        --wc-line: #d5dee8;
        --wc-blue: #1769c2;
        --wc-teal: #12a78f;
        --wc-red: #c84c52;
        --wc-gold: #d99a22;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(3, 9, 17, 0.94), rgba(5, 17, 30, 0.88) 42%, rgba(239, 244, 249, 0.96) 72%),
            var(--hero-image),
            linear-gradient(180deg, #07111e 0%, #eef3f7 100%);
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
        color: var(--wc-ink);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            radial-gradient(circle at 15% 8%, rgba(18, 167, 143, 0.20), transparent 26%),
            radial-gradient(circle at 78% 12%, rgba(217, 154, 34, 0.18), transparent 26%),
            linear-gradient(180deg, rgba(4, 12, 22, 0.45), rgba(239, 244, 249, 0.05) 38%, rgba(239, 244, 249, 0.78) 72%);
        z-index: 0;
    }
    .app-bg-image {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        object-position: center top;
        opacity: 0.72;
        filter: saturate(1.12) contrast(1.08);
        z-index: 0;
        pointer-events: none;
    }
    .block-container {
        padding-top: clamp(0.85rem, 1.8vw, 1.4rem);
        padding-bottom: 2.5rem;
        max-width: min(1480px, calc(100vw - 2rem));
        position: relative;
        z-index: 2;
    }
    [data-testid="stHeader"] {
        background: rgba(248, 251, 255, 0.86);
        backdrop-filter: blur(8px);
    }
    @media (min-width: 1100px) {
        .block-container {
            padding-left: 4.8rem;
        }
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: #101827;
    }
    h4, h5, h6, strong {
        color: #101827;
    }
    div[data-testid="stTabs"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(213, 222, 232, 0.92);
        border-radius: 8px;
        padding: 0.35rem 0.45rem 0 0.45rem;
        box-shadow: 0 16px 36px rgba(7, 20, 34, 0.12);
        overflow-x: auto;
        white-space: nowrap;
    }
    div[data-testid="stTabs"] button {
        border-radius: 8px;
        padding: 0.52rem 0.9rem;
        font-weight: 700;
        color: #26384b;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: #eaf4ff;
        color: #0b5599;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(213, 222, 232, 0.95);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 14px 34px rgba(8, 24, 42, 0.13);
    }
    div[data-testid="stMetric"] label {
        color: var(--wc-muted);
        font-weight: 700;
    }
    .hero {
        border: 1px solid #cfdbe6;
        border-radius: 8px;
        padding: 1.65rem 1.8rem;
        margin-bottom: 1.05rem;
        background:
            linear-gradient(90deg, rgba(2, 8, 14, 0.98) 0%, rgba(4, 18, 31, 0.94) 48%, rgba(4, 18, 31, 0.70) 100%),
            var(--hero-image),
            linear-gradient(135deg, #07111e, #123452);
        background-size: cover;
        background-position: center;
        min-height: clamp(240px, 34vw, 320px);
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        box-shadow: 0 22px 52px rgba(4, 14, 25, 0.30);
    }
    .hero-title {
        font-size: clamp(1.55rem, 3.2vw, 3rem);
        line-height: 1.04;
        font-weight: 850;
        color: #ffffff;
        margin: 0 0 0.45rem 0;
        max-width: 780px;
        text-shadow: 0 3px 16px rgba(0, 0, 0, 0.55);
    }
    .hero-subtitle {
        max-width: 860px;
        color: rgba(255, 255, 255, 0.86);
        font-size: clamp(0.9rem, 1.4vw, 1rem);
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
    }
    .hero-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border: 1px solid #cfdbe6;
        border-radius: 999px;
        padding: 0.32rem 0.62rem;
        background: rgba(255, 255, 255, 0.98);
        color: #314356;
        font-size: 0.82rem;
        font-weight: 750;
    }
    .hero .pill {
        border-color: rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.14);
        color: #ffffff;
        backdrop-filter: blur(8px);
    }
    .section-card {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid var(--wc-line);
        border-radius: 8px;
        padding: clamp(0.8rem, 1.4vw, 1rem);
        box-shadow: 0 8px 24px rgba(31, 45, 61, 0.05);
        margin-bottom: 0.85rem;
    }
    .section-card.visual-section {
        color: #ffffff;
        border-color: rgba(255, 255, 255, 0.34);
        background:
            linear-gradient(90deg, rgba(4, 14, 25, 0.96), rgba(8, 31, 52, 0.88)),
            linear-gradient(135deg, #0a2137, #0d675d);
        background-size: cover;
        background-position: center;
        box-shadow: 0 16px 38px rgba(6, 23, 39, 0.24);
    }
    .section-card.visual-section .section-title {
        color: #ffffff;
    }
    .section-card.visual-section .section-note {
        color: rgba(255, 255, 255, 0.88);
    }
    .section-title {
        font-size: 1rem;
        font-weight: 820;
        color: #17202a;
        margin: 0 0 0.2rem 0;
    }
    .section-note {
        color: var(--wc-muted);
        font-size: 0.86rem;
        margin: 0;
    }
    .status-pass {
        color: #0d6b45;
        background: #e5f6ee;
        border-color: #b9e4cd;
    }
    .status-warn {
        color: #855a08;
        background: #fff4d8;
        border-color: #f2d48a;
    }
    .status-fail {
        color: #9a1f28;
        background: #fde4e6;
        border-color: #f1b9bd;
    }
    .insight-box {
        border-left: 4px solid var(--wc-blue);
        background: #f7fbff;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        color: #24384b;
        margin: 0.5rem 0 1rem 0;
    }
    .visual-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.75rem;
        margin: 0.4rem 0 1rem 0;
    }
    .visual-tile {
        border: 1px solid rgba(255,255,255,0.52);
        border-radius: 8px;
        min-height: 124px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.99), rgba(241, 248, 255, 0.96));
        padding: 0.9rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(8, 24, 42, 0.15);
    }
    .visual-tile::after {
        content: "";
        position: absolute;
        inset: auto -22px -42px auto;
        width: 112px;
        height: 112px;
        border-radius: 50%;
        border: 18px solid rgba(15, 94, 168, 0.08);
    }
    .tile-label {
        color: var(--wc-muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 820;
        margin-bottom: 0.2rem;
    }
    .tile-value {
        color: #142231;
        font-size: clamp(1.28rem, 2.5vw, 1.65rem);
        font-weight: 850;
        line-height: 1;
    }
    .tile-note {
        color: #607080;
        font-size: 0.82rem;
        margin-top: 0.45rem;
        max-width: 220px;
    }
    .pitch-card {
        border: 1px solid #bdd7cd;
        border-radius: 8px;
        min-height: clamp(170px, 26vw, 220px);
        background:
            linear-gradient(90deg, rgba(255,255,255,0.13) 49%, rgba(255,255,255,0.26) 50%, rgba(255,255,255,0.13) 51%),
            repeating-linear-gradient(90deg, #2f8f61 0 52px, #2a845a 52px 104px);
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.28);
    }
    .photo-panel {
        border: 1px solid rgba(255, 255, 255, 0.36);
        border-radius: 8px;
        min-height: clamp(150px, 24vw, 220px);
        padding: 1rem;
        margin: 0.5rem 0 1rem 0;
        color: #ffffff;
        background:
            linear-gradient(90deg, rgba(1, 7, 13, 0.98), rgba(3, 14, 25, 0.92), rgba(3, 14, 25, 0.72)),
            var(--hero-image),
            linear-gradient(135deg, #0a2137, #1769c2);
        background-size: cover;
        background-position: center;
        box-shadow: 0 18px 44px rgba(6, 20, 35, 0.26);
        display: flex;
        align-items: flex-end;
    }
    .photo-panel > div {
        background: rgba(12, 105, 67, 0.94);
        border: 1px solid rgba(188, 239, 210, 0.72);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        max-width: min(760px, 100%);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
    }
    .photo-panel strong {
        display: block;
        font-size: 1.55rem;
        line-height: 1.1;
        margin-bottom: 0.35rem;
        color: #ffffff !important;
        font-weight: 900;
        text-shadow: 0 3px 14px rgba(0, 0, 0, 0.72);
    }
    .photo-panel span {
        color: rgba(255,255,255,0.94) !important;
        font-weight: 650;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.68);
    }
    .stMarkdown, .stText, p, label, span {
        color: #1d2b3a;
    }
    .hero span, .hero p, .photo-panel span, .photo-panel strong,
    .section-card.visual-section p, .section-card.visual-section span,
    .section-card.visual-section strong, .pitch-overlay strong, .pitch-overlay span {
        color: #ffffff;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {
        color: #152235;
    }
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(10, 28, 46, 0.07);
        overflow-x: auto;
    }
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="columnheader"] span,
    div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
        color: #142231 !important;
        background: #eef5fb !important;
        font-weight: 800 !important;
    }
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="gridcell"] span {
        color: #1d2b3a !important;
    }
    div[data-testid="stDataFrame"] [role="row"]:first-child {
        background: #eef5fb !important;
    }
    [data-testid="stVegaLiteChart"] {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(213, 222, 232, 0.95);
        border-radius: 8px;
        padding: 0.75rem;
        box-shadow: 0 12px 30px rgba(8, 24, 42, 0.10);
    }
    [data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.98);
        color: #152235;
        border-radius: 8px;
    }
    [data-testid="stMarkdownContainer"] {
        color: #1d2b3a;
    }
    [data-testid="stMarkdownContainer"] p {
        color: #1d2b3a;
    }
    .hero [data-testid="stMarkdownContainer"] p,
    .photo-panel [data-testid="stMarkdownContainer"] p {
        color: #ffffff;
    }
    .pitch-card::before {
        content: "";
        position: absolute;
        inset: 18px;
        border: 2px solid rgba(255,255,255,0.62);
        border-radius: 6px;
    }
    .pitch-card::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 76px;
        height: 76px;
        transform: translate(-50%, -50%);
        border: 2px solid rgba(255,255,255,0.66);
        border-radius: 50%;
    }
    .pitch-overlay {
        position: relative;
        z-index: 2;
        padding: 1rem;
        color: #ffffff;
        text-shadow: 0 2px 8px rgba(0,0,0,0.28);
    }
    .pitch-overlay strong {
        display: block;
        font-size: 1.35rem;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: clamp(0.6rem, 1.5vw, 1rem);
    }
    @media (max-width: 1099px) {
        .block-container {
            max-width: min(100vw - 1.25rem, 1120px);
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
    }
    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        div[data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
        }
        .hero {
            min-height: 300px;
            padding: 1.15rem;
            background-position: center;
        }
        .hero-row {
            gap: 0.35rem;
        }
        .pill {
            font-size: 0.76rem;
            padding: 0.28rem 0.5rem;
        }
        .photo-panel {
            min-height: 150px;
        }
        .photo-panel strong {
            font-size: 1.22rem;
        }
    }
    @media (max-width: 620px) {
        .stApp {
            background-attachment: scroll;
        }
        [data-testid="stHeader"] {
            height: 2.4rem;
        }
        .app-bg-image {
            opacity: 0.45;
            height: 100%;
        }
        .block-container {
            max-width: 100vw;
            padding-left: 0.55rem;
            padding-right: 0.55rem;
            padding-top: 0.65rem;
        }
        .hero {
            min-height: 210px;
            padding: 0.85rem;
            margin-bottom: 0.75rem;
        }
        .hero-title {
            font-size: 1.42rem;
        }
        .hero-subtitle {
            font-size: 0.82rem;
        }
        .hero-row {
            display: none;
        }
        .section-card,
        .visual-tile,
        .photo-panel,
        .pitch-card {
            border-radius: 8px;
        }
        .section-card {
            padding: 0.72rem;
        }
        .section-title {
            font-size: 0.92rem;
        }
        .section-note {
            font-size: 0.78rem;
        }
        div[data-testid="stTabs"] {
            padding: 0.22rem 0.28rem 0 0.28rem;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
            flex-wrap: nowrap;
            gap: 0.1rem;
            scrollbar-width: thin;
        }
        div[data-testid="stTabs"] button {
            min-width: max-content;
            padding: 0.42rem 0.52rem;
            font-size: 0.78rem;
        }
        div[data-testid="stMetric"] {
            padding: 0.62rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
        .photo-panel {
            min-height: 130px;
            padding: 0.7rem;
        }
        .photo-panel > div {
            padding: 0.62rem 0.7rem;
        }
        .photo-panel strong {
            font-size: 1rem;
        }
        .photo-panel span {
            font-size: 0.78rem;
        }
        .comparison-grid,
        .bracket-grid,
        .visual-grid {
            grid-template-columns: 1fr;
            gap: 0.55rem;
        }
        .pitch-overlay strong {
            font-size: 1.05rem;
        }
        div[data-testid="stDataFrame"] {
            font-size: 0.78rem;
        }
    }
    .small-muted {
        color: var(--wc-muted);
        font-size: 0.82rem;
    }
    div[data-testid="stPopover"] {
        position: fixed;
        left: 0.8rem;
        top: 45%;
        z-index: 9999;
    }
    div[data-testid="stPopover"] > button {
        width: 3.25rem;
        height: 3.25rem;
        border-radius: 999px;
        background: #0c6943;
        color: #ffffff;
        border: 2px solid rgba(255, 255, 255, 0.92);
        box-shadow: 0 14px 36px rgba(3, 13, 23, 0.34);
        font-weight: 900;
        padding: 0;
    }
    div[data-testid="stPopover"] > button:hover {
        background: #0f7f52;
        color: #ffffff;
    }
    @media (max-width: 620px) {
        div[data-testid="stPopover"] {
            left: 0.55rem;
            bottom: 1rem;
            top: auto;
        }
        div[data-testid="stPopover"] > button {
            width: 3rem;
            height: 3rem;
        }
    }
    .comparison-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 1rem 0;
    }
    .compare-card {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(213, 222, 232, 0.95);
        border-radius: 8px;
        padding: 0.95rem;
        box-shadow: 0 12px 30px rgba(8, 24, 42, 0.10);
    }
    .compare-card h3 {
        margin: 0 0 0.35rem 0;
        font-size: 1.18rem;
    }
    .compare-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        border-top: 1px solid #e4ebf2;
        padding-top: 0.42rem;
        margin-top: 0.42rem;
        font-size: 0.86rem;
    }
    .bracket-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.75rem;
        margin: 0.6rem 0 1rem 0;
    }
    .bracket-col {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(213, 222, 232, 0.95);
        border-radius: 8px;
        padding: 0.8rem;
        box-shadow: 0 12px 30px rgba(8, 24, 42, 0.10);
    }
    .bracket-col strong {
        display: block;
        margin-bottom: 0.45rem;
        font-size: 0.86rem;
        text-transform: uppercase;
        color: #536579;
    }
    .bracket-team {
        display: flex;
        justify-content: space-between;
        gap: 0.55rem;
        border: 1px solid #e2ebf3;
        border-radius: 8px;
        padding: 0.42rem 0.5rem;
        margin-bottom: 0.38rem;
        background: #f8fbff;
        font-size: 0.84rem;
        font-weight: 750;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if HERO_IMAGE_URI:
    st.markdown(
        f"<style>:root {{ --hero-image: url('{HERO_IMAGE_URI}'); }}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<img class='app-bg-image' src='{HERO_IMAGE_URI}' alt='football stadium background'>", unsafe_allow_html=True)
else:
    st.markdown("<style>:root { --hero-image: none; }</style>", unsafe_allow_html=True)


def section_header(title: str, note: str | None = None) -> None:
    note_html = f"<p class='section-note'>{html.escape(note)}</p>" if note else ""
    st.markdown(
        f"<div class='section-card visual-section'><p class='section-title'>{html.escape(title)}</p>{note_html}</div>",
        unsafe_allow_html=True,
    )


def status_pill(label: str, status: str) -> str:
    css = {"pass": "status-pass", "warn": "status-warn", "fail": "status-fail"}.get(status, "")
    return f"<span class='pill {css}'>{html.escape(label)}: {html.escape(status.upper())}</span>"


def format_percent(value: float | int | str) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def visual_tile(label: str, value: str, note: str) -> str:
    return (
        "<div class='visual-tile'>"
        f"<div class='tile-label'>{html.escape(label)}</div>"
        f"<div class='tile-value'>{html.escape(value)}</div>"
        f"<div class='tile-note'>{html.escape(note)}</div>"
        "</div>"
    )


def probability_chart_frame(frame: pd.DataFrame, probability_column: str, top_n: int = 12) -> pd.DataFrame:
    if frame.empty or probability_column not in frame.columns:
        return pd.DataFrame()
    view = frame.sort_values(probability_column, ascending=False).head(top_n).copy()
    view["probability_percent"] = view[probability_column] * 100
    return view[["team", "probability_percent"]].set_index("team")


def team_feature_row(team: str) -> pd.Series | None:
    if team_feature_store.empty:
        return None
    rows = team_feature_store[team_feature_store["team"] == team]
    if rows.empty:
        return None
    return rows.iloc[0]


def round_probability_row(team: str) -> pd.Series | None:
    if round_probabilities.empty:
        return None
    rows = round_probabilities[round_probabilities["team"] == team]
    if rows.empty:
        return None
    return rows.iloc[0]


EXPLAINABLE_FEATURES = [
    ("rank_strength_0_100", "FIFA/rank strength", 0.22, "higher"),
    ("form_strength_0_100", "Recent form", 0.18, "higher"),
    ("opponent_adjusted_form_0_100", "Opponent-adjusted form", 0.16, "higher"),
    ("attack_efficiency_proxy", "Attack efficiency", 0.13, "higher"),
    ("defense_resilience_proxy", "Defensive resilience", 0.13, "higher"),
    ("squad_depth_proxy", "Squad depth", 0.08, "higher"),
    ("fitness_risk_score_0_100", "Fitness risk", -0.06, "lower"),
    ("source_reliability_0_1", "Source reliability", 4.0, "higher"),
]


def explain_team_features(team: str) -> pd.DataFrame:
    row = team_feature_row(team)
    if row is None:
        return pd.DataFrame()
    records = []
    for column, label, weight, direction in EXPLAINABLE_FEATURES:
        if column not in team_feature_store.columns:
            continue
        values = pd.to_numeric(team_feature_store[column], errors="coerce")
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value) or values.dropna().empty:
            continue
        baseline = float(values.mean())
        contribution = (float(value) - baseline) * weight
        records.append(
            {
                "feature": label,
                "raw_value": float(value),
                "baseline": baseline,
                "impact_points": contribution,
                "model_direction": direction,
            }
        )
    return pd.DataFrame(records).sort_values("impact_points", ascending=False)


def compare_teams_frame(team_a: str, team_b: str) -> pd.DataFrame:
    rows = team_feature_store[team_feature_store["team"].isin([team_a, team_b])] if not team_feature_store.empty else pd.DataFrame()
    if rows.empty:
        return pd.DataFrame()
    metrics = [
        ("model_strength_score", "Model strength"),
        ("winner_probability", "Winner probability"),
        ("semi_final_probability", "Semi-final probability"),
        ("opponent_adjusted_form_0_100", "Adjusted form"),
        ("attack_efficiency_proxy", "Attack efficiency"),
        ("defense_resilience_proxy", "Defensive resilience"),
        ("fitness_risk_score_0_100", "Fitness risk"),
        ("data_confidence_0_1", "Data confidence"),
    ]
    merged = rows.copy()
    if not round_probabilities.empty:
        merged = merged.merge(
            round_probabilities[["team", "winner_probability", "semi_final_probability"]],
            on="team",
            how="left",
        )
    output = []
    by_team = merged.set_index("team")
    for column, label in metrics:
        if column not in by_team.columns:
            continue
        value_a = pd.to_numeric(pd.Series([by_team.loc[team_a, column]]), errors="coerce").iloc[0] if team_a in by_team.index else None
        value_b = pd.to_numeric(pd.Series([by_team.loc[team_b, column]]), errors="coerce").iloc[0] if team_b in by_team.index else None
        if pd.isna(value_a) or pd.isna(value_b):
            continue
        output.append(
            {
                "metric": label,
                team_a: float(value_a),
                team_b: float(value_b),
                "edge": team_a if value_a > value_b else team_b if value_b > value_a else "Even",
                "gap": abs(float(value_a) - float(value_b)),
            }
        )
    return pd.DataFrame(output)


def bracket_html() -> str:
    if round_probabilities.empty:
        return "<div class='insight-box'>Run tournament predictions to populate the bracket view.</div>"
    rounds = [
        ("Round of 32", "round_of_32_probability", 12),
        ("Round of 16", "round_of_16_probability", 10),
        ("Quarter-finals", "quarter_final_probability", 8),
        ("Semi-finals", "semi_final_probability", 4),
        ("Final", "final_probability", 2),
        ("Champion", "winner_probability", 1),
    ]
    columns = []
    for label, column, count in rounds:
        teams_html = ""
        for row in round_probabilities.sort_values(column, ascending=False).head(count).itertuples():
            teams_html += (
                "<div class='bracket-team'>"
                f"<span>{html.escape(row.team)}</span>"
                f"<span>{getattr(row, column):.1%}</span>"
                "</div>"
            )
        columns.append(f"<div class='bracket-col'><strong>{html.escape(label)}</strong>{teams_html}</div>")
    return "<div class='bracket-grid'>" + "".join(columns) + "</div>"


def photo_panel(title: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="photo-panel">
            <div>
                <strong>{html.escape(title)}</strong>
                <span>{html.escape(note)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_teams(query: str, team_names: list[str]) -> list[str]:
    normalized = query.lower()
    found = []
    for team_name in team_names:
        pattern = r"(^|[^a-z])" + re.escape(team_name.lower()) + r"([^a-z]|$)"
        if re.search(pattern, normalized):
            found.append(team_name)
    aliases = {"usa": "USA", "us": "USA", "iran": "IR Iran", "south korea": "Korea Republic", "turkey": "Turkiye"}
    for alias, canonical in aliases.items():
        if alias in normalized and canonical in team_names and canonical not in found:
            found.append(canonical)
    return found[:2]


def answer_match_assistant(query: str) -> str:
    if not query.strip():
        return "Ask me about a team, a matchup, simulation runs, feature scores, injuries, or data trust."

    q = query.lower()
    team_names = sorted(teams["team"].dropna().unique().tolist(), key=len, reverse=True)
    found_teams = detect_teams(query, team_names)

    if any(term in q for term in ["semi", "semifinal", "semi-final", "semi finalists", "semi-finalists", "top 4", "last four"]):
        if not round_probabilities.empty:
            top = round_probabilities.sort_values("semi_final_probability", ascending=False).head(4)
            teams_text = ", ".join(f"{row.team} ({row.semi_final_probability:.1%})" for row in top.itertuples())
            return (
                f"Most likely 2026 semi-finalists from the current advanced projection: {teams_text}. "
                "This uses group-stage simulation plus an approximate knockout bracket seeded from group performance."
            )
        return "Round-by-round tournament probabilities are not loaded yet. Run `python3 scripts/build_tournament_predictions.py`."

    if any(term in q for term in ["finalist", "finalists", "final two", "reach final"]):
        if not round_probabilities.empty:
            top = round_probabilities.sort_values("final_probability", ascending=False).head(2)
            teams_text = ", ".join(f"{row.team} ({row.final_probability:.1%})" for row in top.itertuples())
            return f"Most likely finalists from the current projection: {teams_text}."
        return "Final probability data is not loaded yet."

    if any(term in q for term in ["winner", "champion", "win the world cup", "trophy"]):
        if not round_probabilities.empty:
            top = round_probabilities.sort_values("winner_probability", ascending=False).head(5)
            teams_text = ", ".join(f"{row.team} ({row.winner_probability:.1%})" for row in top.itertuples())
            return f"Current winner projection top five: {teams_text}."
        return "Winner probability data is not loaded yet."

    if any(term in q for term in ["round of 16", "quarter", "knockout", "bracket", "advance"]):
        if not round_probabilities.empty:
            if found_teams:
                row = round_probability_row(found_teams[0])
                if row is not None:
                    return (
                        f"{found_teams[0]} path probabilities: R32 {row['round_of_32_probability']:.1%}, "
                        f"R16 {row['round_of_16_probability']:.1%}, QF {row['quarter_final_probability']:.1%}, "
                        f"SF {row['semi_final_probability']:.1%}, Final {row['final_probability']:.1%}, "
                        f"Winner {row['winner_probability']:.1%}."
                    )
            top = round_probabilities.sort_values("round_of_16_probability", ascending=False).head(8)
            teams_text = ", ".join(f"{row.team} ({row.round_of_16_probability:.1%})" for row in top.itertuples())
            return f"Most likely Round of 16 teams in the current bracket projection: {teams_text}."
        return "Bracket probability data is not loaded yet."

    if any(term in q for term in ["dark horse", "dark horses", "surprise", "underdog"]):
        if not round_probabilities.empty:
            merged = round_probabilities.merge(
                team_feature_store[["team", "model_strength_score"]], on="team", how="left", suffixes=("", "_feature")
            )
            dark = merged[
                (merged["model_strength_score"] < merged["model_strength_score"].quantile(0.72))
                & (merged["semi_final_probability"] > 0.05)
            ].sort_values("semi_final_probability", ascending=False).head(5)
            teams_text = ", ".join(f"{row.team} ({row.semi_final_probability:.1%} semi-final)" for row in dark.itertuples())
            return f"Current dark-horse candidates: {teams_text or 'none above the threshold yet'}."
        return "Dark-horse projection data is not loaded yet."

    if any(word in q for word in ["simulation", "simulate", "runs"]):
        return (
            "Simulation runs are repeated tournament trials. More runs make probabilities more stable. "
            "This app simulates group-stage results using match probabilities, then counts how often teams advance or win."
        )

    if any(word in q for word in ["feature", "feature store", "confidence", "trust", "validation"]):
        fail_count = int((validation_report["status"] == "fail").sum()) if not validation_report.empty else 0
        warn_count = int((validation_report["status"] == "warn").sum()) if not validation_report.empty else 0
        return (
            f"The feature store is the model-ready table of cleaned inputs. Current validation: {fail_count} fail, "
            f"{warn_count} warning(s). Key warnings are incomplete final squads and sparse strict video metadata."
        )

    if any(word in q for word in ["why", "explain", "shap", "driver", "because"]):
        target = found_teams[0] if found_teams else None
        if target:
            explanation = explain_team_features(target).head(4)
            if not explanation.empty:
                drivers = ", ".join(
                    f"{row.feature} ({row.impact_points:+.1f})" for row in explanation.itertuples()
                )
                return (
                    f"{target}'s model score is mainly driven by: {drivers}. "
                    "These are SHAP-style additive impacts against the current tournament baseline, not a black-box SHAP library output."
                )
        return "Ask 'why Argentina' or 'explain France' and I will show the strongest model drivers."

    if any(word in q for word in ["injury", "injuries", "fitness"]):
        if found_teams and not team_feature_store.empty:
            row = team_feature_store[team_feature_store["team"] == found_teams[0]]
            if not row.empty:
                item = row.iloc[0]
                return (
                    f"{found_teams[0]} fitness signal: availability {format_percent(item['fitness_availability_0_1'])}, "
                    f"fitness risk {item['fitness_risk_score_0_100']:.1f}/100, injury severity total "
                    f"{item['injury_severity_total']:.1f}. This is still placeholder-sensitive until final squads are loaded."
                )
        return "Fitness/injury features use injury severity, squad availability, and player injury-risk placeholders."

    if len(found_teams) >= 2:
        team_a, team_b = found_teams[0], found_teams[1]
        prediction = predict_match(team_a, team_b, teams)
        comparison = compare_teams_frame(team_a, team_b)
        edge_note = ""
        if not comparison.empty:
            edges = comparison.sort_values("gap", ascending=False).head(3)
            edge_note = " Biggest model gaps: " + ", ".join(
                f"{row.metric} edge {row.edge} by {row.gap:.1f}" for row in edges.itertuples()
            ) + "."
        details = [
            f"{team_a} win {prediction['win_a']:.1%}",
            f"draw {prediction['draw']:.1%}",
            f"{team_b} win {prediction['win_b']:.1%}",
        ]
        feature_note = ""
        if not match_feature_store.empty:
            match_row = match_feature_store[
                ((match_feature_store["team_a"] == team_a) & (match_feature_store["team_b"] == team_b))
                | ((match_feature_store["team_a"] == team_b) & (match_feature_store["team_b"] == team_a))
            ]
            if not match_row.empty:
                row = match_row.iloc[0]
                feature_note = (
                    f" Feature gap: strength delta {row['strength_delta_a_minus_b']:.1f}, "
                    f"adjusted-form delta {row['opponent_adjusted_form_delta_a_minus_b']:.1f}, "
                    f"minimum data confidence {format_percent(row['data_confidence_min'])}."
                )
        return f"{team_a} vs {team_b}: " + ", ".join(details) + f". {prediction['explanation']}{feature_note}{edge_note}"

    if len(found_teams) == 1:
        team = found_teams[0]
        base = answer_team_question(team, teams, players)
        if not team_feature_store.empty:
            row = team_feature_store[team_feature_store["team"] == team]
            if not row.empty:
                item = row.iloc[0]
                return (
                    f"{base} Model strength {item['model_strength_score']:.1f}, "
                    f"opponent-adjusted form {item['opponent_adjusted_form_0_100']:.1f}, "
                    f"source reliability {format_percent(item['source_reliability_0_1'])}."
                )
        return base

    if any(word in q for word in ["moment", "event", "goal", "card", "report"]):
        return generate_match_report(events)

    if not round_probabilities.empty:
        top = round_probabilities.sort_values("winner_probability", ascending=False).head(3)
        leaders = ", ".join(f"{row.team} {row.winner_probability:.1%}" for row in top.itertuples())
        return (
            "I can answer winner, semi-finalist, bracket path, matchup, team comparison, explainable AI, fitness, "
            f"and data trust questions. Current winner leaders: {leaders}."
        )
    return "I can answer prediction, comparison, explainability, simulation, fitness, and data-trust questions."


def ensure_chat_history() -> None:
    if "match_chat_history" not in st.session_state:
        st.session_state.match_chat_history = [
            ("assistant", "Hi, I can explain match predictions, team features, simulations, injuries, and event reports.")
        ]


def render_chat_controls(key_prefix: str) -> None:
    ensure_chat_history()
    for role, message in st.session_state.match_chat_history[-6:]:
        with st.chat_message(role):
            st.write(message)
    suggested = st.selectbox(
        "Quick question",
        [
            "",
            "Brazil vs Morocco",
            "Possible 4 semi-finalists",
            "Most likely winner",
            "Dark horse teams",
            "France fitness",
            "What does simulation runs mean?",
            "What does data confidence mean?",
            "Show match event report",
        ],
        key=f"{key_prefix}_suggested",
    )
    query = st.text_input(
        "Your question",
        value=suggested,
        placeholder="Ask about a team or match...",
        key=f"{key_prefix}_query",
    )
    if st.button("Ask AI Assistant", use_container_width=True, key=f"{key_prefix}_button"):
        answer = answer_match_assistant(query)
        st.session_state.match_chat_history.append(("user", query))
        st.session_state.match_chat_history.append(("assistant", answer))
        st.rerun()


def render_floating_chatbot() -> None:
    with st.popover("💬", help="Open AI Match Assistant"):
        st.markdown("### AI Match Assistant")
        st.caption("Ask about matches, teams, simulation, data trust, or confusing model signals.")
        render_chat_controls("floating_chat")


@st.cache_data
def load_all() -> tuple[pd.DataFrame, ...]:
    raw_teams = load_teams()
    recent_form = load_recent_form()
    rankings = load_rankings()
    injuries = load_injuries()
    if not recent_form.empty:
        raw_teams = raw_teams.merge(
            recent_form[["team", "weighted_form_0_100", "points_per_match", "goal_difference"]],
            on="team",
            how="left",
        )
        raw_teams["form_index"] = raw_teams["weighted_form_0_100"].fillna(raw_teams["form_index"])
    if not rankings.empty:
        raw_teams = raw_teams.merge(rankings[["team", "fifa_rank_april_2026"]], on="team", how="left")
    teams = add_team_features(raw_teams, injuries)
    players = load_players()
    fixtures = load_fixtures(teams)
    events = load_events()
    scorer_form = load_scorer_form()
    ingestion_summary = load_ingestion_summary()
    video_metadata = load_video_metadata()
    video_channels = load_video_channels()
    video_policy = load_video_policy()
    kaggle_video_catalog = load_kaggle_video_catalog()
    video_analysis_design = load_video_analysis_design()
    video_ingestion_summary = load_video_ingestion_summary()
    team_feature_store = load_team_feature_store()
    match_feature_store = load_match_feature_store()
    feature_dictionary = load_feature_dictionary()
    feature_engineering_summary = load_feature_engineering_summary()
    cleaning_summary = load_cleaning_summary()
    trust_report = load_trust_report()
    validation_report = load_validation_report()
    round_probabilities = load_tournament_round_probabilities()
    tournament_scenarios = load_tournament_scenarios()
    prediction_summary = load_prediction_summary()
    return (
        teams,
        players,
        injuries,
        fixtures,
        events,
        recent_form,
        rankings,
        scorer_form,
        ingestion_summary,
        video_metadata,
        video_channels,
        video_policy,
        kaggle_video_catalog,
        video_analysis_design,
        video_ingestion_summary,
        team_feature_store,
        match_feature_store,
        feature_dictionary,
        feature_engineering_summary,
        cleaning_summary,
        trust_report,
        validation_report,
        round_probabilities,
        tournament_scenarios,
        prediction_summary,
    )


(
    teams,
    players,
    injuries,
    fixtures,
    events,
    recent_form,
    rankings,
    scorer_form,
    ingestion_summary,
    video_metadata,
    video_channels,
    video_policy,
    kaggle_video_catalog,
    video_analysis_design,
    video_ingestion_summary,
    team_feature_store,
    match_feature_store,
    feature_dictionary,
    feature_engineering_summary,
    cleaning_summary,
    trust_report,
    validation_report,
    round_probabilities,
    tournament_scenarios,
    prediction_summary,
) = load_all()

render_floating_chatbot()

validation_failures = int((validation_report["status"] == "fail").sum()) if not validation_report.empty else 0
validation_warnings = int((validation_report["status"] == "warn").sum()) if not validation_report.empty else 0
trust_failures = int((trust_report["status"] == "fail").sum()) if not trust_report.empty else 0
trust_warnings = int((trust_report["status"] == "warn").sum()) if not trust_report.empty else 0
health_status = "pass" if validation_failures == 0 and trust_failures == 0 else "fail"
health_label = "Model Data Health"

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">World Cup 2026 AI Predictor</div>
        <p class="hero-subtitle">
            Prediction, simulation, tactical event analysis, source governance, and model-ready feature engineering
            for the senior men's tournament.
        </p>
        <div class="hero-row">
            <span class="pill">48 qualified teams</span>
            <span class="pill">{len(team_feature_store)} team feature rows</span>
            <span class="pill">{len(match_feature_store)} match feature rows</span>
            {status_pill(health_label, health_status)}
            <span class="pill">{validation_warnings + trust_warnings} active warnings</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Winner Probability",
        "Teams",
        "Team Comparison",
        "Match Predictor",
        "Explainable AI",
        "Interactive Bracket",
        "Tournament Simulation",
        "Match Analysis",
        "Video Sources",
        "Feature Store",
        "Data Health",
    ]
)

with tabs[0]:
    section_header("Winner Probability Chart", "A boardroom-ready view of tournament winner, finalist, and semi-final probabilities.")
    if round_probabilities.empty:
        st.info("No tournament probability file is loaded yet. Run `python3 scripts/build_tournament_predictions.py`.")
    else:
        probability_mode = st.radio(
            "Probability view",
            ["Winner", "Final", "Semi-final", "Quarter-final", "Round of 16"],
            horizontal=True,
        )
        probability_column = {
            "Winner": "winner_probability",
            "Final": "final_probability",
            "Semi-final": "semi_final_probability",
            "Quarter-final": "quarter_final_probability",
            "Round of 16": "round_of_16_probability",
        }[probability_mode]
        top_probability = round_probabilities.sort_values(probability_column, ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top Team", top_probability["team"])
        c2.metric(f"{probability_mode} Probability", f"{top_probability[probability_column]:.1%}")
        c3.metric("Model Strength", f"{top_probability['model_strength_score']:.1f}")
        c4.metric("Data Confidence", f"{top_probability['data_confidence_0_1']:.1%}")
        chart = probability_chart_frame(round_probabilities, probability_column, top_n=16)
        st.bar_chart(chart, height=360)
        st.dataframe(
            round_probabilities[
                [
                    "team",
                    "group",
                    "model_strength_score",
                    "data_confidence_0_1",
                    "semi_final_probability",
                    "final_probability",
                    "winner_probability",
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "data_confidence_0_1": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%"),
                "semi_final_probability": st.column_config.ProgressColumn("SF", min_value=0, max_value=1, format="%.1%"),
                "final_probability": st.column_config.ProgressColumn("Final", min_value=0, max_value=1, format="%.1%"),
                "winner_probability": st.column_config.ProgressColumn("Winner", min_value=0, max_value=1, format="%.1%"),
            },
        )
        st.download_button(
            "Export winner probabilities CSV",
            round_probabilities.to_csv(index=False).encode("utf-8"),
            file_name="worldcup_2026_winner_probabilities.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tabs[1]:
    section_header("Team Intelligence", "Scan group strength, FIFA rank, recent form, tactical style, and squad/scorer context.")
    photo_panel("Group-by-group football intelligence", "Ratings, form, tactical identity, and squad context layered over trusted data checks.")
    left, right = st.columns([2, 1])
    with left:
        group = st.selectbox("Group", sorted(teams["group"].unique()))
        visible = teams[teams["group"] == group].sort_values("strength", ascending=False)
        group_metrics = st.columns(4)
        group_metrics[0].metric("Group Teams", len(visible))
        group_metrics[1].metric("Top Strength", f"{visible['strength'].max():.1f}")
        group_metrics[2].metric("Avg Form", f"{visible['form_index'].mean():.1f}")
        group_metrics[3].metric("Avg Defense", f"{visible['defense'].mean():.1f}")
        team_columns = [
            "team",
            "confederation",
            "fifa_rank_april_2026",
            "strength",
            "form_index",
            "points_per_match",
            "goal_difference",
            "attack",
            "midfield",
            "defense",
            "depth",
            "tactical_style",
        ]
        st.dataframe(
            visible[[column for column in team_columns if column in visible.columns]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "strength": st.column_config.ProgressColumn("Strength", min_value=0, max_value=100, format="%.1f"),
                "form_index": st.column_config.ProgressColumn("Form", min_value=0, max_value=100, format="%.1f"),
                "attack": st.column_config.ProgressColumn("Attack", min_value=0, max_value=100, format="%d"),
                "midfield": st.column_config.ProgressColumn("Midfield", min_value=0, max_value=100, format="%d"),
                "defense": st.column_config.ProgressColumn("Defense", min_value=0, max_value=100, format="%d"),
                "depth": st.column_config.ProgressColumn("Depth", min_value=0, max_value=100, format="%d"),
            },
        )
        st.bar_chart(visible.set_index("team")["strength"], height=220)
    with right:
        team = st.selectbox("Ask about team", teams["team"].sort_values())
        team_pitch = team_feature_store[team_feature_store["team"] == team] if not team_feature_store.empty else pd.DataFrame()
        if not team_pitch.empty:
            team_row = team_pitch.iloc[0]
            st.markdown(
                f"""
                <div class="pitch-card">
                    <div class="pitch-overlay">
                        <strong>{html.escape(team)}</strong>
                        <span>Strength {team_row['model_strength_score']:.1f} · Confidence {format_percent(team_row['data_confidence_0_1'])}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(f"<div class='insight-box'>{html.escape(answer_team_question(team, teams, players))}</div>", unsafe_allow_html=True)
        if not team_feature_store.empty:
            row = team_feature_store[team_feature_store["team"] == team]
            if not row.empty:
                row = row.iloc[0]
                c1, c2 = st.columns(2)
                c1.metric("Model Strength", f"{row['model_strength_score']:.1f}")
                c2.metric("Confidence", format_percent(row["data_confidence_0_1"]))
        squad = squad_summary(players)
        if not squad.empty:
            st.markdown("**Squad Snapshot**")
            st.dataframe(squad[squad["team"] == team], use_container_width=True, hide_index=True)
        if not scorer_form.empty:
            st.markdown("**Recent National Scorers**")
            st.dataframe(scorer_form[scorer_form["team"] == team].head(8), use_container_width=True, hide_index=True)

with tabs[2]:
    section_header("Team Comparison Engine", "Compare two national teams across model strength, tournament path, form, attack, defense, risk, and confidence.")
    comp_a, comp_b = st.columns(2)
    with comp_a:
        compare_a = st.selectbox("Compare Team A", teams["team"].sort_values(), index=0)
    with comp_b:
        compare_b = st.selectbox("Compare Team B", teams["team"].sort_values(), index=1)
    if compare_a == compare_b:
        st.warning("Choose two different teams for comparison.")
    else:
        match_prediction = predict_match(compare_a, compare_b, teams)
        row_a = team_feature_row(compare_a)
        row_b = team_feature_row(compare_b)
        rp_a = round_probability_row(compare_a)
        rp_b = round_probability_row(compare_b)
        st.markdown(
            "<div class='comparison-grid'>"
            f"<div class='compare-card'><h3>{html.escape(compare_a)}</h3>"
            f"<div class='compare-row'><span>Strength</span><strong>{row_a['model_strength_score']:.1f}</strong></div>"
            f"<div class='compare-row'><span>Winner</span><strong>{rp_a['winner_probability']:.1%}</strong></div>"
            f"<div class='compare-row'><span>Semi-final</span><strong>{rp_a['semi_final_probability']:.1%}</strong></div>"
            f"<div class='compare-row'><span>Confidence</span><strong>{row_a['data_confidence_0_1']:.1%}</strong></div>"
            "</div>"
            f"<div class='compare-card'><h3>{html.escape(compare_b)}</h3>"
            f"<div class='compare-row'><span>Strength</span><strong>{row_b['model_strength_score']:.1f}</strong></div>"
            f"<div class='compare-row'><span>Winner</span><strong>{rp_b['winner_probability']:.1%}</strong></div>"
            f"<div class='compare-row'><span>Semi-final</span><strong>{rp_b['semi_final_probability']:.1%}</strong></div>"
            f"<div class='compare-row'><span>Confidence</span><strong>{row_b['data_confidence_0_1']:.1%}</strong></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{compare_a} win", f"{match_prediction['win_a']:.1%}")
        c2.metric("Draw", f"{match_prediction['draw']:.1%}")
        c3.metric(f"{compare_b} win", f"{match_prediction['win_b']:.1%}")
        st.markdown(f"<div class='insight-box'>{html.escape(str(match_prediction['explanation']))}</div>", unsafe_allow_html=True)
        comparison = compare_teams_frame(compare_a, compare_b)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if not comparison.empty:
            chart_data = comparison.set_index("metric")[[compare_a, compare_b]]
            st.bar_chart(chart_data, height=320)

with tabs[3]:
    section_header("Match Predictor", "Compare two teams with win/draw/loss probabilities and model feature gaps.")
    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox("Team A", teams["team"].sort_values(), index=0)
    with col_b:
        team_b = st.selectbox("Team B", teams["team"].sort_values(), index=1)
    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        prediction = predict_match(team_a, team_b, teams)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team_a} win", f"{prediction['win_a']:.1%}")
        c2.metric("Draw", f"{prediction['draw']:.1%}")
        c3.metric(f"{team_b} win", f"{prediction['win_b']:.1%}")
        st.markdown(
            f"""
            <div class="pitch-card">
                <div class="pitch-overlay">
                    <strong>{html.escape(team_a)} vs {html.escape(team_b)}</strong>
                    <span>{html.escape(team_a)} {prediction['win_a']:.1%} · Draw {prediction['draw']:.1%} · {html.escape(team_b)} {prediction['win_b']:.1%}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='insight-box'>{html.escape(str(prediction['explanation']))}</div>", unsafe_allow_html=True)
        probability_frame = pd.DataFrame(
            {
                "Outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
                "Probability": [prediction["win_a"], prediction["draw"], prediction["win_b"]],
            }
        )
        st.dataframe(
            probability_frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="%.1%")
            },
        )
        if not team_feature_store.empty:
            feature_rows = team_feature_store[team_feature_store["team"].isin([team_a, team_b])]
            compare_columns = [
                "team",
                "model_strength_score",
                "opponent_adjusted_form_0_100",
                "attack_efficiency_proxy",
                "defense_resilience_proxy",
                "fitness_risk_score_0_100",
                "source_reliability_0_1",
            ]
            st.markdown("**Team Feature Comparison**")
            st.dataframe(
                feature_rows[[column for column in compare_columns if column in feature_rows.columns]],
                use_container_width=True,
                hide_index=True,
            )

with tabs[4]:
    section_header("Explainable AI", "SHAP-style model driver view showing which features push a team's projection above or below baseline.")
    explain_team = st.selectbox("Team to explain", teams["team"].sort_values(), index=0)
    explanation = explain_team_features(explain_team)
    rp = round_probability_row(explain_team)
    row = team_feature_row(explain_team)
    if explanation.empty or row is None:
        st.info("Feature explanations are not available yet.")
    else:
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Model Strength", f"{row['model_strength_score']:.1f}")
        x2.metric("Winner", f"{rp['winner_probability']:.1%}" if rp is not None else "n/a")
        x3.metric("Semi-final", f"{rp['semi_final_probability']:.1%}" if rp is not None else "n/a")
        x4.metric("Confidence", f"{row['data_confidence_0_1']:.1%}")
        st.markdown(
            "<div class='insight-box'>This is an explainability layer built from the transparent feature weights used in the prototype. "
            "It behaves like SHAP at the dashboard level by showing each feature's additive impact versus the tournament baseline.</div>",
            unsafe_allow_html=True,
        )
        st.bar_chart(explanation.set_index("feature")["impact_points"], height=320)
        st.dataframe(
            explanation,
            use_container_width=True,
            hide_index=True,
            column_config={
                "impact_points": st.column_config.NumberColumn("Impact points", format="%+.2f"),
                "raw_value": st.column_config.NumberColumn("Team value", format="%.3f"),
                "baseline": st.column_config.NumberColumn("Tournament baseline", format="%.3f"),
            },
        )

with tabs[5]:
    section_header("Interactive Bracket", "Explore likely knockout paths and common scenario sets from the latest tournament simulation.")
    st.markdown(bracket_html(), unsafe_allow_html=True)
    if not tournament_scenarios.empty:
        scenario_type = st.selectbox("Scenario type", sorted(tournament_scenarios["scenario_type"].dropna().unique()))
        scenario_view = tournament_scenarios[tournament_scenarios["scenario_type"] == scenario_type].copy()
        st.dataframe(
            scenario_view.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "probability": st.column_config.ProgressColumn("Scenario probability", min_value=0, max_value=1, format="%.2%")
            },
        )
        st.download_button(
            "Export bracket scenarios CSV",
            scenario_view.to_csv(index=False).encode("utf-8"),
            file_name=f"worldcup_2026_{scenario_type}_scenarios.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No scenario file is loaded yet. Run `python3 scripts/build_tournament_predictions.py`.")

with tabs[6]:
    section_header("Tournament Simulation", "Run repeated group-stage scenarios to estimate advancement and winner probabilities.")
    photo_panel("Tournament probability engine", "Monte Carlo-style group simulations turn match probabilities into qualification and winner projections.")
    iterations = st.slider("Simulation runs", min_value=100, max_value=3000, value=500, step=100)
    results = simulate_group_stage(teams, fixtures, iterations=iterations)
    winners = tournament_winner_projection(results)
    sim_metrics = st.columns(4)
    sim_metrics[0].metric("Runs", f"{iterations:,}")
    sim_metrics[1].metric("Top Advance", f"{results.iloc[0]['team']}", f"{results.iloc[0]['advance_probability']:.1%}")
    sim_metrics[2].metric("Top Winner", f"{winners.iloc[0]['team']}", f"{winners.iloc[0]['winner_probability']:.1%}")
    sim_metrics[3].metric("Teams Simulated", f"{len(results)}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Group Qualification Probabilities**")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "group_win_probability": st.column_config.ProgressColumn("Group Win", min_value=0, max_value=1, format="%.1%"),
                "advance_probability": st.column_config.ProgressColumn("Advance", min_value=0, max_value=1, format="%.1%"),
                "strength": st.column_config.ProgressColumn("Strength", min_value=0, max_value=100, format="%.1f"),
            },
        )
    with c2:
        st.markdown("**Tournament Winner Projection**")
        st.dataframe(
            winners.head(16),
            use_container_width=True,
            hide_index=True,
            column_config={
                "winner_probability": st.column_config.ProgressColumn("Winner", min_value=0, max_value=1, format="%.1%"),
                "advance_probability": st.column_config.ProgressColumn("Advance", min_value=0, max_value=1, format="%.1%"),
                "strength": st.column_config.ProgressColumn("Strength", min_value=0, max_value=100, format="%.1f"),
            },
        )
        winner_chart = winners.head(10).set_index("team")["winner_probability"]
        st.bar_chart(winner_chart, height=240)
    if not round_probabilities.empty:
        st.markdown("**Advanced Round-by-Round Projection**")
        round_columns = [
            "team",
            "group",
            "model_strength_score",
            "data_confidence_0_1",
            "round_of_32_probability",
            "round_of_16_probability",
            "quarter_final_probability",
            "semi_final_probability",
            "final_probability",
            "winner_probability",
        ]
        st.dataframe(
            round_probabilities[[column for column in round_columns if column in round_probabilities.columns]].head(16),
            use_container_width=True,
            hide_index=True,
            column_config={
                "data_confidence_0_1": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%"),
                "round_of_32_probability": st.column_config.ProgressColumn("R32", min_value=0, max_value=1, format="%.1%"),
                "round_of_16_probability": st.column_config.ProgressColumn("R16", min_value=0, max_value=1, format="%.1%"),
                "quarter_final_probability": st.column_config.ProgressColumn("QF", min_value=0, max_value=1, format="%.1%"),
                "semi_final_probability": st.column_config.ProgressColumn("SF", min_value=0, max_value=1, format="%.1%"),
                "final_probability": st.column_config.ProgressColumn("Final", min_value=0, max_value=1, format="%.1%"),
                "winner_probability": st.column_config.ProgressColumn("Winner", min_value=0, max_value=1, format="%.1%"),
            },
        )
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Most Likely Semi-finalists**")
            semi_chart = round_probabilities.sort_values("semi_final_probability", ascending=False).head(8).set_index("team")[
                "semi_final_probability"
            ]
            st.bar_chart(semi_chart, height=260)
        with c4:
            st.markdown("**Common Scenario Sets**")
            st.dataframe(tournament_scenarios.head(12), use_container_width=True, hide_index=True)

with tabs[7]:
    section_header("Match Analysis", "Turn structured event rows into top moments, momentum timeline, and post-match narrative.")
    match_ids = ["All"] + sorted(events["match_id"].dropna().unique().tolist()) if not events.empty else ["All"]
    selected_match = st.selectbox("Event feed", match_ids)
    view_events = events if selected_match == "All" else events[events["match_id"] == selected_match]
    event_metrics = st.columns(4)
    event_metrics[0].metric("Events", len(view_events))
    event_metrics[1].metric("Goals", int((view_events["event_type"] == "goal").sum()) if not view_events.empty else 0)
    event_metrics[2].metric("Big Chances", int((view_events["event_type"] == "big_chance").sum()) if not view_events.empty else 0)
    event_metrics[3].metric("Total xG", f"{view_events['xg'].sum():.2f}" if not view_events.empty else "0.00")
    st.markdown("**AI-style Report**")
    st.markdown(f"<div class='insight-box'>{html.escape(generate_match_report(view_events))}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Moments**")
        st.dataframe(top_moments(view_events), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Momentum Timeline**")
        timeline = momentum_timeline(view_events)
        st.dataframe(timeline, use_container_width=True, hide_index=True)
        if not timeline.empty:
            st.line_chart(timeline.set_index("minute")["momentum"], height=220)

with tabs[8]:
    section_header("Video Sources", "Strictly cleaned source links and metadata for senior men's football relevance.")
    photo_panel("Legit video intelligence layer", "Source links, embeds, platform rules, and metadata only: no copyrighted video downloads.")
    video_metrics = st.columns(4)
    video_metrics[0].metric("Clean Video Rows", len(video_metadata))
    video_metrics[1].metric("Channels Indexed", len(video_channels))
    video_metrics[2].metric("Kaggle Datasets", len(kaggle_video_catalog))
    video_metrics[3].metric("Download Allowed", "No")
    source_filter = st.selectbox(
        "Source",
        ["All"] + sorted(video_metadata["source_name"].dropna().unique().tolist()) if not video_metadata.empty else ["All"],
    )
    view_videos = video_metadata if source_filter == "All" else video_metadata[video_metadata["source_name"] == source_filter]
    if view_videos.empty:
        st.info("No video metadata is loaded yet. Run `python3 scripts/video_source_ingest.py` to refresh public metadata.")
    else:
        video_columns = [
            "source_name",
            "team_or_scope",
            "title",
            "published",
            "url",
            "matched_keywords",
            "relevance_score",
            "rights_posture",
            "download_allowed",
        ]
        st.dataframe(
            view_videos[[column for column in video_columns if column in view_videos.columns]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("URL"),
                "relevance_score": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=5, format="%d"),
            },
        )
        first_video = view_videos.iloc[0]
        if first_video.get("embed_url"):
            st.video(first_video["url"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kaggle/Open Dataset Catalog**")
        st.dataframe(
            kaggle_video_catalog,
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("URL")},
        )
    with c2:
        st.markdown("**Platform Policy Guardrails**")
        st.dataframe(video_policy, use_container_width=True, hide_index=True)

    st.markdown("**Copyright-safe Analysis Design**")
    st.dataframe(video_analysis_design, use_container_width=True, hide_index=True)

with tabs[9]:
    section_header("Feature Store", "Model-ready team and match features with reliability and risk signals.")
    if team_feature_store.empty:
        st.info("No feature store is loaded yet. Run `python3 scripts/build_feature_store.py` to generate it.")
    else:
        feature_metrics = st.columns(4)
        feature_metrics[0].metric("Team Rows", len(team_feature_store))
        feature_metrics[1].metric("Team Columns", len(team_feature_store.columns))
        feature_metrics[2].metric("Match Rows", len(match_feature_store))
        feature_metrics[3].metric("Match Columns", len(match_feature_store.columns))
        feature_columns = [
            "team",
            "group",
            "model_strength_score",
            "data_confidence_0_1",
            "rank_strength_0_100",
            "form_strength_0_100",
            "opponent_adjusted_form_0_100",
            "attack_efficiency_proxy",
            "defense_resilience_proxy",
            "goal_diff_per_match",
            "fitness_availability_0_1",
            "fitness_risk_score_0_100",
            "source_reliability_0_1",
            "recent_goal_scorers",
            "recent_scorer_goals_total",
            "relevant_video_score",
        ]
        st.dataframe(
            team_feature_store[[column for column in feature_columns if column in team_feature_store.columns]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "model_strength_score": st.column_config.ProgressColumn("Strength", min_value=0, max_value=100, format="%.1f"),
                "data_confidence_0_1": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%"),
                "source_reliability_0_1": st.column_config.ProgressColumn("Reliability", min_value=0, max_value=1, format="%.1%"),
                "fitness_risk_score_0_100": st.column_config.ProgressColumn("Fitness Risk", min_value=0, max_value=100, format="%.1f"),
            },
        )
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("**Top Model Strength**")
            st.bar_chart(team_feature_store.head(12).set_index("team")["model_strength_score"], height=260)
        with chart_cols[1]:
            st.markdown("**Confidence vs Reliability**")
            scatter_data = team_feature_store[["team", "data_confidence_0_1", "source_reliability_0_1"]].set_index("team")
            st.scatter_chart(scatter_data, x="data_confidence_0_1", y="source_reliability_0_1", height=260)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Match-level Feature Deltas**")
        st.dataframe(match_feature_store, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Feature Dictionary**")
        st.dataframe(feature_dictionary, use_container_width=True, hide_index=True)
    st.markdown("**Trust Report**")
    st.dataframe(trust_report, use_container_width=True, hide_index=True)

with tabs[10]:
    section_header("Data Health", "Audit the data pipeline, cleaning rules, validation checks, and known limitations.")
    photo_panel("Trust before prediction", "Every model-facing signal is checked for relevance, ranges, duplicates, leakage, and source reliability.")
    health_metrics = st.columns(5)
    health_metrics[0].metric("Validation Pass", int((validation_report["status"] == "pass").sum()) if not validation_report.empty else 0)
    health_metrics[1].metric("Validation Warn", validation_warnings)
    health_metrics[2].metric("Validation Fail", validation_failures)
    health_metrics[3].metric("Trust Warn", trust_warnings)
    health_metrics[4].metric("Trust Fail", trust_failures)
    loaded_counts = pd.DataFrame(
        [
            {"dataset": "teams", "rows": len(teams)},
            {"dataset": "players", "rows": len(players)},
            {"dataset": "injuries", "rows": len(injuries)},
            {"dataset": "fixtures", "rows": len(fixtures)},
            {"dataset": "match_events", "rows": len(events)},
            {"dataset": "recent_form", "rows": len(recent_form)},
            {"dataset": "rankings", "rows": len(rankings)},
            {"dataset": "recent_scorers", "rows": len(scorer_form)},
            {"dataset": "video_metadata", "rows": len(video_metadata)},
            {"dataset": "video_channels", "rows": len(video_channels)},
            {"dataset": "kaggle_video_catalog", "rows": len(kaggle_video_catalog)},
            {"dataset": "team_feature_store", "rows": len(team_feature_store)},
            {"dataset": "match_feature_store", "rows": len(match_feature_store)},
            {"dataset": "validation_checks", "rows": len(validation_report)},
            {"dataset": "trust_checks", "rows": len(trust_report)},
        ]
    )
    st.dataframe(loaded_counts, use_container_width=True, hide_index=True)
    if not ingestion_summary.empty:
        st.markdown("**Web Ingestion Summary**")
        st.dataframe(ingestion_summary, use_container_width=True, hide_index=True)
    if not video_ingestion_summary.empty:
        st.markdown("**Video Ingestion Summary**")
        st.dataframe(video_ingestion_summary, use_container_width=True, hide_index=True)
    if not feature_engineering_summary.empty:
        st.markdown("**Feature Engineering Summary**")
        st.dataframe(feature_engineering_summary, use_container_width=True, hide_index=True)
    if not cleaning_summary.empty:
        st.markdown("**Cleaning Summary**")
        st.dataframe(cleaning_summary, use_container_width=True, hide_index=True)
    if not validation_report.empty:
        st.markdown("**Validation Report**")
        st.dataframe(validation_report, use_container_width=True, hide_index=True)
    if not trust_report.empty:
        st.markdown("**Trust Report**")
        st.dataframe(trust_report, use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='insight-box'>Squads, injuries, and some player ratings are placeholder/live-update fields. "
        "Replace the CSV rows with verified sources as official final squads and match feeds become available.</div>",
        unsafe_allow_html=True,
    )
