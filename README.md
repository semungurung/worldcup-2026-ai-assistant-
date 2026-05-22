# World Cup 2026 AI Prediction and Match Analysis Assistant

V1 prototype for a tournament-level football intelligence system. It is intentionally separate from the earlier injury-prediction app because it combines team data, player/squad inputs, injuries, fixtures, simulations, and match-event analysis.

## What is included

- 48-team World Cup 2026 seed dataset
- Editable CSV inputs for teams, players, injuries, fixtures, and match events
- Team strength feature engineering
- Match win/draw/loss probability model
- Group-stage simulation with 12 groups and best third-place advancement
- Early tournament winner projection
- Structured match-event analysis with top moments and momentum summaries
- Streamlit dashboard

## Run

```bash
cd worldcup_2026_ai_assistant
pip install -r requirements.txt
streamlit run app.py
```

## Refresh public data

```bash
cd worldcup_2026_ai_assistant
python3 scripts/web_ingest.py
```

The ingestion script downloads public international-football result resources, writes raw CSVs to `data/raw/`, and writes model-ready CSVs to `data/derived/`.

Current generated files include:

- `data/raw/international_results.csv`
- `data/raw/international_goalscorers.csv`
- `data/raw/international_shootouts.csv`
- `data/derived/worldcup_team_historical_results.csv`
- `data/derived/recent_team_form.csv`
- `data/derived/fixture_head_to_head.csv`
- `data/derived/recent_scorer_form.csv`
- `data/derived/fifa_rankings_wc_teams.csv`
- `data/derived/teams_enriched.csv`
- `data/data_sources.csv`
- `data/ingestion_summary.csv`

## Refresh video-source metadata

```bash
cd worldcup_2026_ai_assistant
python3 scripts/video_source_ingest.py
```

This collects public metadata and links only. It does not download, copy, or rehost copyrighted video footage.

Generated video files include:

- `data/video_channel_seeds.csv`
- `data/derived/video_channel_index.csv`
- `data/derived/video_metadata.csv`
- `data/derived/video_metadata_relevant.csv`
- `data/kaggle_video_dataset_catalog.csv`
- `data/video_platform_policy.csv`
- `data/video_analysis_design.csv`
- `data/video_ingestion_summary.csv`

## Video and copyright guardrails

For V1, the app should analyze video-adjacent data in this order:

- metadata, links, thumbnails, and public embed URLs
- transcripts/captions only where API/platform terms allow access
- Kaggle/open datasets only after checking licence and provenance
- licensed video/event feeds for full production analysis

Do not scrape Facebook or Instagram pages directly. Meta's automated collection rules require permission for automated collection from Meta products, so this project stores manual/API-only rows for those platforms.

## Build engineered features

```bash
cd worldcup_2026_ai_assistant
python3 scripts/build_feature_store.py
```

This creates model-ready features from team ratings, recent form, FIFA rank, squad placeholders, injury rows, scorer history, head-to-head results, and video-source metadata.

Generated feature files:

- `data/derived/team_feature_store.csv`
- `data/derived/match_feature_store.csv`
- `data/derived/feature_dictionary.csv`
- `data/feature_engineering_summary.csv`

Current feature families:

- team strength: base rating, FIFA-rank strength, recent weighted form, position ratings, host boost
- attacking signal: goals for per match, scorer depth, top-scorer concentration, projected player xG
- defensive signal: goals against per match, goal-difference rate, defensive stability
- fitness signal: injury severity, squad availability, injury-risk placeholders
- match signal: strength delta, rank delta, form delta, attack-vs-defense deltas, head-to-head edge
- data quality: data confidence score based on available form, squad, scorer, video, and ranking coverage
- video signal: relevant metadata volume from official/media sources, without downloading copyrighted footage

## Clean model data

```bash
cd worldcup_2026_ai_assistant
python3 scripts/video_source_ingest.py
python3 scripts/clean_model_data.py
python3 scripts/build_feature_store.py
python3 scripts/validate_model_data.py
python3 scripts/build_tournament_predictions.py
```

Cleaning writes audited model-facing files to `data/cleaned/` and a row-count report to `data/cleaning_summary.csv`.

Current cleaning rules:

- normalize team names and aliases
- keep only World Cup-qualified teams in model-facing datasets
- coerce numeric fields and dates into consistent types
- remove duplicate rows
- remove unplayed/future historical results from form calculations
- remove own-goal rows from scorer-form features
- filter video metadata to senior men's football relevance
- exclude women's, youth, futsal, beach soccer, eFootball, and non-football video rows
- keep video metadata/links only; do not download copyrighted footage

Senior-grade validation and trust reports:

- `data/validation_report.csv`
- `data/trust_report.csv`

Additional senior-grade engineered features:

- opponent-adjusted recent form
- average recent opponent rank
- clean-sheet and failed-to-score rates
- competitive-match share
- attack-efficiency proxy
- defense-resilience proxy
- squad-depth proxy
- fitness-risk score
- source-reliability score
- match-level deltas for adjusted form, attack efficiency, defense resilience, fitness risk, and squad depth

Advanced tournament prediction outputs:

- `data/derived/tournament_round_probabilities.csv`
- `data/derived/tournament_scenarios.csv`
- `data/derived/prediction_summary.csv`

These add round-of-32, round-of-16, quarterfinal, semifinal, final, and winner probabilities for the AI assistant and dashboard. The current knockout path is an approximate seeded bracket until the full official bracket mapping is loaded.

## Data notes

The qualified-team list and opening fixture seed rows follow FIFA's current World Cup 2026 team and schedule pages as checked on May 9, 2026. Squad/player rows are placeholders because final 26-player squads and late injury status are live-update data.

Update these files as new information arrives:

- `data/teams.csv`
- `data/players.csv`
- `data/injuries.csv`
- `data/fixtures.csv`
- `data/match_events.csv`

## Suggested V2

- Connect a licensed football data API for squads, injuries, fixtures, lineups, and event feeds
- Add calibrated Elo/SPI-style ratings from historical match data
- Train match outcome models against international fixtures
- Add knockout-bracket simulation with exact FIFA bracket mapping
- Add LLM retrieval over match reports and event feeds
- Add video/event-data analysis once licensed feeds are available
