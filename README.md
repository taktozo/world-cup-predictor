# Score Predictor

A live Streamlit dashboard that predicts soccer match scores, win/draw/loss
probabilities, both-teams-to-score, and over/under goal totals — for **World
Cup / international matches** and for **club football (EPL + La Liga)**.

**Live app:** https://world-cup-predictor-syyhqq27k8gamnzeabtmjb.streamlit.app
**Repo:** https://github.com/taktozo/world-cup-predictor

Built by Taktozo Productions.

## What it does

Two pages, each backed by its own model:

| | International (`app.py`) | Club Leagues (`pages/1_Club_Leagues.py`) |
|---|---|---|
| Coverage | All international matches, World Cup focus | EPL + La Liga |
| Pick a real upcoming fixture | — | Yes (football-data.org) |
| Model inputs | Elo, recent form, head-to-head | Elo (self-computed), recent form, head-to-head, squad market value |
| Extra context shown | — | Live league table, last-5 form, last-5 h2h |
| "Living" data | Static snapshot, manual refresh | Auto-refreshes from live results every 6h |

Both pages model each side's goals as independent Poisson distributions
(`home_goals ~ Poisson(λ_home)`, `away_goals ~ Poisson(λ_away)`), with λ a
log-linear function of the features above. Win/draw/loss, both-teams-to-score,
and over/under probabilities are all derived by summing the joint
home×away scoreline grid — see `outcome_probabilities_from_grid` /
`btts_probability_from_grid` / `over_under_probabilities_from_grid` in
`src/inference.py` (international) and `src/club/club_inference.py` (club).

## Data sources

**International:**
- Match results: Kaggle `martj42/international-football-results-from-1872-to-2017`
- Elo ratings: Kaggle `saifalnimri/international-football-elo-ratings` (sourced from eloratings.net)

**Club:**
- Match results, betting odds, referee, and match stats: [football-data.co.uk](https://www.football-data.co.uk) (free, no auth, 1993/94–present)
- Squad market value: Kaggle `davidcariboo/player-scores` (Transfermarkt scrape) — computed via a single chronological sweep over `player_valuations.csv`, not the much heavier `game_lineups.csv`
- Real upcoming fixtures: [football-data.org](https://www.football-data.org) free tier (needs a free API token, see Setup)
- Elo: computed in-house using the same formula as eloratings.net (`R_new = R_old + K·G·(W−We)`), since clubelo.com's public API was unreliable when we checked

## Repo layout

```
app.py                          International page (entry point)
pages/1_Club_Leagues.py         Club leagues page
ui.py                           Shared theme CSS + footer, used by both pages

src/                            International pipeline
  download_data.py                Pull raw Kaggle datasets
  merge_elo.py                    Attach pre-match Elo to each result (as-of merge)
  add_form_features.py            Rolling last-10-match goals for/against
  add_h2h_features.py             Shrunk head-to-head history between each pair
  export_latest_*.py              Small "current state" snapshots the app reads live
  export_training_data.py         Lean committed CSV the deployed app trains from
  features.py / inference.py      Feature building + prediction math
  train_baseline.py / predict.py  Offline evaluation / CLI for trying matchups

src/club/                       Club pipeline (same shape, club-specific data)
  download_club_data.py            Pull football-data.co.uk season CSVs
  compute_elo.py                   Self-computed club Elo
  add_form_h2h.py                  Rolling form + head-to-head (club version)
  add_odds_features.py             Bookmaker-odds-implied probabilities
  add_squad_value.py               Squad market value via valuation sweep
  add_referee_features.py          Referee card-tendency
  export_snapshots.py              Live-lookup snapshots (Elo/form/h2h/squad value)
  export_recent_history.py         Standings/recent-form/recent-h2h display data
  fixtures.py                      Real upcoming fixtures (football-data.org)
  live_refresh.py                  Incremental Elo/form/h2h updates from new results
  standings.py                     League table computation (zero-fillable roster)
  club_features.py / club_inference.py / train_baseline.py

data/                            Committed: small snapshots + lean training CSVs
                                  Gitignored: raw/intermediate pipeline files (regenerate via src/ scripts)
notebooks/                       EDA
```

## Running locally

```
pip install -r requirements-dev.txt   # includes kaggle, jupyter (dev-only extras)
python src/download_data.py           # needs a Kaggle API token, see kaggle.com/settings
python src/merge_elo.py
python src/add_form_features.py
python src/add_h2h_features.py
python src/export_latest_elo.py && python src/export_latest_form.py && python src/export_latest_h2h.py
python src/export_training_data.py
streamlit run app.py
```

The club pipeline follows the same pattern from `src/club/` (see that
directory's scripts in rough dependency order: download → compute_elo →
add_form_h2h → add_odds_features → add_squad_value → add_referee_features →
export_training_data → export_snapshots → export_recent_history).

## Secrets

Two API tokens, neither committed to git:

- **`FOOTBALL_DATA_API_TOKEN`** (optional) — powers "real upcoming fixture" mode
  on the Club Leagues page. Get a free token at football-data.org. Locally, put
  it in `.streamlit/secrets.toml`; on Streamlit Cloud, add it under the app's
  Settings → Secrets.
- **Kaggle API token** — only needed to re-run the data pipeline (`src/download_data.py`,
  `src/club/download_club_data.py`'s Transfermarkt calls). Not needed to just run the app.

Without the football-data.org token, the Club Leagues page still works fine in
"Hypothetical matchup" mode.

## Known limitations

- **Squad value doesn't live-refresh.** Elo/form/h2h auto-update from new
  match results; squad value only updates when the Transfermarkt pipeline is
  re-run manually (it changes slowly outside transfer windows, so this is a
  low-priority gap).
- **A few newly-promoted/long-relegated clubs** (e.g. Coventry City, Racing
  Santander) lack recent squad-value data, so fixtures involving them are
  excluded from the "real upcoming fixture" picker rather than predicted
  with stale data.
- **No true in-play updating.** Predictions are pre-match only; nothing
  updates mid-game.
- **Betting odds and referee** are used as features in the club model's
  offline research/evaluation only (not the deployed "live" model), since
  neither is knowable for a hypothetical matchup — only for specific,
  already-scheduled real fixtures.
- **Streamlit multi-page collision gotcha (bit us twice):** every `.py` module
  imported via `sys.path` tricks, and every `@st.cache_data`/`@st.cache_resource`
  wrapper function, must have a name that's unique across *both* pages -- they
  run in one shared Python process, so a same-named module or wrapper in the
  other pipeline silently shadows this one instead of raising an error. Fixed
  twice already: `features.py`/`inference.py` -> `club_features.py`/
  `club_inference.py`, and `fixtures.py` -> `club_fixtures.py`. Two more
  filenames are still duplicated between `src/` and `src/club/`
  (`export_training_data.py`, `train_baseline.py`) but are currently safe
  since neither page imports them (only run standalone via `python
  script.py`) -- **if either ever gets imported by app.py or the club page,
  rename the club copy first.**

## Model performance (last evaluated)

| | Outcome accuracy | vs. naive baseline |
|---|---|---|
| International, World Cup subset | 57.1% | 47.8% |
| Club (live model: no odds/referee) | 52.6% | 44.7% |
| Club (research model: +odds/referee) | 53.4% | — (close to bookmakers' own 54.5%) |
