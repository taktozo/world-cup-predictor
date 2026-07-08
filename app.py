"""Streamlit dashboard for the World Cup score predictor.

Run locally with: streamlit run app.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import streamlit as st

from features import FEATURE_COLUMNS
from fixtures import get_upcoming_fixtures
from inference import fit_model, h2h_diff_for_pair, load_latest_elo, load_latest_form, load_latest_h2h, predict_match
from ui import apply_theme, render_footer

OUTCOME_COLORS = {"Home win": "#4C72B0", "Draw": "#8C8C8C", "Away win": "#C44E52"}
TRAINING_DATA_PATH = Path(__file__).resolve().parent / "data" / "training_data.csv"

st.set_page_config(page_title="World Cup Score Predictor", page_icon="⚽")
apply_theme()


def _model_cache_key() -> str:
    """Changes whenever the training data or feature set changes.

    st.cache_resource keys on the wrapped function's own source code, not on
    what it calls into -- so a code push that only changes build_features()
    or training_data.csv (not fit_model() itself) can otherwise leave a stale
    cached model in memory if the host reuses the running process instead of
    a full restart. Passing this in makes that impossible to miss.
    """
    file_hash = hashlib.md5(TRAINING_DATA_PATH.read_bytes()).hexdigest()
    columns_hash = hashlib.md5(",".join(FEATURE_COLUMNS).encode()).hexdigest()
    return f"{file_hash}-{columns_hash}"


@st.cache_resource
def get_model(cache_key: str):
    return fit_model()


@st.cache_data
def get_latest_elo():
    return load_latest_elo()


@st.cache_data
def get_latest_form():
    return load_latest_form()


@st.cache_data
def get_latest_h2h():
    return load_latest_h2h()


@st.cache_data(ttl=6 * 60 * 60)  # tournament fixtures/results can change during an active World Cup
def get_fixtures(_api_token):
    return get_upcoming_fixtures(_api_token)


st.title("⚽ World Cup Score Predictor")
st.caption(
    "Baseline model: independent Poisson regressions on Elo rating and venue. "
    "Predictions are a starting point, not betting advice."
)

latest_elo = get_latest_elo()
latest_form = get_latest_form()
latest_h2h = get_latest_h2h()
model = get_model(_model_cache_key())
teams = sorted(set(latest_elo.index) & set(latest_form.index))

try:
    api_token = st.secrets.get("FOOTBALL_DATA_API_TOKEN")
except FileNotFoundError:
    api_token = None  # no secrets.toml configured at all -- fall back gracefully
mode_options = ["Real upcoming fixture", "Hypothetical matchup"] if api_token else ["Hypothetical matchup"]
mode = st.radio("Mode", mode_options, horizontal=True)
if not api_token:
    st.caption("Real fixtures unavailable (no football-data.org API token configured).")

home_team = away_team = None
neutral = True

if mode == "Real upcoming fixture":
    try:
        fixtures = get_fixtures(api_token)
    except Exception as e:
        st.error(f"Couldn't fetch fixtures right now ({e}). Try hypothetical matchup mode instead.")
        fixtures = None

    if fixtures is not None:
        fixtures = fixtures[fixtures["home_team"].isin(teams) & fixtures["away_team"].isin(teams)]
    if fixtures is None or fixtures.empty:
        st.warning("No predictable upcoming World Cup fixtures found. Try hypothetical matchup mode instead.")
    else:
        labels = [
            f"{row.date.strftime('%a %d %b')} ({row.stage}): {row.home_team} vs {row.away_team}"
            for row in fixtures.itertuples()
        ]
        choice = st.selectbox("Upcoming fixture", labels)
        chosen = fixtures.iloc[labels.index(choice)]
        home_team, away_team = chosen["home_team"], chosen["away_team"]
else:
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with col2:
        away_default = "Argentina" if "Argentina" in teams else teams[1]
        away_team = st.selectbox("Away team", teams, index=teams.index(away_default))

    neutral = st.checkbox("Neutral venue (e.g. World Cup match)", value=True)

if home_team == away_team:
    st.warning("Pick two different teams.")
else:
    result = predict_match(home_team, away_team, neutral, latest_elo, latest_form, latest_h2h, model)

    col_elo1, col_elo2, col_h2h = st.columns(3)
    with col_elo1:
        st.metric(f"{home_team} Elo", f"{result['home_elo']:.0f}")
    with col_elo2:
        st.metric(f"{away_team} Elo", f"{result['away_elo']:.0f}")
    with col_h2h:
        h2h_pair_key = tuple(sorted([home_team, away_team]))
        h2h_count = (
            int(latest_h2h.loc[h2h_pair_key, "count"]) if h2h_pair_key in latest_h2h.index else 0
        )
        st.metric("Past meetings", h2h_count)

    st.subheader("Expected score")
    st.markdown(
        f"### {home_team} **{result['home_goals']:.2f}** — **{result['away_goals']:.2f}** {away_team}"
    )

    st.subheader("Outcome probabilities")
    labels = [f"{home_team} win", "Draw", f"{away_team} win"]
    values = [result["p_home_win"], result["p_draw"], result["p_away_win"]]
    colors = [OUTCOME_COLORS["Home win"], OUTCOME_COLORS["Draw"], OUTCOME_COLORS["Away win"]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.0%}", va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    st.pyplot(fig)

    col_btts, col_ou = st.columns(2)

    with col_btts:
        st.subheader("Both teams to score")
        btts_labels = ["Yes", "No"]
        btts_values = [result["p_btts_yes"], result["p_btts_no"]]
        btts_colors = [OUTCOME_COLORS["Home win"], OUTCOME_COLORS["Draw"]]

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(btts_labels, btts_values, color=btts_colors, width=0.5)
        for bar, val in zip(bars, btts_values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.0%}", ha="center", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    with col_ou:
        st.subheader("Over/under total goals")
        lines = list(result["over_under"].keys())
        over_values = list(result["over_under"].values())

        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.bar([str(line) for line in lines], over_values, color=OUTCOME_COLORS["Home win"], width=0.6)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Line")
        ax.set_ylabel("P(over)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

st.divider()
st.caption(
    "Elo ratings from eloratings.net (via Kaggle), snapshotted at build time. "
    "Match history from martj42/international-football-results-from-1872-to-2017."
)
render_footer()
