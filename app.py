"""Streamlit dashboard for the World Cup score predictor.

Run locally with: streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import streamlit as st

from inference import load_latest_elo, load_model, predict_match

OUTCOME_COLORS = {"Home win": "#4C72B0", "Draw": "#8C8C8C", "Away win": "#C44E52"}

st.set_page_config(page_title="World Cup Score Predictor", page_icon="⚽")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_latest_elo():
    return load_latest_elo()


st.title("⚽ World Cup Score Predictor")
st.caption(
    "Baseline model: independent Poisson regressions on Elo rating and venue. "
    "Predictions are a starting point, not betting advice."
)

latest_elo = get_latest_elo()
model = get_model()
teams = sorted(latest_elo.index)

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
    result = predict_match(home_team, away_team, neutral, latest_elo, model)

    st.metric(f"{home_team} Elo", f"{result['home_elo']:.0f}")
    st.metric(f"{away_team} Elo", f"{result['away_elo']:.0f}", delta=None)

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

st.divider()
st.caption(
    "Elo ratings from eloratings.net (via Kaggle), snapshotted at build time. "
    "Match history from martj42/international-football-results-from-1872-to-2017."
)
