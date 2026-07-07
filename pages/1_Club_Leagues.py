"""Streamlit page for the club-league (EPL / La Liga) score predictor."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "club"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from club_features import LIVE_FEATURE_COLUMNS
from club_inference import (
    fit_live_model,
    load_latest_elo,
    load_latest_form,
    load_latest_h2h,
    load_latest_squad_value,
    predict_match,
)
from live_refresh import get_refreshed_snapshots

OUTCOME_COLORS = {"Home win": "#4C72B0", "Draw": "#8C8C8C", "Away win": "#C44E52"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "club"

st.set_page_config(page_title="Club League Score Predictor", page_icon="⚽")


def _model_cache_key() -> str:
    """See app.py's _model_cache_key for why this exists: st.cache_resource
    keys on this function's own source, not on the training data it reads,
    so a data/feature update wouldn't otherwise invalidate a running app's
    cached model."""
    file_hash = hashlib.md5((DATA_DIR / "training_data.csv").read_bytes()).hexdigest()
    columns_hash = hashlib.md5(",".join(LIVE_FEATURE_COLUMNS).encode()).hexdigest()
    return f"{file_hash}-{columns_hash}"


@st.cache_resource
def get_model(cache_key: str):
    return fit_live_model()


@st.cache_data
def get_latest_elo():
    return load_latest_elo()


@st.cache_data
def get_latest_form():
    return load_latest_form()


@st.cache_data
def get_latest_h2h():
    return load_latest_h2h()


@st.cache_data
def get_latest_squad_value():
    return load_latest_squad_value()


@st.cache_data
def get_team_league():
    return pd.read_csv(DATA_DIR / "team_league.csv").set_index("team")["league"]


@st.cache_data
def get_committed_max_date():
    return pd.read_csv(DATA_DIR / "training_data.csv", usecols=["Date"], parse_dates=["Date"])["Date"].max()


@st.cache_data(ttl=6 * 60 * 60)  # re-check for new results every 6 hours
def get_live_snapshots(_elo, _form, _h2h, committed_max_date):
    return get_refreshed_snapshots(_elo, _form, _h2h, committed_max_date)


st.title("⚽ Club League Score Predictor")
st.caption(
    "EPL & La Liga. Baseline model: independent Poisson regressions on Elo, recent form, "
    "head-to-head, and squad market value. Predictions are a starting point, not betting advice."
)

committed_elo = get_latest_elo()
committed_form = get_latest_form()
committed_h2h = get_latest_h2h()
latest_squad_value = get_latest_squad_value()
team_league = get_team_league()
model = get_model(_model_cache_key())

committed_max_date = get_committed_max_date()
latest_elo, latest_form, latest_h2h, n_new_matches = get_live_snapshots(
    committed_elo, committed_form, committed_h2h, committed_max_date
)
if n_new_matches:
    st.caption(f"Ratings updated with {n_new_matches} match(es) played since the last full data refresh.")

usable_teams = set(latest_elo.index) & set(latest_form.index) & set(latest_squad_value.index)

league = st.radio("League", ["EPL", "La Liga"], horizontal=True)
teams = sorted(t for t in usable_teams if team_league.get(t) == league)

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", teams, index=0)
with col2:
    away_team = st.selectbox("Away team", teams, index=min(1, len(teams) - 1))

if home_team == away_team:
    st.warning("Pick two different teams.")
else:
    result = predict_match(home_team, away_team, latest_elo, latest_form, latest_h2h, latest_squad_value, model)

    col_elo1, col_elo2, col_h2h = st.columns(3)
    with col_elo1:
        st.metric(f"{home_team} Elo", f"{result['home_elo']:.0f}")
    with col_elo2:
        st.metric(f"{away_team} Elo", f"{result['away_elo']:.0f}")
    with col_h2h:
        h2h_pair_key = tuple(sorted([home_team, away_team]))
        h2h_count = int(latest_h2h.loc[h2h_pair_key, "count"]) if h2h_pair_key in latest_h2h.index else 0
        st.metric("Past meetings", h2h_count)

    st.caption(
        f"Squad value: {home_team} €{result['home_squad_value'] / 1e6:.0f}m vs. "
        f"{away_team} €{result['away_squad_value'] / 1e6:.0f}m"
    )

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
    "Match/odds/referee data from football-data.co.uk. Squad market value from "
    "Transfermarkt (davidcariboo/player-scores on Kaggle). Elo computed in-house."
)
