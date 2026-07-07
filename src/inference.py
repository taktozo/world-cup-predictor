"""Shared inference helpers: load the trained model and score a matchup.

Used by predict.py (CLI), train_baseline.py (evaluation), and app.py
(Streamlit dashboard) so the outcome-probability math lives in one place.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_GOALS = 10  # grid cutoff for outcome-probability summation


def load_latest_elo() -> pd.Series:
    latest = pd.read_csv(DATA_DIR / "latest_elo.csv")
    return latest.set_index("team")["rating"]


def fit_model() -> dict:
    """Fit the Poisson model from the committed training CSV.

    Training only takes a couple of seconds, so the deployed app does this
    once at startup (cached) instead of loading a pickled model -- pickles
    are fragile across scikit-learn versions/Python versions, which is
    exactly what broke the first deploy attempt.
    """
    df = pd.read_csv(DATA_DIR / "training_data.csv")
    X = build_features(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        home_model.fit(X, df["home_score"])
        away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        away_model.fit(X, df["away_score"])

    return {"home_model": home_model, "away_model": away_model}


def outcome_probabilities(home_lambda: np.ndarray, away_lambda: np.ndarray) -> np.ndarray:
    """Return an (n, 3) array of [P(home win), P(draw), P(away win)]."""
    goals = np.arange(MAX_GOALS + 1)
    home_pmf = poisson.pmf(goals[None, :], home_lambda[:, None])  # (n, MAX_GOALS+1)
    away_pmf = poisson.pmf(goals[None, :], away_lambda[:, None])

    joint = home_pmf[:, :, None] * away_pmf[:, None, :]  # (n, home_goals, away_goals)
    home_win = np.triu(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), k=1).T  # home > away
    draw = np.eye(MAX_GOALS + 1)
    away_win = np.triu(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), k=1)  # away > home

    p_home = (joint * home_win).sum(axis=(1, 2))
    p_draw = (joint * draw).sum(axis=(1, 2))
    p_away = (joint * away_win).sum(axis=(1, 2))
    probs = np.stack([p_home, p_draw, p_away], axis=1)
    return probs / probs.sum(axis=1, keepdims=True)  # renormalize for the MAX_GOALS truncation


def predict_match(home_team: str, away_team: str, neutral: bool, latest_elo: pd.Series, model: dict) -> dict:
    row = pd.DataFrame(
        [{"home_elo": latest_elo[home_team], "away_elo": latest_elo[away_team], "neutral": neutral}]
    )
    X = build_features(row)

    home_lambda = model["home_model"].predict(X)[0]
    away_lambda = model["away_model"].predict(X)[0]
    probs = outcome_probabilities(np.array([home_lambda]), np.array([away_lambda]))[0]

    return {
        "home_elo": latest_elo[home_team],
        "away_elo": latest_elo[away_team],
        "home_goals": home_lambda,
        "away_goals": away_lambda,
        "p_home_win": probs[0],
        "p_draw": probs[1],
        "p_away_win": probs[2],
    }
