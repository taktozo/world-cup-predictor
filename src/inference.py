"""Shared inference helpers: fit the model and score a matchup.

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

from add_rest_days import DEFAULT_REST_DAYS, MAX_REST_DAYS
from features import build_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_GOALS = 10  # grid cutoff for outcome-probability summation
OVER_UNDER_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
H2H_SHRINKAGE_K = 4  # head-to-head history is shrunk toward 0 as count/(count+K)


def load_latest_elo() -> pd.Series:
    latest = pd.read_csv(DATA_DIR / "latest_elo.csv")
    return latest.set_index("team")["rating"]


def load_latest_form() -> pd.DataFrame:
    latest = pd.read_csv(DATA_DIR / "latest_form.csv")
    return latest.set_index("team")


def load_latest_match_date() -> pd.Series:
    latest = pd.read_csv(DATA_DIR / "latest_match_date.csv", parse_dates=["last_match_date"])
    return latest.set_index("team")["last_match_date"]


def rest_days_for_team(team: str, match_date: pd.Timestamp, latest_match_date: pd.Series) -> float:
    if team not in latest_match_date.index or pd.isna(latest_match_date[team]):
        return DEFAULT_REST_DAYS
    days = (match_date - latest_match_date[team]).days
    return float(np.clip(days, 0, MAX_REST_DAYS))


def load_latest_h2h() -> pd.DataFrame:
    """Pairwise head-to-head snapshot: one row per pair of teams that have ever met."""
    latest = pd.read_csv(DATA_DIR / "latest_h2h.csv")
    latest.index = pd.MultiIndex.from_arrays([latest["team_a"], latest["team_b"]])
    return latest


def h2h_diff_for_pair(home_team: str, away_team: str, latest_h2h: pd.DataFrame) -> float:
    """Home team's shrunk historical goal-difference edge over away_team, or 0 if they've never met."""
    pair_key = tuple(sorted([home_team, away_team]))
    if pair_key not in latest_h2h.index:
        return 0.0

    row = latest_h2h.loc[pair_key]
    sign = 1 if home_team == pair_key[0] else -1
    return sign * row["mean_diff_team_a"] * row["count"] / (row["count"] + H2H_SHRINKAGE_K)


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


def joint_score_grid(home_lambda: np.ndarray, away_lambda: np.ndarray) -> np.ndarray:
    """Return an (n, MAX_GOALS+1, MAX_GOALS+1) joint P(home goals, away goals) grid.

    Assumes home/away goal counts are independent Poisson given the lambdas --
    the standard simplifying assumption for this style of baseline model.
    """
    goals = np.arange(MAX_GOALS + 1)
    home_pmf = poisson.pmf(goals[None, :], home_lambda[:, None])  # (n, MAX_GOALS+1)
    away_pmf = poisson.pmf(goals[None, :], away_lambda[:, None])
    return home_pmf[:, :, None] * away_pmf[:, None, :]


def outcome_probabilities(home_lambda: np.ndarray, away_lambda: np.ndarray) -> np.ndarray:
    """Return an (n, 3) array of [P(home win), P(draw), P(away win)]."""
    joint = joint_score_grid(home_lambda, away_lambda)
    return outcome_probabilities_from_grid(joint)


def outcome_probabilities_from_grid(joint: np.ndarray) -> np.ndarray:
    n = joint.shape[1]
    home_win = np.triu(np.ones((n, n)), k=1).T  # home > away
    draw = np.eye(n)
    away_win = np.triu(np.ones((n, n)), k=1)  # away > home

    p_home = (joint * home_win).sum(axis=(1, 2))
    p_draw = (joint * draw).sum(axis=(1, 2))
    p_away = (joint * away_win).sum(axis=(1, 2))
    probs = np.stack([p_home, p_draw, p_away], axis=1)
    return probs / probs.sum(axis=1, keepdims=True)  # renormalize for the MAX_GOALS truncation


def btts_probability_from_grid(joint: np.ndarray) -> np.ndarray:
    """P(both teams score >= 1)."""
    total = joint.sum(axis=(1, 2))
    p_yes = joint[:, 1:, 1:].sum(axis=(1, 2))
    return p_yes / total


def over_under_probabilities_from_grid(joint: np.ndarray, lines=OVER_UNDER_LINES) -> dict:
    """Return {line: P(total goals > line)} for each line, e.g. 2.5 -> P(3+ combined goals)."""
    n = joint.shape[1]
    total_goals = np.add.outer(np.arange(n), np.arange(n))  # (n, n), total goals per (home, away) cell
    total_prob = joint.sum(axis=(1, 2))

    return {line: (joint * (total_goals > line)[None, :, :]).sum(axis=(1, 2)) / total_prob for line in lines}


def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool,
    latest_elo: pd.Series,
    latest_form: pd.DataFrame,
    latest_h2h: pd.DataFrame,
    latest_match_date: pd.Series,
    model: dict,
    match_date: pd.Timestamp | None = None,
) -> dict:
    match_date = match_date or pd.Timestamp.now().normalize()
    home_rest_days = rest_days_for_team(home_team, match_date, latest_match_date)
    away_rest_days = rest_days_for_team(away_team, match_date, latest_match_date)

    row = pd.DataFrame(
        [
            {
                "home_elo": latest_elo[home_team],
                "away_elo": latest_elo[away_team],
                "neutral": neutral,
                "home_form_goals_for": latest_form.loc[home_team, "form_goals_for"],
                "home_form_goals_against": latest_form.loc[home_team, "form_goals_against"],
                "away_form_goals_for": latest_form.loc[away_team, "form_goals_for"],
                "away_form_goals_against": latest_form.loc[away_team, "form_goals_against"],
                "h2h_diff": h2h_diff_for_pair(home_team, away_team, latest_h2h),
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
            }
        ]
    )
    X = build_features(row)

    home_lambda = np.array([model["home_model"].predict(X)[0]])
    away_lambda = np.array([model["away_model"].predict(X)[0]])
    joint = joint_score_grid(home_lambda, away_lambda)

    outcome_probs = outcome_probabilities_from_grid(joint)[0]
    btts_prob = btts_probability_from_grid(joint)[0]
    over_under = over_under_probabilities_from_grid(joint)

    return {
        "home_elo": latest_elo[home_team],
        "away_elo": latest_elo[away_team],
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "home_goals": home_lambda[0],
        "away_goals": away_lambda[0],
        "p_home_win": outcome_probs[0],
        "p_draw": outcome_probs[1],
        "p_away_win": outcome_probs[2],
        "p_btts_yes": btts_prob,
        "p_btts_no": 1 - btts_prob,
        "over_under": {line: probs[0] for line, probs in over_under.items()},
    }
