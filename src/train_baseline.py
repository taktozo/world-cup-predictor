"""Baseline model: independent Poisson regressions for home/away goals.

Each side's goal count is modeled as Poisson(lambda), with lambda a
log-linear function of Elo ratings and venue. Match outcome probabilities
(home win / draw / away win) are derived by summing the joint distribution
over a home x away scoreline grid, assuming independence between sides --
the standard baseline approach in football score prediction.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, build_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_GOALS = 10  # grid cutoff for outcome-probability summation


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results_with_elo.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_elo", "away_elo", "home_score", "away_score"])
    return df.sort_values("date").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_frac: float = 0.15):
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_date = df["date"].iloc[cutoff_idx]
    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]
    return train, test, cutoff_date


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


def actual_outcome_labels(df: pd.DataFrame) -> np.ndarray:
    conditions = [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]]
    return np.select(conditions, ["home", "draw"], default="away")


def main() -> None:
    df = load_training_data()
    train, test, cutoff_date = time_split(df)
    print(f"Train: {len(train):,} matches (through {train['date'].max().date()})")
    print(f"Test:  {len(test):,} matches (from {cutoff_date.date()} onward)")

    X_train, X_test = build_features(train), build_features(test)

    def make_model():
        return make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        home_model = make_model().fit(X_train, train["home_score"])
        away_model = make_model().fit(X_train, train["away_score"])

    home_pred = home_model.predict(X_test)
    away_pred = away_model.predict(X_test)

    print(f"\nHome goals MAE: {mean_absolute_error(test['home_score'], home_pred):.3f}")
    print(f"Away goals MAE: {mean_absolute_error(test['away_score'], away_pred):.3f}")

    probs = outcome_probabilities(home_pred, away_pred)
    y_true = actual_outcome_labels(test)
    y_pred = np.array(["home", "draw", "away"])[probs.argmax(axis=1)]

    print(f"\nOutcome accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Outcome log-loss: {log_loss(y_true, probs, labels=['home', 'draw', 'away']):.3f}")

    naive_pred = np.full(len(test), "home")
    print(f"Naive 'always home win' accuracy: {accuracy_score(y_true, naive_pred):.3f}")

    wc_mask = test["tournament"] == "FIFA World Cup"
    if wc_mask.any():
        print(f"\nWorld Cup test matches: {wc_mask.sum()}")
        print(f"World Cup outcome accuracy: {accuracy_score(y_true[wc_mask.values], y_pred[wc_mask.values]):.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"home_model": home_model, "away_model": away_model, "feature_columns": FEATURE_COLUMNS}, MODELS_DIR / "baseline_poisson.joblib")
    print(f"\nSaved model to {MODELS_DIR / 'baseline_poisson.joblib'}")


if __name__ == "__main__":
    main()
