"""Train and evaluate the club Poisson model, benchmarked against the
bookmaker odds themselves -- since we have odds as a feature, we can check
whether the model's predictions add anything beyond what the market already
prices in, not just whether it beats a naive baseline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from features import FEATURE_COLUMNS, build_features
from inference import fit_model, outcome_probabilities

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
REQUIRED_COLUMNS = [
    "home_elo", "away_elo", "home_form_goals_for", "home_form_goals_against",
    "away_form_goals_for", "away_form_goals_against", "h2h_diff",
    "odds_p_home", "odds_p_draw", "odds_p_away", "home_squad_value",
    "away_squad_value", "referee_avg_cards", "FTHG", "FTAG",
]


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "matches_with_referee.csv", low_memory=False, parse_dates=["Date"])
    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df.sort_values("Date").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_frac: float = 0.15):
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_date = df["Date"].iloc[cutoff_idx]
    return df[df["Date"] < cutoff_date], df[df["Date"] >= cutoff_date], cutoff_date


def actual_outcome_labels(df: pd.DataFrame) -> np.ndarray:
    conditions = [df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]]
    return np.select(conditions, ["home", "draw"], default="away")


def main() -> None:
    df = load_training_data()
    train, test, cutoff_date = time_split(df)
    print(f"Train: {len(train):,} matches (through {train['Date'].max().date()})")
    print(f"Test:  {len(test):,} matches (from {cutoff_date.date()} onward)")

    model = fit_model(train)
    X_test = build_features(test)
    home_pred = model["home_model"].predict(X_test)
    away_pred = model["away_model"].predict(X_test)

    print(f"\nHome goals MAE: {mean_absolute_error(test['FTHG'], home_pred):.3f}")
    print(f"Away goals MAE: {mean_absolute_error(test['FTAG'], away_pred):.3f}")

    probs = outcome_probabilities(home_pred, away_pred)
    y_true = actual_outcome_labels(test)
    y_pred = np.array(["home", "draw", "away"])[probs.argmax(axis=1)]

    print(f"\nModel outcome accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Model log-loss: {log_loss(y_true, probs, labels=['home', 'draw', 'away']):.3f}")

    naive_pred = np.full(len(test), "home")
    print(f"Naive 'always home win' accuracy: {accuracy_score(y_true, naive_pred):.3f}")

    odds_probs = test[["odds_p_home", "odds_p_draw", "odds_p_away"]].to_numpy()
    odds_pred = np.array(["home", "draw", "away"])[odds_probs.argmax(axis=1)]
    print(f"\nBookmaker-odds accuracy:  {accuracy_score(y_true, odds_pred):.3f}")
    print(f"Bookmaker-odds log-loss:  {log_loss(y_true, odds_probs, labels=['home', 'draw', 'away']):.3f}")


if __name__ == "__main__":
    main()
