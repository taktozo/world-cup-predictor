"""Shared feature construction for both training and live inference.

Kept separate from merge_elo.py so the same feature-building logic can be
reused when predicting a single upcoming fixture, not just a historical
results table.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "neutral",
    "home_form_goals_for",
    "home_form_goals_against",
    "away_form_goals_for",
    "away_form_goals_against",
    "h2h_diff",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given rows with elo/form/h2h/neutral columns, return the model feature matrix."""
    features = pd.DataFrame(index=df.index)
    features["home_elo"] = df["home_elo"]
    features["away_elo"] = df["away_elo"]
    features["elo_diff"] = df["home_elo"] - df["away_elo"]
    features["neutral"] = df["neutral"].astype(int)
    features["home_form_goals_for"] = df["home_form_goals_for"]
    features["home_form_goals_against"] = df["home_form_goals_against"]
    features["away_form_goals_for"] = df["away_form_goals_for"]
    features["away_form_goals_against"] = df["away_form_goals_against"]
    features["h2h_diff"] = df["h2h_diff"]
    return features[FEATURE_COLUMNS]
