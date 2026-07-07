"""Shared feature construction for both training and live inference.

Kept separate from merge_elo.py so the same feature-building logic can be
reused when predicting a single upcoming fixture, not just a historical
results table.
"""

import pandas as pd

FEATURE_COLUMNS = ["home_elo", "away_elo", "elo_diff", "neutral"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given rows with home_elo/away_elo/neutral, return the model feature matrix."""
    features = pd.DataFrame(index=df.index)
    features["home_elo"] = df["home_elo"]
    features["away_elo"] = df["away_elo"]
    features["elo_diff"] = df["home_elo"] - df["away_elo"]
    features["neutral"] = df["neutral"].astype(int)
    return features[FEATURE_COLUMNS]
