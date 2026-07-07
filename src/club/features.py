"""Shared feature construction for the club model (training + live inference).

Two feature sets: the full research set (includes odds + referee, used to
evaluate against the bookmaker benchmark in train_baseline.py) and the live
set (drops odds/referee, since neither is knowable for an arbitrary
hypothetical matchup a dashboard user picks -- only for specific,
already-scheduled real fixtures). The deployed dashboard uses the live set.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_goals_for",
    "home_form_goals_against",
    "away_form_goals_for",
    "away_form_goals_against",
    "h2h_diff",
    "odds_p_home",
    "odds_p_draw",
    "odds_p_away",
    "squad_value_diff",
    "referee_avg_cards",
]

LIVE_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_goals_for",
    "home_form_goals_against",
    "away_form_goals_for",
    "away_form_goals_against",
    "h2h_diff",
    "squad_value_diff",
]


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["home_elo"] = df["home_elo"]
    features["away_elo"] = df["away_elo"]
    features["elo_diff"] = df["home_elo"] - df["away_elo"]
    features["home_form_goals_for"] = df["home_form_goals_for"]
    features["home_form_goals_against"] = df["home_form_goals_against"]
    features["away_form_goals_for"] = df["away_form_goals_for"]
    features["away_form_goals_against"] = df["away_form_goals_against"]
    features["h2h_diff"] = df["h2h_diff"]
    features["squad_value_diff"] = np.log1p(df["home_squad_value"]) - np.log1p(df["away_squad_value"])
    return features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = _base_features(df)
    features["odds_p_home"] = df["odds_p_home"]
    features["odds_p_draw"] = df["odds_p_draw"]
    features["odds_p_away"] = df["odds_p_away"]
    features["referee_avg_cards"] = df["referee_avg_cards"]
    return features[FEATURE_COLUMNS]


def build_live_features(df: pd.DataFrame) -> pd.DataFrame:
    features = _base_features(df)
    return features[LIVE_FEATURE_COLUMNS]
