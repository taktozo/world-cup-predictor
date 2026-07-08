"""Shared inference helpers for the club model: fit and score a matchup.

Same joint-Poisson-grid approach as the international model
(src/inference.py) -- duplicated rather than imported since the two
pipelines live in separate script directories with independent sys.path
setups, matching how each pipeline is otherwise self-contained.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from club_add_rest_days import DEFAULT_REST_DAYS, MAX_REST_DAYS
from club_features import build_features, build_live_features

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
MAX_GOALS = 10
OVER_UNDER_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]


def fit_model(training_data: pd.DataFrame) -> dict:
    """Full research model (includes odds + referee) -- used only for
    train_baseline.py's offline evaluation against the bookmaker benchmark."""
    X = build_features(training_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        home_model.fit(X, training_data["FTHG"])
        away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        away_model.fit(X, training_data["FTAG"])
    return {"home_model": home_model, "away_model": away_model}


def fit_live_model() -> dict:
    """Live/deployable model (no odds, no referee -- neither is knowable for
    an arbitrary hypothetical matchup), trained from the committed lean CSV."""
    df = pd.read_csv(DATA_DIR / "training_data.csv")
    X = build_live_features(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        home_model.fit(X, df["FTHG"])
        away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        away_model.fit(X, df["FTAG"])
    return {"home_model": home_model, "away_model": away_model}


def load_latest_elo() -> pd.Series:
    latest = pd.read_csv(DATA_DIR / "latest_elo.csv")
    return latest.set_index("team")["rating"]


def load_latest_form() -> pd.DataFrame:
    latest = pd.read_csv(DATA_DIR / "latest_form.csv")
    return latest.set_index("team")


def load_latest_h2h() -> pd.DataFrame:
    latest = pd.read_csv(DATA_DIR / "latest_h2h.csv")
    latest.index = pd.MultiIndex.from_arrays([latest["team_a"], latest["team_b"]])
    return latest


def load_latest_squad_value() -> pd.Series:
    latest = pd.read_csv(DATA_DIR / "latest_squad_value.csv")
    return latest.set_index("team")["squad_value"]


def load_latest_match_date() -> pd.Series:
    latest = pd.read_csv(DATA_DIR / "latest_match_date.csv", parse_dates=["last_match_date"])
    return latest.set_index("team")["last_match_date"]


def rest_days_for_team(team: str, match_date: pd.Timestamp, latest_match_date: pd.Series) -> float:
    if team not in latest_match_date.index or pd.isna(latest_match_date[team]):
        return DEFAULT_REST_DAYS
    days = (match_date - latest_match_date[team]).days
    return float(np.clip(days, 1, MAX_REST_DAYS))


H2H_SHRINKAGE_K = 4


def h2h_diff_for_pair(home_team: str, away_team: str, latest_h2h: pd.DataFrame) -> float:
    pair_key = tuple(sorted([home_team, away_team]))
    if pair_key not in latest_h2h.index:
        return 0.0
    row = latest_h2h.loc[pair_key]
    sign = 1 if home_team == pair_key[0] else -1
    return sign * row["mean_diff_team_a"] * row["count"] / (row["count"] + H2H_SHRINKAGE_K)


def predict_match(
    home_team: str,
    away_team: str,
    latest_elo: pd.Series,
    latest_form: pd.DataFrame,
    latest_h2h: pd.DataFrame,
    latest_squad_value: pd.Series,
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
                "home_form_goals_for": latest_form.loc[home_team, "form_goals_for"],
                "home_form_goals_against": latest_form.loc[home_team, "form_goals_against"],
                "away_form_goals_for": latest_form.loc[away_team, "form_goals_for"],
                "away_form_goals_against": latest_form.loc[away_team, "form_goals_against"],
                "h2h_diff": h2h_diff_for_pair(home_team, away_team, latest_h2h),
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
                "home_squad_value": latest_squad_value[home_team],
                "away_squad_value": latest_squad_value[away_team],
            }
        ]
    )
    X = build_live_features(row)

    home_lambda = np.array([model["home_model"].predict(X)[0]])
    away_lambda = np.array([model["away_model"].predict(X)[0]])
    joint = joint_score_grid(home_lambda, away_lambda)

    outcome_probs = outcome_probabilities_from_grid(joint)[0]
    btts_prob = btts_probability_from_grid(joint)[0]
    over_under = over_under_probabilities_from_grid(joint)

    return {
        "home_elo": latest_elo[home_team],
        "away_elo": latest_elo[away_team],
        "home_squad_value": latest_squad_value[home_team],
        "away_squad_value": latest_squad_value[away_team],
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


def joint_score_grid(home_lambda: np.ndarray, away_lambda: np.ndarray) -> np.ndarray:
    goals = np.arange(MAX_GOALS + 1)
    home_pmf = poisson.pmf(goals[None, :], home_lambda[:, None])
    away_pmf = poisson.pmf(goals[None, :], away_lambda[:, None])
    return home_pmf[:, :, None] * away_pmf[:, None, :]


def outcome_probabilities_from_grid(joint: np.ndarray) -> np.ndarray:
    n = joint.shape[1]
    home_win = np.triu(np.ones((n, n)), k=1).T
    draw = np.eye(n)
    away_win = np.triu(np.ones((n, n)), k=1)

    p_home = (joint * home_win).sum(axis=(1, 2))
    p_draw = (joint * draw).sum(axis=(1, 2))
    p_away = (joint * away_win).sum(axis=(1, 2))
    probs = np.stack([p_home, p_draw, p_away], axis=1)
    return probs / probs.sum(axis=1, keepdims=True)


def outcome_probabilities(home_lambda: np.ndarray, away_lambda: np.ndarray) -> np.ndarray:
    return outcome_probabilities_from_grid(joint_score_grid(home_lambda, away_lambda))


def btts_probability_from_grid(joint: np.ndarray) -> np.ndarray:
    total = joint.sum(axis=(1, 2))
    p_yes = joint[:, 1:, 1:].sum(axis=(1, 2))
    return p_yes / total


def over_under_probabilities_from_grid(joint: np.ndarray, lines=OVER_UNDER_LINES) -> dict:
    n = joint.shape[1]
    total_goals = np.add.outer(np.arange(n), np.arange(n))
    total_prob = joint.sum(axis=(1, 2))
    return {line: (joint * (total_goals > line)[None, :, :]).sum(axis=(1, 2)) / total_prob for line in lines}
