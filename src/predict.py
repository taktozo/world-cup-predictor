"""Try the baseline model on an actual matchup.

Usage:
    python predict.py "Argentina" "France" --neutral
    python predict.py "Brazil" "Germany"
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import build_features
from train_baseline import MAX_GOALS, outcome_probabilities

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_latest_elo() -> pd.Series:
    elo = pd.read_csv(DATA_DIR / "eloratings.csv")
    elo["date"] = pd.to_datetime(elo["date"], format="mixed")
    elo["team"] = elo["team"].str.replace("\xa0", " ", regex=False)
    latest = elo.sort_values("date").groupby("team").tail(1)
    return latest.set_index("team")["rating"]


def predict(home_team: str, away_team: str, neutral: bool) -> None:
    latest_elo = load_latest_elo()
    for team in (home_team, away_team):
        if team not in latest_elo.index:
            raise SystemExit(f"No Elo rating found for {team!r}. Check spelling against eloratings.csv.")

    row = pd.DataFrame(
        [{"home_elo": latest_elo[home_team], "away_elo": latest_elo[away_team], "neutral": neutral}]
    )
    X = build_features(row)

    saved = joblib.load(MODELS_DIR / "baseline_poisson.joblib")
    home_lambda = saved["home_model"].predict(X)
    away_lambda = saved["away_model"].predict(X)
    probs = outcome_probabilities(home_lambda, away_lambda)[0]

    print(f"{home_team} (Elo {latest_elo[home_team]:.0f}) vs {away_team} (Elo {latest_elo[away_team]:.0f})")
    print(f"Neutral venue: {neutral}")
    print(f"\nExpected score: {home_team} {home_lambda[0]:.2f} - {away_lambda[0]:.2f} {away_team}")
    print(f"\nP({home_team} win) = {probs[0]:.1%}")
    print(f"P(draw)            = {probs[1]:.1%}")
    print(f"P({away_team} win) = {probs[2]:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--neutral", action="store_true", help="Set if played at a neutral venue")
    args = parser.parse_args()
    predict(args.home_team, args.away_team, args.neutral)


if __name__ == "__main__":
    main()
