"""Compute club Elo ratings from scratch, following the same formula used
by eloratings.net for international football (R_new = R_old + K*G*(W-We)).

clubelo.com would normally provide this for free, but its API is currently
returning 502 errors -- computing our own avoids depending on a third-party
service that may be down or discontinued, and reuses logic we already
understand from the international model.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"

STARTING_RATING = 1500.0
K = 20  # flat weight -- all matches are regular domestic league games
HOME_ADVANTAGE = 100


def _expected_result(rating_diff: float) -> float:
    return 1 / (10 ** (-rating_diff / 400) + 1)


def _goal_margin_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("Date").reset_index(drop=True)
    ratings = defaultdict(lambda: STARTING_RATING)

    home_elo = np.empty(len(matches))
    away_elo = np.empty(len(matches))

    for i, row in enumerate(matches.itertuples()):
        home, away = row.HomeTeam, row.AwayTeam
        r_home, r_away = ratings[home], ratings[away]
        home_elo[i] = r_home
        away_elo[i] = r_away

        goal_diff = row.FTHG - row.FTAG
        if goal_diff > 0:
            w = 1.0
        elif goal_diff == 0:
            w = 0.5
        else:
            w = 0.0

        we = _expected_result(r_home - r_away + HOME_ADVANTAGE)
        g = _goal_margin_multiplier(abs(goal_diff))
        delta = K * g * (w - we)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    matches["home_elo"] = home_elo
    matches["away_elo"] = away_elo
    return matches


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_raw.csv", low_memory=False)
    matches = matches.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    matches["Date"] = pd.to_datetime(matches["Date"], format="mixed", dayfirst=True)

    merged = compute_elo(matches)

    print(merged[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "home_elo", "away_elo"]].tail(5))

    out_path = DATA_DIR / "matches_with_elo.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
