"""Convert bookmaker odds into overround-normalized implied probabilities.

Prefers the multi-bookmaker average (AvgH/D/A, available from ~2020) and
falls back to Bet365's odds (B365H/D/A, available from ~2003) for older
seasons where the average columns don't exist. Matches before ~2003 have
no odds at all and are left NaN -- handled by dropna in export_training_data.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"


def add_odds_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    home_odds = matches["AvgH"].fillna(matches["B365H"])
    draw_odds = matches["AvgD"].fillna(matches["B365D"])
    away_odds = matches["AvgA"].fillna(matches["B365A"])

    inv_home = 1 / home_odds
    inv_draw = 1 / draw_odds
    inv_away = 1 / away_odds
    overround = inv_home + inv_draw + inv_away

    matches["odds_p_home"] = inv_home / overround
    matches["odds_p_draw"] = inv_draw / overround
    matches["odds_p_away"] = inv_away / overround
    return matches


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_with_features.csv", low_memory=False, parse_dates=["Date"])
    merged = add_odds_features(matches)

    coverage = 1 - merged["odds_p_home"].isna().mean()
    print(f"odds_p_home: {coverage:.1%} coverage")
    print(f"Earliest match with odds: {merged.loc[merged['odds_p_home'].notna(), 'Date'].min().date()}")

    out_path = DATA_DIR / "matches_with_odds.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
