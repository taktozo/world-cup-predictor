"""Export a lean training set for the deployed app's live model (no odds/
referee -- see features.py for why). Committed to the repo so the app can
train itself at startup, same pattern as the international pipeline.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"

REQUIRED_COLUMNS = [
    "home_elo", "away_elo", "home_form_goals_for", "home_form_goals_against",
    "away_form_goals_for", "away_form_goals_against", "h2h_diff",
    "home_rest_days", "away_rest_days", "home_squad_value", "away_squad_value",
]
LEAN_COLUMNS = ["Date", *REQUIRED_COLUMNS, "FTHG", "FTAG", "League"]


def main() -> None:
    df = pd.read_csv(DATA_DIR / "matches_with_rest.csv", low_memory=False, parse_dates=["Date"])
    df = df.dropna(subset=REQUIRED_COLUMNS)
    lean = df[LEAN_COLUMNS]

    out_path = DATA_DIR / "training_data.csv"
    lean.to_csv(out_path, index=False)
    print(f"Saved {len(lean):,} rows to {out_path}")
    print(f"Date range: {lean['Date'].min().date()} to {lean['Date'].max().date()}")


if __name__ == "__main__":
    main()
