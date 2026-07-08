"""Export a lean training set for the deployed app to fit its model from.

Only the columns the model actually needs (Elo, form, h2h, venue, goals) --
this is committed to the repo (unlike the raw Kaggle CSVs or the full
merged results_with_h2h.csv) so the deployed dashboard can train itself
at startup without needing a live Kaggle download or a pickled model
(which is fragile across scikit-learn versions).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FORM_COLUMNS = ["home_form_goals_for", "home_form_goals_against", "away_form_goals_for", "away_form_goals_against"]
LEAN_COLUMNS = [
    "date",
    "home_elo",
    "away_elo",
    "neutral",
    *FORM_COLUMNS,
    "h2h_diff",
    "home_rest_days",
    "away_rest_days",
    "home_score",
    "away_score",
    "tournament",
]


def main() -> None:
    df = pd.read_csv(DATA_DIR / "results_with_rest.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_elo", "away_elo", "home_score", "away_score", *FORM_COLUMNS])
    lean = df[LEAN_COLUMNS]

    out_path = DATA_DIR / "training_data.csv"
    lean.to_csv(out_path, index=False)
    print(f"Saved {len(lean):,} rows to {out_path}")


if __name__ == "__main__":
    main()
