"""Export a lean training set for the deployed app to fit its model from.

Only the columns the model actually needs (Elo ratings, venue, goals) --
this is committed to the repo (unlike the raw Kaggle CSVs or the full
merged results_with_elo.csv) so the deployed dashboard can train itself
at startup without needing a live Kaggle download or a pickled model
(which is fragile across scikit-learn versions).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "results_with_elo.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_elo", "away_elo", "home_score", "away_score"])
    lean = df[["date", "home_elo", "away_elo", "neutral", "home_score", "away_score", "tournament"]]

    out_path = DATA_DIR / "training_data.csv"
    lean.to_csv(out_path, index=False)
    print(f"Saved {len(lean):,} rows to {out_path}")


if __name__ == "__main__":
    main()
