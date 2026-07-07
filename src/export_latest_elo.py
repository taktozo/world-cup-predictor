"""Export a small snapshot of each team's most recent Elo rating.

The full eloratings.csv (~6.7k rows, all rating history) stays out of
git per our data policy. This snapshot (one row per team) is tiny and
gets committed, since the deployed dashboard needs it to populate the
team dropdowns and look up current ratings without live Kaggle access.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    elo = pd.read_csv(DATA_DIR / "eloratings.csv")
    elo["date"] = pd.to_datetime(elo["date"], format="mixed")
    elo["team"] = elo["team"].str.replace("\xa0", " ", regex=False)

    latest = elo.sort_values("date").groupby("team").tail(1)
    latest = latest[["team", "rating", "date"]].rename(columns={"date": "as_of_date"})
    latest = latest.sort_values("team").reset_index(drop=True)

    out_path = DATA_DIR / "latest_elo.csv"
    latest.to_csv(out_path, index=False)
    print(f"Saved {len(latest)} teams to {out_path}")


if __name__ == "__main__":
    main()
