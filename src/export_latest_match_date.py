"""Export each team's most recent match date -- needed to compute rest
days for a hypothetical or real upcoming fixture at prediction time.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "results_with_rest.csv", parse_dates=["date"])
    matches = matches.dropna(subset=["home_score", "away_score"])  # only completed matches

    long = pd.concat(
        [
            matches[["date", "home_team"]].rename(columns={"home_team": "team"}),
            matches[["date", "away_team"]].rename(columns={"away_team": "team"}),
        ]
    )
    latest = long.sort_values("date").groupby("team").tail(1)
    latest.columns = ["last_match_date", "team"]
    latest = latest[["team", "last_match_date"]].sort_values("team").reset_index(drop=True)

    out_path = DATA_DIR / "latest_match_date.csv"
    latest.to_csv(out_path, index=False)
    print(f"Saved {len(latest)} teams to {out_path}")


if __name__ == "__main__":
    main()
