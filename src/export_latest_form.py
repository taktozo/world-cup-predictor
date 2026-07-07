"""Export each team's current rolling form (last N completed matches).

Small committed snapshot, same role as latest_elo.csv: lets the deployed
app look up a team's current form without needing the full match history.
"""

from pathlib import Path

import pandas as pd

from add_form_features import FORM_WINDOW, _team_match_log

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "results_with_elo.csv", parse_dates=["date"])
    matches = matches.dropna(subset=["home_score", "away_score"])  # only completed matches

    log = _team_match_log(matches).sort_values(["team", "date"])
    last_n = log.groupby("team").tail(FORM_WINDOW)
    latest = last_n.groupby("team")[["goals_for", "goals_against"]].mean().reset_index()
    latest.columns = ["team", "form_goals_for", "form_goals_against"]
    latest = latest.sort_values("team").reset_index(drop=True)

    out_path = DATA_DIR / "latest_form.csv"
    latest.to_csv(out_path, index=False)
    print(f"Saved {len(latest)} teams to {out_path}")


if __name__ == "__main__":
    main()
