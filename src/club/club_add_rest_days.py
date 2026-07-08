"""Add a rest-days feature: days since each team's previous match, as a
fatigue / fixture-congestion proxy. Only reflects rest between domestic
league matches in our data (doesn't see European/cup fixtures played
in between), so it's an approximation, not a perfect fatigue measure.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
DEFAULT_REST_DAYS = 7  # typical weekly gap -- used when a team has no prior match in our data
MAX_REST_DAYS = 120  # caps at a full summer break; beyond this it's a relegation/promotion gap, not real rest


def _team_match_dates(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "team"})
    home["orig_index"] = matches.index
    away = matches[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "team"})
    away["orig_index"] = matches.index

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "Date", "orig_index"])


def add_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    log = _team_match_dates(matches)
    log["previous_date"] = log.groupby("team")["Date"].shift(1)
    log["rest_days"] = (log["Date"] - log["previous_date"]).dt.days
    log["rest_days"] = log["rest_days"].fillna(DEFAULT_REST_DAYS).clip(upper=MAX_REST_DAYS)

    # concat (before sorting) put all home rows at index [0, len(matches)), away rows after -- see add_form_h2h.py
    log["side"] = ["home" if i < len(matches) else "away" for i in log.index]

    home_rest = log[log["side"] == "home"].set_index("orig_index")["rest_days"].rename("home_rest_days")
    away_rest = log[log["side"] == "away"].set_index("orig_index")["rest_days"].rename("away_rest_days")

    matches = matches.copy()
    matches = matches.join(home_rest).join(away_rest)
    matches["rest_days_diff"] = matches["home_rest_days"] - matches["away_rest_days"]
    return matches


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_with_referee.csv", low_memory=False, parse_dates=["Date"])
    matches = matches.sort_values("Date").reset_index(drop=True)
    merged = add_rest_days(matches)

    print(merged[["home_rest_days", "away_rest_days", "rest_days_diff"]].describe())

    out_path = DATA_DIR / "matches_with_rest.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
