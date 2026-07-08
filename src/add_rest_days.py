"""Add a rest-days feature: days since each team's previous match. For
international football this mostly reflects the calendar of international
breaks (typically several weeks to a few months apart) rather than
club-style weekly fixture congestion -- see the historical distribution
check in main() for why the cap differs from the club pipeline's.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_REST_DAYS = 60  # typical gap between international windows -- used when a team has no prior match
MAX_REST_DAYS = 365  # caps at a year; beyond this it's a genuine multi-year absence, not meaningful "rest"


def _team_match_dates(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["date", "home_team"]].rename(columns={"home_team": "team"})
    home["orig_index"] = matches.index
    away = matches[["date", "away_team"]].rename(columns={"away_team": "team"})
    away["orig_index"] = matches.index

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "date", "orig_index"])


def add_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    log = _team_match_dates(matches)
    log["previous_date"] = log.groupby("team")["date"].shift(1)
    log["rest_days"] = (log["date"] - log["previous_date"]).dt.days
    log["rest_days"] = log["rest_days"].fillna(DEFAULT_REST_DAYS).clip(upper=MAX_REST_DAYS)

    log["side"] = ["home" if i < len(matches) else "away" for i in log.index]
    home_rest = log[log["side"] == "home"].set_index("orig_index")["rest_days"].rename("home_rest_days")
    away_rest = log[log["side"] == "away"].set_index("orig_index")["rest_days"].rename("away_rest_days")

    matches = matches.copy()
    matches = matches.join(home_rest).join(away_rest)
    matches["rest_days_diff"] = matches["home_rest_days"] - matches["away_rest_days"]
    return matches


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "results_with_h2h.csv", parse_dates=["date"])
    matches = matches.sort_values("date").reset_index(drop=True)
    merged = add_rest_days(matches)

    print(merged[["home_rest_days", "away_rest_days", "rest_days_diff"]].describe())
    print()
    print("Percentiles:", merged["home_rest_days"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_dict())

    out_path = DATA_DIR / "results_with_rest.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
