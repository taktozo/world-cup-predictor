"""Export small "current state" snapshots per club -- Elo, form, h2h,
squad value -- for the live dashboard to look up when a user picks a
matchup. Same role as the international pipeline's latest_elo.csv etc.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
FORM_WINDOW = 10


def export_latest_elo(matches: pd.DataFrame) -> None:
    long = pd.concat(
        [
            matches[["Date", "HomeTeam", "home_elo"]].rename(columns={"HomeTeam": "team", "home_elo": "rating"}),
            matches[["Date", "AwayTeam", "away_elo"]].rename(columns={"AwayTeam": "team", "away_elo": "rating"}),
        ]
    )
    latest = long.sort_values("Date").groupby("team").tail(1)[["team", "rating"]]
    latest.sort_values("team").to_csv(DATA_DIR / "latest_elo.csv", index=False)
    print(f"latest_elo.csv: {len(latest)} teams")


def export_latest_form(matches: pd.DataFrame) -> None:
    home = matches[["Date", "HomeTeam", "FTHG", "FTAG"]].copy()
    home.columns = ["date", "team", "goals_for", "goals_against"]
    away = matches[["Date", "AwayTeam", "FTAG", "FTHG"]].copy()
    away.columns = ["date", "team", "goals_for", "goals_against"]
    log = pd.concat([home, away]).sort_values(["team", "date"])

    last_n = log.groupby("team").tail(FORM_WINDOW)
    latest = last_n.groupby("team")[["goals_for", "goals_against"]].mean().reset_index()
    latest.columns = ["team", "form_goals_for", "form_goals_against"]
    latest.sort_values("team").to_csv(DATA_DIR / "latest_form.csv", index=False)
    print(f"latest_form.csv: {len(latest)} teams")


def export_latest_h2h(matches: pd.DataFrame) -> None:
    pair_key = [tuple(sorted(pair)) for pair in zip(matches["HomeTeam"], matches["AwayTeam"])]
    team_a = pd.Series([p[0] for p in pair_key], index=matches.index)
    team_b = pd.Series([p[1] for p in pair_key], index=matches.index)

    is_team_a_home = matches["HomeTeam"] == team_a
    goal_diff_team_a = matches["FTHG"] - matches["FTAG"]
    goal_diff_team_a = goal_diff_team_a.where(is_team_a_home, -goal_diff_team_a)

    summary = pd.DataFrame({"team_a": team_a, "team_b": team_b, "goal_diff_team_a": goal_diff_team_a})
    latest = summary.groupby(["team_a", "team_b"]).agg(
        mean_diff_team_a=("goal_diff_team_a", "mean"), count=("goal_diff_team_a", "size")
    )
    latest = latest.reset_index().sort_values(["team_a", "team_b"])
    latest.to_csv(DATA_DIR / "latest_h2h.csv", index=False)
    print(f"latest_h2h.csv: {len(latest):,} team pairs")


def export_latest_squad_value(matches: pd.DataFrame) -> None:
    long = pd.concat(
        [
            matches[["Date", "HomeTeam", "home_squad_value"]].rename(
                columns={"HomeTeam": "team", "home_squad_value": "squad_value"}
            ),
            matches[["Date", "AwayTeam", "away_squad_value"]].rename(
                columns={"AwayTeam": "team", "away_squad_value": "squad_value"}
            ),
        ]
    )
    long = long.dropna(subset=["squad_value"])
    latest = long.sort_values("Date").groupby("team").tail(1)[["team", "squad_value"]]
    latest.sort_values("team").to_csv(DATA_DIR / "latest_squad_value.csv", index=False)
    print(f"latest_squad_value.csv: {len(latest)} teams")


def export_latest_match_date(matches: pd.DataFrame) -> None:
    long = pd.concat(
        [
            matches[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
            matches[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
        ]
    )
    latest = long.sort_values("Date").groupby("team").tail(1)
    latest.columns = ["last_match_date", "team"]
    latest[["team", "last_match_date"]].sort_values("team").to_csv(DATA_DIR / "latest_match_date.csv", index=False)
    print(f"latest_match_date.csv: {len(latest)} teams")


def export_team_league(matches: pd.DataFrame) -> None:
    long = pd.concat(
        [
            matches[["Date", "HomeTeam", "League"]].rename(columns={"HomeTeam": "team"}),
            matches[["Date", "AwayTeam", "League"]].rename(columns={"AwayTeam": "team"}),
        ]
    )
    latest = long.sort_values("Date").groupby("team").tail(1)[["team", "League"]]
    latest.columns = ["team", "league"]
    latest.sort_values("team").to_csv(DATA_DIR / "team_league.csv", index=False)
    print(f"team_league.csv: {len(latest)} teams")


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_with_rest.csv", low_memory=False, parse_dates=["Date"])
    export_latest_elo(matches)
    export_latest_form(matches)
    export_latest_h2h(matches)
    export_latest_squad_value(matches)
    export_latest_match_date(matches)
    export_team_league(matches)


if __name__ == "__main__":
    main()
