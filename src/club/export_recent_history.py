"""Export small, committed match-level history for display purposes:
- standings.csv: final table of the last completed season, per league
- recent_matches.csv: each team's last 10 results (for a "recent form" view)
- recent_h2h.csv: each pair's last 5 meetings (for a head-to-head view)

Separate from the aggregate snapshots (latest_elo.csv etc.) used for
prediction -- these are for showing the user context, not model inputs.

build_match_log()/build_h2h_log() are also imported directly by the
Streamlit page to fold in matches fetched live by live_refresh.py, so the
displayed "recent form"/"last 5 meetings" tables can reflect new results
without needing this whole script re-run.
"""

from pathlib import Path

import pandas as pd

from standings import compute_standings

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
RECENT_FORM_WINDOW = 10
RECENT_H2H_WINDOW = 5


def export_standings(matches: pd.DataFrame) -> None:
    latest_season_idx = matches.groupby("League")["Date"].idxmax()
    tables = []
    for league, idx in zip(matches.loc[latest_season_idx, "League"], latest_season_idx):
        season = matches.loc[idx, "Season"]
        season_matches = matches[(matches["League"] == league) & (matches["Season"] == season)]
        table = compute_standings(season_matches)
        table["league"] = league
        table["season"] = season
        tables.append(table)

    combined = pd.concat(tables).reset_index()
    combined.to_csv(DATA_DIR / "standings.csv", index=False)
    print(f"standings.csv: {len(combined)} rows across {combined['league'].nunique()} leagues")


def build_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (both home and away appearances), full
    history -- not truncated to a recency window, so callers can combine
    multiple sources before taking the most recent N."""
    if matches.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "goals_for", "goals_against", "venue", "result"])

    home = matches[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against"]
    home["venue"] = "H"

    away = matches[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG"]].copy()
    away.columns = ["date", "team", "opponent", "goals_for", "goals_against"]
    away["venue"] = "A"

    long = pd.concat([home, away]).sort_values(["team", "date"])
    long["result"] = pd.Series(
        pd.cut(
            long["goals_for"] - long["goals_against"],
            bins=[-float("inf"), -0.5, 0.5, float("inf")],
            labels=["L", "D", "W"],
        ),
        index=long.index,
    )
    return long


def build_h2h_log(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match with a team_a/team_b pair key (alphabetical), full
    history -- not truncated, same reasoning as build_match_log()."""
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "team_a", "team_b"]
    if matches.empty:
        return pd.DataFrame(columns=cols)

    pair_key = [tuple(sorted(pair)) for pair in zip(matches["HomeTeam"], matches["AwayTeam"])]
    matches = matches.assign(pair_key=pair_key)

    recent = matches[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "pair_key"]].rename(
        columns={"Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team", "FTHG": "home_score", "FTAG": "away_score"}
    )
    recent[["team_a", "team_b"]] = pd.DataFrame(recent["pair_key"].tolist(), index=recent.index)
    return recent.drop(columns="pair_key")[cols]


def export_recent_matches(matches: pd.DataFrame) -> None:
    log = build_match_log(matches)
    recent = log.groupby("team").tail(RECENT_FORM_WINDOW)
    recent.to_csv(DATA_DIR / "recent_matches.csv", index=False)
    print(f"recent_matches.csv: {len(recent)} rows for {recent['team'].nunique()} teams")


def export_recent_h2h(matches: pd.DataFrame) -> None:
    log = build_h2h_log(matches).sort_values("date")
    recent = log.groupby(["team_a", "team_b"]).tail(RECENT_H2H_WINDOW)
    recent.to_csv(DATA_DIR / "recent_h2h.csv", index=False)
    print(f"recent_h2h.csv: {len(recent)} rows across {recent[['team_a','team_b']].drop_duplicates().shape[0]} pairs")


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_with_referee.csv", low_memory=False, parse_dates=["Date"])
    export_standings(matches)
    export_recent_matches(matches)
    export_recent_h2h(matches)


if __name__ == "__main__":
    main()
