"""Export small, committed match-level history for display purposes:
- recent_matches.csv: each team's last 10 results (for a "recent form" view)
- recent_h2h.csv: each pair's last 5 meetings (for a head-to-head view)

No standings/league-table export here (unlike the club pipeline) --
international football (World Cup, qualifiers, friendlies) doesn't have
a season-long league table the way EPL/La Liga do, so that concept
doesn't translate.

build_match_log()/build_h2h_log() are separate from the truncated exports
so they're reusable if we ever want to fold in live-fetched matches, same
pattern as the club pipeline's export_recent_history.py.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RECENT_FORM_WINDOW = 10
RECENT_H2H_WINDOW = 5


def build_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (both home and away appearances), full
    history -- not truncated to a recency window."""
    if matches.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "goals_for", "goals_against", "venue", "result"])

    home = matches[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against"]
    home["venue"] = "H"

    away = matches[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
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
    """One row per match with a team_a/team_b pair key (alphabetical), full history."""
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "team_a", "team_b"]
    if matches.empty:
        return pd.DataFrame(columns=cols)

    pair_key = [tuple(sorted(pair)) for pair in zip(matches["home_team"], matches["away_team"])]
    matches = matches.assign(pair_key=pair_key)

    recent = matches[["date", "home_team", "away_team", "home_score", "away_score", "pair_key"]].copy()
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
    matches = pd.read_csv(DATA_DIR / "results_with_rest.csv", parse_dates=["date"])
    matches = matches.dropna(subset=["home_score", "away_score"])  # only completed matches
    export_recent_matches(matches)
    export_recent_h2h(matches)


if __name__ == "__main__":
    main()
