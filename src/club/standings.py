"""Compute a league standings table from a set of matches.

Used to build the live, current-season table -- since it needs to show
every team in the league at 0 played/0 points before a ball is kicked
(not last season's final table), it takes the season's full team roster
separately from the played-matches data, so teams with no results yet
still appear correctly at zero rather than being absent.
"""

import pandas as pd


def compute_standings(matches: pd.DataFrame, teams: list[str] | None = None) -> pd.DataFrame:
    """matches needs HomeTeam, AwayTeam, FTHG, FTAG columns (can be empty).

    If `teams` is given, every one of those teams appears in the output
    (zero-filled if they haven't played yet) -- otherwise only teams that
    appear in `matches` are included.
    """
    home = matches[["HomeTeam", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "team", "FTHG": "goals_for", "FTAG": "goals_against"}
    )
    away = matches[["AwayTeam", "FTAG", "FTHG"]].rename(
        columns={"AwayTeam": "team", "FTAG": "goals_for", "FTHG": "goals_against"}
    )
    long = pd.concat([home, away], ignore_index=True)

    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["draw"] = (long["goals_for"] == long["goals_against"]).astype(int)
    long["loss"] = (long["goals_for"] < long["goals_against"]).astype(int)

    table = long.groupby("team").agg(
        played=("goals_for", "size"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        losses=("loss", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )

    if teams is not None:
        table = table.reindex(teams, fill_value=0)

    table["goal_diff"] = table["goals_for"] - table["goals_against"]
    table["points"] = table["wins"] * 3 + table["draws"]

    table = table.sort_values(["points", "goal_diff", "goals_for"], ascending=False)
    table = table.reset_index(names="team")
    table.index = table.index + 1
    table.index.name = "position"
    return table
