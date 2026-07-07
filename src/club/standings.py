"""Compute a league standings table from a set of matches.

Shared between the committed export (last completed season) and the live
page (recomputing the in-progress season fresh from live_refresh's fetched
matches, since that's small enough -- at most 380 rows -- to redo from
scratch each time rather than needing incremental updates).
"""

import pandas as pd


def compute_standings(matches: pd.DataFrame) -> pd.DataFrame:
    """matches needs HomeTeam, AwayTeam, FTHG, FTAG columns."""
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
    table["goal_diff"] = table["goals_for"] - table["goals_against"]
    table["points"] = table["wins"] * 3 + table["draws"]

    table = table.sort_values(["points", "goal_diff", "goals_for"], ascending=False)
    table = table.reset_index()
    table.index = table.index + 1
    table.index.name = "position"
    return table
