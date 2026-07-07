"""Add rolling form and head-to-head features to club matches.

Same approach as the international pipeline (src/add_form_features.py,
src/add_h2h_features.py), adapted to football-data.co.uk's column names.
Kept as one file since club matches use a different schema than results.csv
and there's no shared code to reuse cleanly across the two pipelines.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
FORM_WINDOW = 10
H2H_SHRINKAGE_K = 4


def _team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["Date", "HomeTeam", "FTHG", "FTAG"]].copy()
    home.columns = ["date", "team", "goals_for", "goals_against"]
    home["orig_index"] = matches.index

    away = matches[["Date", "AwayTeam", "FTAG", "FTHG"]].copy()
    away.columns = ["date", "team", "goals_for", "goals_against"]
    away["orig_index"] = matches.index

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "date", "orig_index"])


def add_form_features(matches: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    log = _team_match_log(matches)
    grouped = log.groupby("team")
    log["form_goals_for"] = grouped["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    log["form_goals_against"] = grouped["goals_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    log["side"] = np.where(log.index < len(matches), "home", "away")
    home_form = log[log["side"] == "home"].set_index("orig_index")[["form_goals_for", "form_goals_against"]]
    away_form = log[log["side"] == "away"].set_index("orig_index")[["form_goals_for", "form_goals_against"]]
    home_form.columns = ["home_form_goals_for", "home_form_goals_against"]
    away_form.columns = ["away_form_goals_for", "away_form_goals_against"]

    return matches.join(home_form).join(away_form)


def add_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["pair_key"] = [tuple(sorted(pair)) for pair in zip(matches["HomeTeam"], matches["AwayTeam"])]

    team_a = matches["pair_key"].str[0]
    is_team_a_home = matches["HomeTeam"] == team_a
    matches["_goal_diff_team_a"] = matches["FTHG"] - matches["FTAG"]
    matches.loc[~is_team_a_home, "_goal_diff_team_a"] = -matches.loc[~is_team_a_home, "_goal_diff_team_a"]

    grouped = matches.groupby("pair_key")["_goal_diff_team_a"]
    mean_diff_team_a = grouped.transform(lambda s: s.shift(1).expanding().mean())
    h2h_count = grouped.transform(lambda s: s.shift(1).expanding().count())

    sign = pd.Series(1, index=matches.index)
    sign[~is_team_a_home] = -1
    raw_diff = (mean_diff_team_a * sign).fillna(0.0)
    h2h_count = h2h_count.fillna(0)

    matches["h2h_diff"] = raw_diff * h2h_count / (h2h_count + H2H_SHRINKAGE_K)
    matches["h2h_count"] = h2h_count
    return matches.drop(columns=["pair_key", "_goal_diff_team_a"])


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_with_elo.csv", low_memory=False)
    matches["Date"] = pd.to_datetime(matches["Date"], format="mixed", dayfirst=True)
    matches = matches.sort_values("Date").reset_index(drop=True)

    merged = add_form_features(matches)
    merged = add_h2h_features(merged)

    for col in ["home_form_goals_for", "away_form_goals_for"]:
        print(f"{col}: {1 - merged[col].isna().mean():.1%} coverage")
    print(f"Matches with at least one prior h2h meeting: {(merged['h2h_count'] > 0).mean():.1%}")

    out_path = DATA_DIR / "matches_with_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
