"""Add a head-to-head feature: this specific pair's history against each other.

Two teams often have only met a handful of times in 150 years of data, and
old meetings say little given squad/manager turnover -- so the raw average
goal difference in their past meetings is shrunk toward 0 based on how many
times they've actually played, via count / (count + H2H_SHRINKAGE_K). With
zero prior meetings the feature is exactly 0 (no info, no bias); with many
meetings it approaches the raw historical average.
"""

from pathlib import Path

import pandas as pd

from inference import H2H_SHRINKAGE_K

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _pair_key(home_team: pd.Series, away_team: pd.Series) -> pd.Series:
    return [tuple(sorted(pair)) for pair in zip(home_team, away_team)]


def add_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["pair_key"] = _pair_key(matches["home_team"], matches["away_team"])

    # goal_diff from the perspective of "team_a", the alphabetically-first team in the pair,
    # so the same historical games read identically regardless of who's listed as home/away below
    team_a = matches["pair_key"].str[0]
    is_team_a_home = matches["home_team"] == team_a
    matches["_goal_diff_team_a"] = matches["home_score"] - matches["away_score"]
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
    matches = pd.read_csv(DATA_DIR / "results_with_form.csv", parse_dates=["date"])
    merged = add_h2h_features(matches)

    have_history = merged["h2h_count"] > 0
    print(f"Matches with at least one prior meeting: {have_history.mean():.1%}")
    print(merged["h2h_count"].describe())

    out_path = DATA_DIR / "results_with_h2h.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
