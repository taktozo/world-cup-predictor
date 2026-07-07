"""Export a pairwise head-to-head snapshot: one row per pair of teams that
have ever met, with the all-time average goal difference (from the
alphabetically-first team's perspective) and how many times they've played.

Small committed lookup, same role as latest_elo.csv/latest_form.csv: lets the
deployed app compute a live h2h_diff feature for any two teams without the
full match history.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "results_with_elo.csv")
    matches = matches.dropna(subset=["home_score", "away_score"])  # only completed matches

    pair_key = [tuple(sorted(pair)) for pair in zip(matches["home_team"], matches["away_team"])]
    team_a = pd.Series([p[0] for p in pair_key], index=matches.index)
    team_b = pd.Series([p[1] for p in pair_key], index=matches.index)

    is_team_a_home = matches["home_team"] == team_a
    goal_diff_team_a = matches["home_score"] - matches["away_score"]
    goal_diff_team_a = goal_diff_team_a.where(is_team_a_home, -goal_diff_team_a)

    summary = pd.DataFrame({"team_a": team_a, "team_b": team_b, "goal_diff_team_a": goal_diff_team_a})
    latest = summary.groupby(["team_a", "team_b"]).agg(
        mean_diff_team_a=("goal_diff_team_a", "mean"), count=("goal_diff_team_a", "size")
    )
    latest = latest.reset_index().sort_values(["team_a", "team_b"]).reset_index(drop=True)

    out_path = DATA_DIR / "latest_h2h.csv"
    latest.to_csv(out_path, index=False)
    print(f"Saved {len(latest):,} team pairs to {out_path}")


if __name__ == "__main__":
    main()
