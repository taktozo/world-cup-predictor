"""Add rolling recent-form features to each match.

For every match, each team gets its trailing-N-match average goals
scored and goals conceded (across both home and away matches), as of
strictly before the current match -- a lightweight stand-in for
per-team attack/defense strength and current form, computed the same
way for both sides so it's comparable.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FORM_WINDOW = 10  # trailing matches to average over


def _team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["date", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "goals_for", "goals_against"]
    home["orig_index"] = matches.index

    away = matches[["date", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "goals_for", "goals_against"]
    away["orig_index"] = matches.index

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "date", "orig_index"])


def add_form_features(matches: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    log = _team_match_log(matches)

    grouped = log.groupby("team")
    # shift(1) excludes the match itself -- form is only ever known from prior matches
    log["form_goals_for"] = grouped["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    log["form_goals_against"] = grouped["goals_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    # split by whether this log row came from the home or away half of the match --
    # concat (before sorting) put all home rows at index [0, len(matches)) and away rows after
    log["side"] = np.where(log.index < len(matches), "home", "away")
    home_form = log[log["side"] == "home"].set_index("orig_index")[["form_goals_for", "form_goals_against"]]
    away_form = log[log["side"] == "away"].set_index("orig_index")[["form_goals_for", "form_goals_against"]]
    home_form.columns = ["home_form_goals_for", "home_form_goals_against"]
    away_form.columns = ["away_form_goals_for", "away_form_goals_against"]

    return matches.join(home_form).join(away_form)


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "results_with_elo.csv", parse_dates=["date"])
    merged = add_form_features(matches)

    for col in ["home_form_goals_for", "away_form_goals_for"]:
        coverage = 1 - merged[col].isna().mean()
        print(f"{col}: {coverage:.1%} coverage")

    out_path = DATA_DIR / "results_with_form.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
