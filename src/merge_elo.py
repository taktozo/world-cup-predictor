"""Attach each team's pre-match Elo rating to every row in results.csv.

For every match, looks up each team's most recent Elo rating as of
(on or before) the match date. Elo ratings are stored "long" (one row
per team per rating change), so this uses an as-of merge rather than
a plain join.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# eloratings.csv team names that differ from results.csv, mapped elo -> results.
TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "Democratic Republic of Congo": "DR Congo",
    "Ireland": "Republic of Ireland",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "East Timor": "Timor-Leste",
    "US Virgin Islands": "United States Virgin Islands",
    "Macao": "Macau",
}


def load_results(path: Path = DATA_DIR / "results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def load_elo(path: Path = DATA_DIR / "eloratings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    # eloratings.net uses non-breaking spaces between words in team names.
    df["team"] = df["team"].str.replace("\xa0", " ", regex=False)
    df["team"] = df["team"].replace(TEAM_ALIASES)
    return df


def _attach_elo(matches: pd.DataFrame, elo: pd.DataFrame, team_col: str, out_col: str) -> pd.DataFrame:
    side = matches[["date", team_col]].reset_index().rename(columns={team_col: "team"})
    side = side.sort_values("date")
    elo_sorted = elo.sort_values("date")

    merged = pd.merge_asof(
        side,
        elo_sorted[["date", "team", "rating"]],
        on="date",
        by="team",
        direction="backward",
    )
    merged = merged.set_index("index").sort_index()
    return matches.assign(**{out_col: merged["rating"]})


def add_elo_ratings(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    matches = _attach_elo(matches, elo, "home_team", "home_elo")
    matches = _attach_elo(matches, elo, "away_team", "away_elo")
    return matches


def main() -> None:
    matches = load_results()
    elo = load_elo()
    merged = add_elo_ratings(matches, elo)

    for col, team_col in [("home_elo", "home_team"), ("away_elo", "away_team")]:
        missing = merged[col].isna()
        coverage = 1 - missing.mean()
        print(f"{col}: {coverage:.1%} coverage ({missing.sum()} matches missing a rating)")
        if missing.any():
            unmatched_teams = sorted(merged.loc[missing, team_col].unique())
            print(f"  sample unmatched teams: {unmatched_teams[:15]}")

    out_path = DATA_DIR / "results_with_elo.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
