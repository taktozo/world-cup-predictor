"""Add a referee tendency feature: this referee's rolling average cards
shown per game (yellow + red), as-of each match (excluding the match
itself). Referee-only known for ~39% of matches (from 2000 onward) and
cards recorded for ~70% -- missing values default to the overall average
(a neutral assumption for unknown/rare referees) rather than 0.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
REFEREE_FORM_WINDOW = 20


def add_referee_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["total_cards"] = matches[["HY", "AY", "HR", "AR"]].sum(axis=1, min_count=1)

    has_ref = matches["Referee"].notna()
    overall_avg_cards = matches.loc[has_ref, "total_cards"].mean()

    grouped = matches.loc[has_ref].groupby("Referee")["total_cards"]
    rolling_avg = grouped.transform(lambda s: s.shift(1).rolling(REFEREE_FORM_WINDOW, min_periods=1).mean())

    matches["referee_avg_cards"] = overall_avg_cards
    matches.loc[has_ref, "referee_avg_cards"] = rolling_avg.fillna(overall_avg_cards)
    return matches


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_with_squad_value.csv", low_memory=False, parse_dates=["Date"])
    matches = matches.sort_values("Date").reset_index(drop=True)
    merged = add_referee_features(matches)

    print(f"referee_avg_cards: {1 - merged['referee_avg_cards'].isna().mean():.1%} coverage (should be 100%)")
    print(merged[["Referee", "referee_avg_cards"]].dropna().drop_duplicates("Referee").sort_values(
        "referee_avg_cards"
    ).iloc[[0, -1]])

    out_path = DATA_DIR / "matches_with_referee.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
