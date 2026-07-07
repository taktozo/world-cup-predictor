"""Add squad total market value as a feature, computed from Transfermarkt
player valuations rather than the much heavier (349MB) game-lineups file.

player_valuations.csv's own `player_club_domestic_competition_id` column
turned out to be unreliable (rows tagged GB1/ES1 included Bayern Munich,
Galatasaray, etc.) -- instead we filter by `current_club_id` against the
verified club list from clubs.csv's own domestic_competition_id field, and
map club_id -> football-data.co.uk short names by hand (anchored on the
numeric id, not the free-text name, to avoid encoding/transliteration
mismatches).

Squad value at any date = sum of each player's most recent valuation as of
that date, for all players whose latest known club is this one -- tracked
with a single chronological sweep over valuation events rather than a
per-match lookup, since that's O(events + dates) instead of O(events x dates).
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"

# club_id (from Transfermarkt clubs.csv, filtered to domestic_competition_id
# in GB1/ES1) -> football-data.co.uk's short team name for that same club.
CLUB_ID_TO_FD_NAME = {
    11: "Arsenal", 989: "Bournemouth", 405: "Aston Villa", 621: "Ath Bilbao",
    1148: "Brentford", 1237: "Brighton", 1132: "Burnley", 1244: "Leganes",
    603: "Cardiff", 631: "Chelsea", 331: "Osasuna", 13: "Ath Madrid",
    873: "Crystal Palace", 2687: "Cadiz", 993: "Cordoba", 1108: "Alaves",
    897: "La Coruna", 1531: "Elche", 29: "Everton", 931: "Fulham",
    131: "Barcelona", 3709: "Getafe", 12321: "Girona", 16795: "Granada",
    1110: "Huddersfield", 3008: "Hull", 677: "Ipswich", 399: "Leeds",
    1003: "Leicester", 3368: "Levante", 31: "Liverpool", 1031: "Luton",
    281: "Man City", 985: "Man United", 641: "Middlesbrough", 1084: "Malaga",
    762: "Newcastle", 1123: "Norwich", 703: "Nott'm Forest", 1039: "QPR",
    367: "Vallecano", 1032: "Reading", 150: "Betis", 940: "Celta",
    237: "Mallorca", 418: "Real Madrid", 2497: "Oviedo", 681: "Sociedad",
    366: "Valladolid", 142: "Zaragoza", 714: "Espanol", 1533: "Eibar",
    5358: "Huesca", 368: "Sevilla", 350: "Sheffield United", 180: "Southampton",
    2448: "Sp Gijon", 512: "Stoke", 289: "Sunderland", 2288: "Swansea",
    148: "Tottenham", 3302: "Almeria", 472: "Las Palmas", 1049: "Valencia",
    1050: "Villarreal", 1010: "Watford", 984: "West Brom", 379: "West Ham",
    1071: "Wigan", 543: "Wolves",
}


def compute_squad_values_at_dates(valuations: pd.DataFrame, match_dates: pd.Series) -> pd.DataFrame:
    """One chronological sweep: at each unique match date, snapshot each
    club's total squad value (sum of latest known valuation per player)."""
    valuations = valuations.sort_values("date")
    unique_dates = sorted(match_dates.unique())

    player_last = {}  # player_id -> (club_id, value)
    club_total = defaultdict(float)
    snapshots = []  # (date, club_id, total_value)

    events = valuations[["date", "player_id", "current_club_id", "market_value_in_eur"]].itertuples(index=False)
    event = next(events, None)

    for match_date in unique_dates:
        while event is not None and event.date <= match_date:
            pid = event.player_id
            if pid in player_last:
                old_club, old_value = player_last[pid]
                club_total[old_club] -= old_value
            club_total[event.current_club_id] += event.market_value_in_eur
            player_last[pid] = (event.current_club_id, event.market_value_in_eur)
            event = next(events, None)

        for club_id in CLUB_ID_TO_FD_NAME:
            value = club_total[club_id] if club_id in club_total else float("nan")
            snapshots.append((match_date, club_id, value))

    return pd.DataFrame(snapshots, columns=["date", "club_id", "squad_value"])


def add_squad_value_features(matches: pd.DataFrame) -> pd.DataFrame:
    valuations = pd.read_csv(DATA_DIR / "_player_valuations_raw.csv")
    valuations = valuations[valuations["current_club_id"].isin(CLUB_ID_TO_FD_NAME.keys())].copy()
    valuations["date"] = pd.to_datetime(valuations["date"])

    snapshots = compute_squad_values_at_dates(valuations, matches["Date"])
    snapshots["team"] = snapshots["club_id"].map(CLUB_ID_TO_FD_NAME)
    snapshots = snapshots.drop(columns="club_id").set_index(["date", "team"])["squad_value"]

    matches = matches.copy()
    matches["home_squad_value"] = matches.set_index(["Date", "HomeTeam"]).index.map(snapshots)
    matches["away_squad_value"] = matches.set_index(["Date", "AwayTeam"]).index.map(snapshots)
    return matches


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches_with_odds.csv", low_memory=False, parse_dates=["Date"])
    merged = add_squad_value_features(matches)

    coverage = 1 - merged["home_squad_value"].isna().mean()
    print(f"home_squad_value: {coverage:.1%} coverage")
    print(f"Earliest match with squad value: {merged.loc[merged['home_squad_value'].notna(), 'Date'].min().date()}")

    out_path = DATA_DIR / "matches_with_squad_value.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
