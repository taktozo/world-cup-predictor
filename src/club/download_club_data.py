"""Download EPL + La Liga historical match data from football-data.co.uk.

Free, no auth, one CSV per season per league. Concatenated into a single
raw file; older seasons have fewer columns (no odds/stats), which is fine --
downstream feature scripts handle missing columns per-era.
"""

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "club"
LEAGUES = {"E0": "EPL", "SP1": "La Liga"}
FIRST_SEASON_START_YEAR = 1993
LAST_SEASON_START_YEAR = 2025  # 2025/26 season


def season_codes():
    for year in range(FIRST_SEASON_START_YEAR, LAST_SEASON_START_YEAR + 1):
        yield f"{year % 100:02d}{(year + 1) % 100:02d}"


def download_all() -> pd.DataFrame:
    frames = []
    for season in season_codes():
        for div, league_name in LEAGUES.items():
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200 or not resp.content:
                continue
            try:
                df = pd.read_csv(pd.io.common.BytesIO(resp.content))
            except (UnicodeDecodeError, pd.errors.ParserError):
                # A few older-season files have stray bad bytes/extra fields
                try:
                    df = pd.read_csv(
                        pd.io.common.BytesIO(resp.content), encoding="latin1", on_bad_lines="skip"
                    )
                except Exception as e:
                    print(f"  skipping {league_name} {season}: {e}")
                    continue
            if "HomeTeam" not in df.columns:  # empty/malformed file
                continue
            df["League"] = league_name
            df["Season"] = season
            frames.append(df)
            print(f"  {league_name} {season}: {len(df)} matches")

    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined = download_all()
    out_path = DATA_DIR / "matches_raw.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined):,} matches to {out_path}")


if __name__ == "__main__":
    main()
