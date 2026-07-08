"""Fetch real upcoming World Cup fixtures from football-data.org.

Free tier, 10 calls/minute, requires a registered API token (passed in,
not stored here -- the deployed app reads it from st.secrets). Mirrors
src/club/fixtures.py's approach for the club leagues page.
"""

import pandas as pd
import requests

API_BASE = "https://api.football-data.org/v4"
COMPETITION_CODE = "WC"

# football-data.org name -> our latest_elo.csv naming convention (which
# follows eloratings.net, not results.csv -- see merge_elo.py's own alias
# table for why those two differ from each other).
TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "Democratic Republic of Congo",
}


def get_upcoming_fixtures(api_token: str) -> pd.DataFrame:
    """Returns columns: date, stage, home_team, away_team (aliased), for
    not-yet-played World Cup matches."""
    resp = requests.get(
        f"{API_BASE}/competitions/{COMPETITION_CODE}/matches",
        params={"status": "SCHEDULED"},
        headers={"X-Auth-Token": api_token},
        timeout=10,
    )
    resp.raise_for_status()

    rows = []
    for m in resp.json()["matches"]:
        if m["homeTeam"]["name"] is None or m["awayTeam"]["name"] is None:
            continue  # bracket slot not yet determined (e.g. semifinal before quarterfinals finish)
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        rows.append(
            {
                "date": pd.Timestamp(m["utcDate"]).tz_localize(None),
                "stage": m["stage"].replace("_", " ").title(),
                "home_team": TEAM_ALIASES.get(home, home),
                "away_team": TEAM_ALIASES.get(away, away),
            }
        )

    df = pd.DataFrame(rows, columns=["date", "stage", "home_team", "away_team"])
    return df.sort_values("date").reset_index(drop=True)
