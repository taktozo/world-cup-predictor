"""Fetch real upcoming EPL/La Liga fixtures from football-data.org.

Free tier, 10 calls/minute, requires a registered API token (passed in,
not stored here -- the deployed app reads it from st.secrets).
"""

import pandas as pd
import requests

API_BASE = "https://api.football-data.org/v4"
COMPETITIONS = {"PL": "EPL", "PD": "La Liga"}

# football-data.org shortName -> our football-data.co.uk naming convention.
TEAM_ALIASES = {
    "Brighton Hove": "Brighton",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Nottingham": "Nott'm Forest",
    "Coventry City": "Coventry",
    "Alavés": "Alaves",
    "Athletic": "Ath Bilbao",
    "Atleti": "Ath Madrid",
    "Barça": "Barcelona",
    "Deportivo": "La Coruna",
    "Espanyol": "Espanol",
    "Málaga": "Malaga",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Sevilla FC": "Sevilla",
}


def _fetch_competition(api_token: str, code: str) -> list[dict]:
    resp = requests.get(
        f"{API_BASE}/competitions/{code}/matches",
        params={"status": "SCHEDULED"},
        headers={"X-Auth-Token": api_token},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["matches"]


def get_upcoming_fixtures(api_token: str) -> pd.DataFrame:
    """Returns columns: date, league, home_team, away_team (aliased to our naming)."""
    rows = []
    for code, league_name in COMPETITIONS.items():
        for m in _fetch_competition(api_token, code):
            home = m["homeTeam"]["shortName"]
            away = m["awayTeam"]["shortName"]
            rows.append(
                {
                    "date": pd.Timestamp(m["utcDate"]).tz_localize(None),
                    "league": league_name,
                    "home_team": TEAM_ALIASES.get(home, home),
                    "away_team": TEAM_ALIASES.get(away, away),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)
