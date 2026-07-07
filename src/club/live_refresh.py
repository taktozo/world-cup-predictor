"""Keep the club model's ratings current between manual data refreshes.

Fetches whichever season's file is "current" right now directly from
football-data.co.uk (same free, no-auth source as the historical download),
filters to matches after the committed snapshot's cutoff date, and applies
the same incremental update rules used to build the snapshots in the first
place -- so predictions reflect real results without needing a manual
re-run of the full pipeline + git push each time.

Squad value is NOT refreshed live (would need the much heavier Transfermarkt
pipeline); it stays at its last committed snapshot, which is an acceptable
staleness since squad value moves slowly outside transfer windows.
"""

from datetime import date

import pandas as pd
import requests

from compute_elo import HOME_ADVANTAGE, K, STARTING_RATING, _expected_result, _goal_margin_multiplier

LEAGUES = {"E0": "EPL", "SP1": "La Liga"}
FORM_WINDOW = 10


def current_season_code(today: date | None = None) -> str:
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_current_season_matches() -> pd.DataFrame:
    season = current_season_code()
    frames = []
    for div, league_name in LEAGUES.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
        try:
            resp = requests.get(url, timeout=5)
        except requests.RequestException:
            continue
        if resp.status_code != 200 or not resp.content:
            continue
        try:
            df = pd.read_csv(pd.io.common.BytesIO(resp.content))
        except Exception:
            continue
        if "HomeTeam" not in df.columns:
            continue
        df["League"] = league_name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "League"])

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["Date"] = pd.to_datetime(combined["Date"], format="mixed", dayfirst=True)
    return combined.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).sort_values("Date")


def refresh_elo(latest_elo: pd.Series, new_matches: pd.DataFrame) -> pd.Series:
    ratings = latest_elo.to_dict()
    for row in new_matches.itertuples():
        r_home = ratings.setdefault(row.HomeTeam, STARTING_RATING)
        r_away = ratings.setdefault(row.AwayTeam, STARTING_RATING)

        goal_diff = row.FTHG - row.FTAG
        w = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)
        we = _expected_result(r_home - r_away + HOME_ADVANTAGE)
        g = _goal_margin_multiplier(abs(goal_diff))
        delta = K * g * (w - we)

        ratings[row.HomeTeam] = r_home + delta
        ratings[row.AwayTeam] = r_away - delta

    return pd.Series(ratings, name="rating")


def refresh_form(latest_form: pd.DataFrame, new_matches: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    # Seed each team's window with its committed mean repeated `window` times --
    # exact once `window` real matches have accumulated on top, a reasonable
    # approximation before that since we don't store raw per-match history here.
    histories = {
        team: {
            "for": [row["form_goals_for"]] * window,
            "against": [row["form_goals_against"]] * window,
        }
        for team, row in latest_form.iterrows()
    }

    def ensure(team):
        if team not in histories:
            histories[team] = {"for": [0.0] * window, "against": [0.0] * window}

    for row in new_matches.itertuples():
        ensure(row.HomeTeam)
        ensure(row.AwayTeam)
        histories[row.HomeTeam]["for"].append(row.FTHG)
        histories[row.HomeTeam]["against"].append(row.FTAG)
        histories[row.AwayTeam]["for"].append(row.FTAG)
        histories[row.AwayTeam]["against"].append(row.FTHG)

    records = []
    for team, hist in histories.items():
        records.append(
            {
                "team": team,
                "form_goals_for": sum(hist["for"][-window:]) / window,
                "form_goals_against": sum(hist["against"][-window:]) / window,
            }
        )
    return pd.DataFrame(records).set_index("team")


def refresh_h2h(latest_h2h: pd.DataFrame, new_matches: pd.DataFrame) -> pd.DataFrame:
    state = {idx: (row["mean_diff_team_a"], row["count"]) for idx, row in latest_h2h.iterrows()}

    for row in new_matches.itertuples():
        pair_key = tuple(sorted([row.HomeTeam, row.AwayTeam]))
        team_a = pair_key[0]
        goal_diff = row.FTHG - row.FTAG if row.HomeTeam == team_a else row.FTAG - row.FTHG

        old_mean, old_count = state.get(pair_key, (0.0, 0))
        new_count = old_count + 1
        new_mean = (old_mean * old_count + goal_diff) / new_count
        state[pair_key] = (new_mean, new_count)

    records = [{"team_a": k[0], "team_b": k[1], "mean_diff_team_a": v[0], "count": v[1]} for k, v in state.items()]
    result = pd.DataFrame(records)
    result.index = pd.MultiIndex.from_arrays([result["team_a"], result["team_b"]])
    return result


def refresh_last_match_date(latest_match_date: pd.Series, new_matches: pd.DataFrame) -> pd.Series:
    dates = latest_match_date.to_dict()
    for row in new_matches.itertuples():
        dates[row.HomeTeam] = row.Date
        dates[row.AwayTeam] = row.Date
    return pd.Series(dates, name="last_match_date")


def get_refreshed_snapshots(latest_elo, latest_form, latest_h2h, latest_match_date, committed_max_date: pd.Timestamp):
    """Returns (elo, form, h2h, last_match_date, n_new_matches, current_season_matches).
    Falls back to the committed snapshots unchanged if the live fetch fails or
    has nothing new. current_season_matches is returned even when empty (e.g.
    the new season hasn't started) so callers can build a live, zero-filled
    standings table rather than just skipping the update."""
    empty_matches = pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "League"])
    try:
        fetched = fetch_current_season_matches()
    except Exception:
        return latest_elo, latest_form, latest_h2h, latest_match_date, 0, empty_matches

    new_matches = fetched[fetched["Date"] > committed_max_date]
    if new_matches.empty:
        return latest_elo, latest_form, latest_h2h, latest_match_date, 0, empty_matches

    updated_elo = refresh_elo(latest_elo, new_matches)
    updated_form = refresh_form(latest_form, new_matches)
    updated_h2h = refresh_h2h(latest_h2h, new_matches)
    updated_match_date = refresh_last_match_date(latest_match_date, new_matches)
    return updated_elo, updated_form, updated_h2h, updated_match_date, len(new_matches), new_matches
