"""Try the baseline model on an actual matchup.

Usage:
    python predict.py "Argentina" "France" --neutral
    python predict.py "Brazil" "Germany"
"""

import argparse

from inference import fit_model, load_latest_elo, predict_match


def predict(home_team: str, away_team: str, neutral: bool) -> None:
    latest_elo = load_latest_elo()
    for team in (home_team, away_team):
        if team not in latest_elo.index:
            raise SystemExit(f"No Elo rating found for {team!r}. Check spelling against data/latest_elo.csv.")

    model = fit_model()
    result = predict_match(home_team, away_team, neutral, latest_elo, model)

    print(f"{home_team} (Elo {result['home_elo']:.0f}) vs {away_team} (Elo {result['away_elo']:.0f})")
    print(f"Neutral venue: {neutral}")
    print(f"\nExpected score: {home_team} {result['home_goals']:.2f} - {result['away_goals']:.2f} {away_team}")
    print(f"\nP({home_team} win) = {result['p_home_win']:.1%}")
    print(f"P(draw)            = {result['p_draw']:.1%}")
    print(f"P({away_team} win) = {result['p_away_win']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--neutral", action="store_true", help="Set if played at a neutral venue")
    args = parser.parse_args()
    predict(args.home_team, args.away_team, args.neutral)


if __name__ == "__main__":
    main()
