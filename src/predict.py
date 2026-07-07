"""Try the baseline model on an actual matchup.

Usage:
    python predict.py "Argentina" "France" --neutral
    python predict.py "Brazil" "Germany"
"""

import argparse

from inference import fit_model, h2h_diff_for_pair, load_latest_elo, load_latest_form, load_latest_h2h, predict_match


def predict(home_team: str, away_team: str, neutral: bool) -> None:
    latest_elo = load_latest_elo()
    latest_form = load_latest_form()
    latest_h2h = load_latest_h2h()
    for team in (home_team, away_team):
        if team not in latest_elo.index:
            raise SystemExit(f"No Elo rating found for {team!r}. Check spelling against data/latest_elo.csv.")
        if team not in latest_form.index:
            raise SystemExit(f"No form data found for {team!r}. Check spelling against data/latest_form.csv.")

    model = fit_model()
    result = predict_match(home_team, away_team, neutral, latest_elo, latest_form, latest_h2h, model)

    print(f"{home_team} (Elo {result['home_elo']:.0f}) vs {away_team} (Elo {result['away_elo']:.0f})")
    print(f"Neutral venue: {neutral}")
    h2h = h2h_diff_for_pair(home_team, away_team, latest_h2h)
    print(f"Head-to-head (shrunk goal diff, {home_team}'s favor): {h2h:+.2f}")
    print(f"\nExpected score: {home_team} {result['home_goals']:.2f} - {result['away_goals']:.2f} {away_team}")
    print(f"\nP({home_team} win) = {result['p_home_win']:.1%}")
    print(f"P(draw)            = {result['p_draw']:.1%}")
    print(f"P({away_team} win) = {result['p_away_win']:.1%}")
    print(f"\nBoth teams to score: Yes {result['p_btts_yes']:.1%} / No {result['p_btts_no']:.1%}")
    print("\nOver/Under total goals:")
    for line, p_over in result["over_under"].items():
        print(f"  Over {line} = {p_over:.1%}  |  Under {line} = {1 - p_over:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--neutral", action="store_true", help="Set if played at a neutral venue")
    args = parser.parse_args()
    predict(args.home_team, args.away_team, args.neutral)


if __name__ == "__main__":
    main()
