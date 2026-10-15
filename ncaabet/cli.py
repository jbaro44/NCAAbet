"""ncaabet command line: ratings, predictions, slates, edges, and backtests."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import date, timedelta

import pandas as pd

from . import data
from .backtest import accuracy_report, ats_report, calibration
from .lines import TeamResolver, fetch_odds_api, find_edges, read_lines_csv
from .model import Model

HISTORY_SEASONS = 5


def _seasons(current: int) -> range:
    return range(current - HISTORY_SEASONS, current + 1)


def _model(current: int, refresh: bool = False) -> tuple[Model, pd.DataFrame]:
    """Walk-forward model through all history, cached on disk until the data changes."""
    if refresh:
        data.load_schedule(current, refresh=True)
        data.load_team_box(current, refresh=True)
    games = data.load_games(_seasons(current))
    key = (len(games), str(games["date"].max()), current)
    cache = data.data_dir() / "model.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved["key"] == key:
            return saved["model"], saved["preds"]

    def progress(season, i, n):
        if i == n or i % 25 == 0:
            print(f"\r  rating season {season}: day {i}/{n}", end="", file=sys.stderr, flush=True)

    print("Building ratings from game history...", file=sys.stderr)
    model = Model()
    preds = model.run(games, progress)
    print(file=sys.stderr)
    if model.season != current:
        model.start_season(current)  # preseason: carry last year's ratings forward
    with open(cache, "wb") as f:
        pickle.dump({"key": key, "model": model, "preds": preds}, f)
    return model, preds


def _resolver(current: int) -> TeamResolver:
    return TeamResolver(data.team_directory(_seasons(current)))


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def cmd_update(args, current):
    for season in _seasons(current):
        data.load_schedule(season, refresh=True)
        data.load_team_box(season, refresh=True)
    _model(current)
    print("Data and ratings are up to date.")


def cmd_ratings(args, current):
    model, _ = _model(current)
    names = _resolver(current).display
    table = model.ratings.table(names)
    recent = data.load_games([current - 1, current])
    active = set(recent["home_id"]) | set(recent["away_id"])
    table = table[table["team_id"].isin(active)].reset_index(drop=True)
    table.index += 1
    label = "" if (recent["season"] == current).any() else "preseason "
    print(f"{current} {label}ratings (points per 100 possessions, adjusted for opponent and venue)\n")
    print(table.drop(columns="team_id").head(args.top).round(1).to_string())


def cmd_predict(args, current):
    model, _ = _model(current)
    res = _resolver(current)
    h, a = res.resolve(args.home), res.resolve(args.away)
    p = model.predict(h, a, args.neutral)
    hn, an = res.display[h], res.display[a]
    site = "neutral site" if args.neutral else f"at {hn}"
    fav, line = (hn, p.spread) if p.margin >= 0 else (an, -p.spread)
    print(f"{an} vs {hn} ({site})")
    print(f"  Projected score: {hn} {p.home_pts:.1f}, {an} {p.away_pts:.1f}")
    print(f"  Model line:      {fav} {line:+.1f}   Total {p.total:.1f}")
    print(f"  Win probability: {hn} {_fmt_pct(p.home_win_prob)}, {an} {_fmt_pct(1 - p.home_win_prob)}")
    print(f"  (efficiency says {p.eff_margin:+.1f}, Elo says {p.elo_margin:+.1f} for {hn})")


def cmd_slate(args, current):
    model, _ = _model(current)
    start = pd.Timestamp(args.date) if args.date else None
    games = data.load_upcoming(current)
    if start is None:
        start = games["date"].min() if not games.empty else pd.Timestamp(date.today())
    games = games[(games["date"] >= start) & (games["date"] < start + timedelta(days=args.days))]
    if games.empty:
        print("No scheduled games in that window.")
        return
    rows = []
    for g in games.itertuples(index=False):
        p = model.predict(g.home_id, g.away_id, g.neutral)
        fav, line = (g.home_name, p.spread) if p.margin >= 0 else (g.away_name, -p.spread)
        rows.append({"date": g.date.date(), "matchup": f"{g.away_name} {'vs' if g.neutral else '@'} {g.home_name}",
                     "favorite": fav, "line": round(line, 1), "total": round(p.total, 1),
                     "fav_win%": round(100 * max(p.home_win_prob, 1 - p.home_win_prob), 1)})
    df = pd.DataFrame(rows)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {len(df)} games to {args.csv}")
    print(df.to_string(index=False))


def cmd_edges(args, current):
    model, _ = _model(current, refresh=args.refresh)
    if args.lines:
        lines = read_lines_csv(args.lines)
    else:
        key = os.environ.get("ODDS_API_KEY")
        if not key:
            sys.exit("Pass --lines FILE.csv, or set ODDS_API_KEY to pull live odds from the-odds-api.com")
        lines = fetch_odds_api(key, args.book)
    upcoming = data.load_upcoming(current)
    neutral = {(h, a): n for h, a, n in zip(upcoming["home_id"], upcoming["away_id"], upcoming["neutral"])}
    edges = find_edges(model, lines, _resolver(current), neutral, args.kelly, args.trust)
    if edges.empty:
        print("No lines to evaluate.")
        return
    edges = edges[(edges["edge_pts"] >= args.min_edge) & (edges["ev"] > 0)]
    if edges.empty:
        print(f"No bets with at least {args.min_edge} points of edge. Passing is a position too.")
        return
    show = edges[["date", "away", "home", "market", "pick", "odds", "model_spread", "model_total",
                  "edge_pts", "prob", "ev", "kelly"]].copy()
    show["prob"] = show["prob"].map(_fmt_pct)
    show["ev"] = show["ev"].map(lambda x: f"{100 * x:+.1f}%")
    show["odds"] = show["odds"].astype(int)
    show["stake"] = (edges["kelly"] * args.bankroll).round(2)
    show = show.drop(columns="kelly").round(1)
    print(show.to_string(index=False))
    print(f"\nProbabilities blend {args.trust:g} model / {1 - args.trust:g} market. Stakes are {args.kelly:g}x Kelly"
          f" on a {args.bankroll:g} bankroll. Model spreads are from the home side.")


def cmd_backtest(args, current):
    _, preds = _model(current)
    if args.seasons:
        preds = preds[preds["season"].isin(args.seasons)]
    else:
        preds = preds[preds["season"] > preds["season"].min()]  # first season is burn-in
    print("Walk-forward accuracy (every prediction made before the game, using only prior results)\n")
    rep = accuracy_report(preds)
    rep["winner_pct"] = rep["winner_pct"].map(_fmt_pct)
    print(rep.round(3).to_string())
    print("\nCalibration: favorite's predicted vs actual win rate\n")
    print(calibration(preds).round(3).to_string())
    if args.lines:
        print("\nAgainst the spread, by minimum model edge (points)\n")
        print(ats_report(preds, args.lines, _resolver(current)).round(3).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ncaabet", description="NCAA basketball spread & total model")
    parser.add_argument("--season", dest="current", type=int, default=None,
                        help="season to work in, named for the year it ends (default: today's)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("update", help="download the latest results and rebuild ratings")

    p = sub.add_parser("ratings", help="team power ratings")
    p.add_argument("--top", type=int, default=25)

    p = sub.add_parser("predict", help="predict one game")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--neutral", action="store_true")

    p = sub.add_parser("slate", help="predict scheduled games")
    p.add_argument("--date", help="first day (YYYY-MM-DD); default: next day with games")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--csv", help="also write the slate to this CSV")

    p = sub.add_parser("edges", help="compare the model to sportsbook lines")
    p.add_argument("--lines", help="CSV of lines (see README); default: live odds via ODDS_API_KEY")
    p.add_argument("--book", help="bookmaker key for live odds (e.g. draftkings); default: consensus")
    p.add_argument("--min-edge", type=float, default=2.0, help="minimum points of disagreement")
    p.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    p.add_argument("--trust", type=float, default=0.3,
                   help="weight on the model vs the market line, 0-1 (default 0.3)")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--refresh", action="store_true", help="pull today's results first")

    p = sub.add_parser("backtest", help="walk-forward accuracy, and ATS results given historical lines")
    p.add_argument("--seasons", type=int, nargs="+", help="limit to these seasons")
    p.add_argument("--lines", help="CSV of historical lines with a date column")

    args = parser.parse_args(argv)
    current = args.current or data.season_for_date(date.today())
    commands = {"update": cmd_update, "ratings": cmd_ratings, "predict": cmd_predict,
                "slate": cmd_slate, "edges": cmd_edges, "backtest": cmd_backtest}
    commands[args.command](args, current)


if __name__ == "__main__":
    main()
