"""Sportsbook lines: load from CSV or The Odds API, match team names, and find edges.

CSV format (header required; only home/away are mandatory):

    date,home,away,neutral,spread,total,home_odds,away_odds,over_odds,under_odds
    2026-11-03,Duke,Kansas,1,-2.5,151.5,-110,-110,-110,-110

`spread` is the home team's line (-2.5 = home favored by 2.5). Missing odds default to -110.
"""

from __future__ import annotations

import difflib
import re
from statistics import median

import pandas as pd
import requests

from .betting import expected_value, kelly
from .model import Model, norm_cdf

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
ODDS_COLUMNS = ["home_odds", "away_odds", "over_odds", "under_odds"]

_ABBREV = [
    (r"\bst\b\.?", "state"), (r"\bmt\b\.?", "mount"), (r"\buniv\b\.?", "university"),
    (r"&", "and"), (r"[^a-z0-9 ]", ""), (r"\s+", " "),
]


def normalize(name: str) -> str:
    s = name.lower().replace("'", "").replace(".", " ")
    for pat, rep in _ABBREV:
        s = re.sub(pat, rep, s)
    return s.strip()


class TeamResolver:
    """Maps the many spellings sportsbooks use onto ESPN team ids."""

    def __init__(self, directory: pd.DataFrame, aliases: dict[str, int] | None = None):
        self.names: dict[str, int] = {}
        self.display: dict[int, str] = {}
        # Least specific first so fuller names win collisions ("Miami" vs "Miami (OH)").
        for col in ("abbreviation", "location", "short_name", "display_name"):
            for tid, name in zip(directory["team_id"], directory[col]):
                if isinstance(name, str) and name:
                    self.names[normalize(name)] = int(tid)
        for tid, name in zip(directory["team_id"], directory["display_name"]):
            self.display[int(tid)] = name
        for alias, tid in (aliases or {}).items():
            self.names[normalize(alias)] = tid

    def resolve(self, name: str) -> int:
        key = normalize(name)
        if key in self.names:
            return self.names[key]
        close = difflib.get_close_matches(key, self.names.keys(), n=1, cutoff=0.85)
        if close:
            return self.names[close[0]]
        raise KeyError(f"Can't match team name {name!r}; add it to aliases or use the ESPN name")


def read_lines_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"home", "away"} - set(df.columns)
    if missing:
        raise ValueError(f"lines file is missing columns: {sorted(missing)}")
    for col in ODDS_COLUMNS:
        df[col] = df[col].fillna(-110) if col in df else -110
    for col in ("spread", "total", "neutral", "date"):
        if col not in df:
            df[col] = pd.NA
    return df


def fetch_odds_api(api_key: str, book: str | None = None) -> pd.DataFrame:
    """Current NCAAB spreads and totals. Uses one book if given, else the consensus (median) line."""
    resp = requests.get(ODDS_API_URL, params={
        "apiKey": api_key, "regions": "us", "markets": "spreads,totals", "oddsFormat": "american",
    }, timeout=30)
    resp.raise_for_status()
    rows = []
    for ev in resp.json():
        home, away = ev["home_team"], ev["away_team"]
        spreads, totals = [], []
        for bm in ev.get("bookmakers", []):
            if book and bm["key"] != book:
                continue
            for mkt in bm["markets"]:
                outs = {o["name"]: o for o in mkt["outcomes"]}
                if mkt["key"] == "spreads" and home in outs and away in outs:
                    spreads.append((outs[home]["point"], outs[home]["price"], outs[away]["price"]))
                elif mkt["key"] == "totals" and "Over" in outs and "Under" in outs:
                    totals.append((outs["Over"]["point"], outs["Over"]["price"], outs["Under"]["price"]))
        if not spreads and not totals:
            continue
        rows.append({
            "date": ev["commence_time"][:10], "home": home, "away": away, "neutral": pd.NA,
            "spread": median(s[0] for s in spreads) if spreads else pd.NA,
            "total": median(t[0] for t in totals) if totals else pd.NA,
            "home_odds": spreads[0][1] if book and spreads else -110,
            "away_odds": spreads[0][2] if book and spreads else -110,
            "over_odds": totals[0][1] if book and totals else -110,
            "under_odds": totals[0][2] if book and totals else -110,
        })
    return pd.DataFrame(rows)


def find_edges(model: Model, lines: pd.DataFrame, resolver: TeamResolver,
               neutral_lookup: dict[tuple[int, int], bool] | None = None,
               kelly_fraction: float = 0.25, trust: float = 0.3) -> pd.DataFrame:
    """Score every spread and total against the model. One row per (game, market).

    Closing lines are sharper than any results-only model, so probabilities come from
    the market number moved `trust` of the way toward the model's, not from the model
    alone. trust=1 takes the model at face value, which wildly overstates edges.
    """
    neutral_lookup = neutral_lookup or {}
    out = []
    for row in lines.itertuples(index=False):
        h, a = resolver.resolve(str(row.home)), resolver.resolve(str(row.away))
        if pd.notna(row.neutral):
            neutral = bool(int(row.neutral))
        else:
            # Books don't flag neutral sites; trust the ESPN schedule. If the schedule has the
            # teams the other way round, the book's "home" team isn't really at home.
            neutral = neutral_lookup.get((h, a), (a, h) in neutral_lookup)
        p = model.predict(h, a, neutral)
        base = {"date": row.date, "home": resolver.display.get(h, row.home),
                "away": resolver.display.get(a, row.away), "neutral": neutral,
                "model_spread": round(p.spread, 1), "model_total": round(p.total, 1),
                "home_win_prob": p.home_win_prob}

        if pd.notna(row.spread):
            spread = float(row.spread)
            p_home = norm_cdf(trust * (p.margin + spread) / p.margin_sigma)
            sides = [(f"{base['home']} {spread:+g}", p_home, row.home_odds),
                     (f"{base['away']} {-spread:+g}", 1 - p_home, row.away_odds)]
            pick, prob, odds = max(sides, key=lambda s: expected_value(s[1], s[2]))
            out.append({**base, "market": "spread", "line": spread, "pick": pick,
                        "edge_pts": abs(p.margin + spread), "prob": prob, "odds": odds,
                        "ev": expected_value(prob, odds), "kelly": kelly(prob, odds, kelly_fraction)})

        if pd.notna(row.total):
            total = float(row.total)
            p_over = norm_cdf(trust * (p.total - total) / p.total_sigma)
            sides = [(f"Over {total:g}", p_over, row.over_odds), (f"Under {total:g}", 1 - p_over, row.under_odds)]
            pick, prob, odds = max(sides, key=lambda s: expected_value(s[1], s[2]))
            out.append({**base, "market": "total", "line": total, "pick": pick,
                        "edge_pts": abs(p.total - total), "prob": prob, "odds": odds,
                        "ev": expected_value(prob, odds), "kelly": kelly(prob, odds, kelly_fraction)})

    df = pd.DataFrame(out)
    return df.sort_values("ev", ascending=False).reset_index(drop=True) if not df.empty else df
