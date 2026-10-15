"""Score walk-forward predictions against results, and (given historical lines) against the market."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .betting import profit
from .lines import TeamResolver, read_lines_csv


def accuracy_report(preds: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    """Per-season accuracy on games where both teams have at least min_games of history."""
    q = preds[(preds["home_games_before"] >= min_games) & (preds["away_games_before"] >= min_games)]

    def stats(g: pd.DataFrame) -> pd.Series:
        err = g["margin"] - g["pred_margin"]
        terr = g["total"] - g["pred_total"]
        won = (g["margin"] > 0).astype(float)
        return pd.Series({
            "games": len(g),
            "margin_mae": err.abs().mean(),
            "margin_rmse": np.sqrt((err ** 2).mean()),
            "total_mae": terr.abs().mean(),
            "winner_pct": ((g["pred_margin"] > 0) == (g["margin"] > 0)).mean(),
            "brier": ((g["home_win_prob"] - won) ** 2).mean(),
        })

    by_season = q.groupby("season").apply(stats, include_groups=False)
    by_season.loc["all"] = stats(q)
    by_season["games"] = by_season["games"].astype(int)
    return by_season


def calibration(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Do 70% favorites win 70% of the time?"""
    fav_prob = np.maximum(preds["home_win_prob"], 1 - preds["home_win_prob"])
    fav_won = np.where(preds["home_win_prob"] >= 0.5, preds["margin"] > 0, preds["margin"] < 0)
    df = pd.DataFrame({"prob": fav_prob, "won": fav_won})
    df["bucket"] = pd.cut(df["prob"], np.linspace(0.5, 1.0, bins // 2 + 1), include_lowest=True)
    return df.groupby("bucket", observed=True).agg(games=("won", "size"), predicted=("prob", "mean"),
                                                   actual=("won", "mean"))


def ats_report(preds: pd.DataFrame, lines_path, resolver: TeamResolver,
               thresholds=(0, 1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Against-the-spread record by model edge, using a CSV of historical closing lines (needs a date column)."""
    lines = read_lines_csv(lines_path).dropna(subset=["spread", "date"])
    lines["date"] = pd.to_datetime(lines["date"])
    lines["home_id"] = [resolver.resolve(str(n)) for n in lines["home"]]
    lines["away_id"] = [resolver.resolve(str(n)) for n in lines["away"]]
    m = preds.merge(lines[["date", "home_id", "away_id", "spread", "home_odds", "away_odds"]],
                    on=["date", "home_id", "away_id"], how="inner")
    if m.empty:
        raise ValueError("No lines matched any predicted game; check dates and team names")

    m["edge"] = m["pred_margin"] + m["spread"]  # >0: model likes home against the number
    m["bet_home"] = m["edge"] > 0
    result = m["margin"] + m["spread"]
    m["won"] = np.where(result == 0, np.nan, np.where(m["bet_home"], result > 0, result < 0))
    m["odds"] = np.where(m["bet_home"], m["home_odds"], m["away_odds"])
    m["profit"] = [profit(None if np.isnan(w) else bool(w), o) for w, o in zip(m["won"], m["odds"])]

    rows = []
    for t in thresholds:
        b = m[m["edge"].abs() >= t]
        decided = b["won"].dropna()
        rows.append({"min_edge": t, "bets": len(b), "wins": int(decided.sum()),
                     "losses": int(len(decided) - decided.sum()), "pushes": int(b["won"].isna().sum()),
                     "win_pct": decided.mean() if len(decided) else np.nan,
                     "units": b["profit"].sum(), "roi": b["profit"].sum() / len(b) if len(b) else np.nan})
    return pd.DataFrame(rows)
