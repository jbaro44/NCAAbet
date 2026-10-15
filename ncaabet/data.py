"""Download, cache, and tidy NCAA men's basketball results.

Source: sportsdataverse's public ESPN mirrors (https://github.com/sportsdataverse/sportsdataverse-data).
Each season ships as a schedule parquet (venue, neutral site, status) and a
team box-score parquet (one row per team per game).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SCHEDULE_URL = RELEASES + "/espn_mens_college_basketball_schedules/mbb_schedule_{season}.parquet"
TEAM_BOX_URL = RELEASES + "/espn_mens_college_basketball_team_boxscores/team_box_{season}.parquet"

# D1 teams play ~30 games; non-D1 opponents appear in at most a few.
D1_MIN_GAMES = 10

GAME_COLUMNS = [
    "game_id", "season", "date", "neutral", "postseason",
    "home_id", "away_id", "home_name", "away_name",
    "home_score", "away_score", "home_poss", "away_poss",
]


def data_dir() -> Path:
    path = Path(os.environ.get("NCAABET_DATA", Path.cwd() / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(url: str, dest: Path, refresh: bool = False) -> Path | None:
    """Download url to dest unless cached. Returns None when the file doesn't exist upstream."""
    if dest.exists() and not refresh:
        return dest
    resp = requests.get(url, timeout=120)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def load_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    path = _fetch(SCHEDULE_URL.format(season=season), data_dir() / f"schedule_{season}.parquet", refresh)
    if path is None:
        raise FileNotFoundError(f"No schedule published for season {season}")
    return pd.read_parquet(path)


def load_team_box(season: int, refresh: bool = False) -> pd.DataFrame | None:
    path = _fetch(TEAM_BOX_URL.format(season=season), data_dir() / f"team_box_{season}.parquet", refresh)
    return None if path is None else pd.read_parquet(path)


def possessions(fga, oreb, tov, fta):
    """Standard possession estimate."""
    return fga - oreb + tov + 0.475 * fta


def build_games(schedule: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """One row per completed game with scores and possession estimates for both sides."""
    box = box.copy()
    box["poss"] = possessions(
        box["field_goals_attempted"], box["offensive_rebounds"],
        box["total_turnovers"], box["free_throws_attempted"],
    )
    cols = ["game_id", "team_id", "team_display_name", "team_score", "poss"]
    home = box.loc[box["team_home_away"] == "home", cols]
    away = box.loc[box["team_home_away"] == "away", cols]
    g = home.merge(away, on="game_id", suffixes=("_h", "_a"))
    g = g.rename(columns={
        "team_id_h": "home_id", "team_id_a": "away_id",
        "team_display_name_h": "home_name", "team_display_name_a": "away_name",
        "team_score_h": "home_score", "team_score_a": "away_score",
        "poss_h": "home_poss", "poss_a": "away_poss",
    })

    sched = schedule[["game_id", "season", "game_date", "neutral_site", "season_type"]].drop_duplicates("game_id")
    g = g.merge(sched, on="game_id", how="inner")
    g["date"] = pd.to_datetime(g["game_date"])
    g["neutral"] = g["neutral_site"].fillna(False).astype(bool)
    g["postseason"] = g["season_type"] == 3

    # Drop games with broken box scores (missing stats or nonsense possession counts).
    ok = (g["home_poss"].between(40, 110) & g["away_poss"].between(40, 110)
          & (g["home_score"] > 0) & (g["away_score"] > 0))
    # Only D1 vs D1. Non-D1 opponents show up in a handful of games each and are
    # wildly mismatched; left in, they distort ratings and the home-court estimate.
    counts = box["team_id"].value_counts()
    d1 = set(counts[counts >= D1_MIN_GAMES].index)
    ok &= g["home_id"].isin(d1) & g["away_id"].isin(d1)
    g = g.loc[ok, GAME_COLUMNS]
    g[["home_id", "away_id"]] = g[["home_id", "away_id"]].astype(int)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def load_games(seasons, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for season in seasons:
        box = load_team_box(season, refresh)
        if box is None:
            continue
        frames.append(build_games(load_schedule(season, refresh), box))
    if not frames:
        return pd.DataFrame(columns=GAME_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_upcoming(season: int, refresh: bool = False) -> pd.DataFrame:
    """Scheduled games that haven't been played yet."""
    s = load_schedule(season, refresh)
    s = s.loc[~s["status_type_completed"].fillna(False).astype(bool)]
    out = pd.DataFrame({
        "game_id": s["game_id"],
        "season": s["season"],
        "date": pd.to_datetime(s["game_date"]),
        "tipoff": s["game_date_time"],
        "neutral": s["neutral_site"].fillna(False).astype(bool),
        "home_id": s["home_id"].astype(int),
        "away_id": s["away_id"].astype(int),
        "home_name": s["home_display_name"],
        "away_name": s["away_display_name"],
    })
    return out.sort_values(["tipoff", "game_id"]).reset_index(drop=True)


def team_directory(seasons, refresh: bool = False) -> pd.DataFrame:
    """Every name variant we know for each team, used to match sportsbook names to ESPN ids."""
    rows = []
    for season in seasons:
        try:
            s = load_schedule(season, refresh)
        except FileNotFoundError:
            continue
        for side in ("home", "away"):
            cols = {f"{side}_id": "team_id", f"{side}_display_name": "display_name",
                    f"{side}_short_display_name": "short_name", f"{side}_location": "location",
                    f"{side}_abbreviation": "abbreviation"}
            rows.append(s[list(cols)].rename(columns=cols).assign(season=season))
    teams = pd.concat(rows, ignore_index=True).dropna(subset=["team_id"])
    teams["team_id"] = teams["team_id"].astype(int)
    # Keep the most recent naming for each team.
    return teams.sort_values("season").drop_duplicates("team_id", keep="last").reset_index(drop=True)


def season_for_date(date) -> int:
    """NCAA seasons are named for the year they end in (Nov 2026 -> 2027)."""
    date = pd.Timestamp(date)
    return date.year + 1 if date.month >= 8 else date.year


__all__ = [
    "load_games", "load_upcoming", "load_schedule", "team_directory",
    "build_games", "possessions", "season_for_date", "data_dir",
]
