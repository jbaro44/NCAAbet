"""Walk-forward prediction engine: efficiency ratings + Elo blended into a spread and total.

Every prediction is made using only games played before that day, so the
output of `run` is an honest out-of-sample record usable for backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import pandas as pd

from .ratings import Elo, EfficiencyRatings, TeamRatings


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


@dataclass
class Prediction:
    home_id: int
    away_id: int
    neutral: bool
    margin: float  # home minus away
    total: float
    eff_margin: float
    elo_margin: float
    margin_sigma: float
    total_sigma: float

    @property
    def home_pts(self) -> float:
        return (self.total + self.margin) / 2

    @property
    def away_pts(self) -> float:
        return (self.total - self.margin) / 2

    @property
    def spread(self) -> float:
        """The model's line, written from the home side like a sportsbook (-5.5 = home favored)."""
        return -self.margin

    @property
    def home_win_prob(self) -> float:
        return norm_cdf(self.margin / self.margin_sigma)

    def home_cover_prob(self, home_spread: float) -> float:
        """P(home covers home_spread). Pushes are treated as half a cover either way."""
        return norm_cdf((self.margin + home_spread) / self.margin_sigma)

    def over_prob(self, total_line: float) -> float:
        return norm_cdf((self.total - total_line) / self.total_sigma)


@dataclass
class Model:
    efficiency: EfficiencyRatings = field(default_factory=EfficiencyRatings)
    elo: Elo = field(default_factory=Elo)
    eff_weight: float = 0.9  # share of the blended margin that comes from efficiency ratings
    carryover: float = 0.8  # how much of last season's efficiency rating survives the offseason
    margin_sigma: float = 11.6
    total_sigma: float = 17.0

    ratings: TeamRatings | None = None
    season: int | None = None
    _season_start: TeamRatings | None = None

    def start_season(self, season: int) -> None:
        if self.ratings is not None:
            self._season_start = self.ratings.regressed(self.carryover)
        else:
            self._season_start = None
        self.ratings = self._season_start
        self.elo.new_season()
        self.season = season

    def predict(self, home: int, away: int, neutral: bool) -> Prediction:
        r = self.ratings or TeamRatings({}, {}, {})
        hp, ap, _ = r.predict(home, away, neutral)
        eff_margin = hp - ap
        elo_margin = self.elo.margin(home, away, neutral)
        margin = self.eff_weight * eff_margin + (1 - self.eff_weight) * elo_margin
        return Prediction(home, away, bool(neutral), margin, hp + ap, eff_margin, elo_margin,
                          self.margin_sigma, self.total_sigma)

    def run(self, games: pd.DataFrame, progress=None) -> pd.DataFrame:
        """Walk forward through games day by day, predicting each before learning from it."""
        records = []
        for season, sg in games.groupby("season", sort=True):
            self.start_season(int(season))
            sg = sg.sort_values("date")
            dates = sg["date"].drop_duplicates().tolist()
            played_count: dict[int, int] = {}
            for i, day in enumerate(dates):
                played = sg[sg["date"] < day]
                self.ratings = self.efficiency.fit(played, day, self._season_start)
                todays = sg[sg["date"] == day]
                for g in todays.itertuples(index=False):
                    p = self.predict(g.home_id, g.away_id, g.neutral)
                    records.append({
                        "game_id": g.game_id, "season": season, "date": day,
                        "home_id": g.home_id, "away_id": g.away_id,
                        "home_name": g.home_name, "away_name": g.away_name,
                        "neutral": g.neutral, "postseason": g.postseason,
                        "pred_margin": p.margin, "pred_total": p.total,
                        "eff_margin": p.eff_margin, "elo_margin": p.elo_margin,
                        "home_win_prob": p.home_win_prob,
                        "margin": g.home_score - g.away_score,
                        "total": g.home_score + g.away_score,
                        "home_games_before": played_count.get(g.home_id, 0),
                        "away_games_before": played_count.get(g.away_id, 0),
                    })
                self.elo.update(todays)
                for t in pd.concat([todays["home_id"], todays["away_id"]]):
                    played_count[t] = played_count.get(t, 0) + 1
                if progress:
                    progress(season, i + 1, len(dates))
            self.ratings = self.efficiency.fit(sg, dates[-1] + pd.Timedelta(days=1), self._season_start)
        return pd.DataFrame(records)
