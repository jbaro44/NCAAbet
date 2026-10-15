"""Team rating systems.

EfficiencyRatings: opponent-adjusted offense/defense (points per 100 possessions)
and tempo, solved by iterative weighted least squares with recency weighting and
a Bayesian prior carried over from last season. Same idea as KenPom/Torvik.

Elo: margin-of-victory Elo, a second, differently-shaped opinion for blending.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Prior for D1 teams with no history (programs moving up from D2/FCS):
# well below average, about 12 points per 100 possessions worse.
UNKNOWN_OFF = -6.0
UNKNOWN_DEF = 6.0


@dataclass
class TeamRatings:
    """Snapshot of efficiency ratings. off/deff are relative to league average (mu)."""

    off: dict[int, float]
    deff: dict[int, float]  # positive = leaky defense
    tempo: dict[int, float]
    mu: float = 104.0  # league points per 100 possessions
    tau: float = 68.0  # league possessions per game
    hca: float = 1.6  # home offense boost, points per 100 possessions (applied to both sides)
    unknown: tuple[float, float] = (0.0, 0.0)

    def team(self, tid: int) -> tuple[float, float, float]:
        return (self.off.get(tid, self.unknown[0]), self.deff.get(tid, self.unknown[1]),
                self.tempo.get(tid, 0.0))

    def net(self, tid: int) -> float:
        o, d, _ = self.team(tid)
        return o - d

    def predict(self, home: int, away: int, neutral: bool) -> tuple[float, float, float]:
        """Return (home points, away points, possessions)."""
        ho, hd, ht = self.team(home)
        ao, ad, at = self.team(away)
        loc = 0.0 if neutral else 1.0
        poss = self.tau + ht + at
        home_pts = (self.mu + ho + ad + self.hca * loc) * poss / 100
        away_pts = (self.mu + ao + hd - self.hca * loc) * poss / 100
        return home_pts, away_pts, poss

    def regressed(self, keep: float) -> "TeamRatings":
        """Carry ratings into a new season, shrinking toward average."""
        return TeamRatings(
            off={t: v * keep for t, v in self.off.items()},
            deff={t: v * keep for t, v in self.deff.items()},
            tempo={t: v * keep for t, v in self.tempo.items()},
            mu=self.mu, tau=self.tau, hca=self.hca,
            unknown=(UNKNOWN_OFF, UNKNOWN_DEF),
        )

    def table(self, names: dict[int, str] | None = None) -> pd.DataFrame:
        ids = sorted(self.off)
        df = pd.DataFrame({
            "team_id": ids,
            "adj_off": [self.mu + self.off[t] for t in ids],
            "adj_def": [self.mu + self.deff[t] for t in ids],
            "adj_tempo": [self.tau + self.tempo[t] for t in ids],
        })
        df["adj_em"] = df["adj_off"] - df["adj_def"]
        if names is not None:
            df.insert(1, "team", df["team_id"].map(names))
        return df.sort_values("adj_em", ascending=False).reset_index(drop=True)


@dataclass
class EfficiencyRatings:
    half_life_days: float = 120.0
    prior_weight: float = 3.0  # prior counts as this many full-weight games
    iterations: int = 40

    def fit(self, games: pd.DataFrame, as_of, prior: TeamRatings | None = None) -> TeamRatings:
        prior = prior or TeamRatings({}, {}, {})
        if games.empty:
            return prior

        teams = np.unique(np.concatenate([games["home_id"].to_numpy(), games["away_id"].to_numpy()]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = games["home_id"].map(idx).to_numpy()
        a = games["away_id"].map(idx).to_numpy()

        poss = (games["home_poss"].to_numpy() + games["away_poss"].to_numpy()) / 2
        age = (pd.Timestamp(as_of) - games["date"]).dt.days.to_numpy()
        w = 0.5 ** (age / self.half_life_days)
        loc = np.where(games["neutral"].to_numpy(), 0.0, 1.0)

        # Two offensive observations per game: home offense vs away defense and vice versa.
        off_t = np.concatenate([h, a])
        def_t = np.concatenate([a, h])
        ppp = 100 * np.concatenate([games["home_score"], games["away_score"]]) / np.concatenate([poss, poss])
        side = np.concatenate([loc, -loc])
        ww = np.concatenate([w, w])

        o0 = np.array([prior.off.get(t, prior.unknown[0]) for t in teams])
        d0 = np.array([prior.deff.get(t, prior.unknown[1]) for t in teams])
        p0 = np.array([prior.tempo.get(t, 0.0) for t in teams])
        lam = self.prior_weight
        o, d, p = o0.copy(), d0.copy(), p0.copy()
        mu, hca, tau = prior.mu, prior.hca, prior.tau

        w_off = np.bincount(off_t, ww, n) + lam
        w_def = np.bincount(def_t, ww, n) + lam
        w_tmp = np.bincount(h, w, n) + np.bincount(a, w, n) + lam
        for _ in range(self.iterations):
            r = ppp - mu - d[def_t] - hca * side
            o = (np.bincount(off_t, ww * r, n) + lam * o0) / w_off
            r = ppp - mu - o[off_t] - hca * side
            d = (np.bincount(def_t, ww * r, n) + lam * d0) / w_def
            r = ppp - o[off_t] - d[def_t] - hca * side
            mu = (np.sum(ww * r) + 50 * prior.mu) / (np.sum(ww) + 50)
            r = ppp - mu - o[off_t] - d[def_t]
            hca = (np.sum(ww * side * r) + 200 * prior.hca) / (np.sum(ww * side**2) + 200)

            r = poss - tau - p[a]
            rr = poss - tau - p[h]
            p = (np.bincount(h, w * r, n) + np.bincount(a, w * rr, n) + lam * p0) / w_tmp
            tau = (np.sum(w * (poss - p[h] - p[a])) + 50 * prior.tau) / (np.sum(w) + 50)

        out = TeamRatings(dict(prior.off), dict(prior.deff), dict(prior.tempo),
                          mu=float(mu), tau=float(tau), hca=float(hca), unknown=prior.unknown)
        for t, i in idx.items():
            out.off[int(t)] = float(o[i])
            out.deff[int(t)] = float(d[i])
            out.tempo[int(t)] = float(p[i])
        return out


@dataclass
class Elo:
    k: float = 38.0
    hca: float = 85.0  # Elo points
    points_per_elo: float = 1 / 28.0
    carryover: float = 0.7
    unknown_rating: float = 1500.0
    ratings: dict[int, float] = field(default_factory=dict)

    def rating(self, tid: int) -> float:
        return self.ratings.get(tid, self.unknown_rating)

    def diff(self, home: int, away: int, neutral: bool) -> float:
        return self.rating(home) - self.rating(away) + (0.0 if neutral else self.hca)

    def margin(self, home: int, away: int, neutral: bool) -> float:
        return self.diff(home, away, neutral) * self.points_per_elo

    def update(self, games: pd.DataFrame) -> None:
        """Update with a batch of games played on the same day (all use pre-day ratings)."""
        deltas: dict[int, float] = {}
        for g in games.itertuples(index=False):
            d = self.diff(g.home_id, g.away_id, g.neutral)
            expected = 1 / (1 + 10 ** (-d / 400))
            mov = g.home_score - g.away_score
            won = 1.0 if mov > 0 else 0.0
            # 538-style multiplier: bigger wins count more, but less so when the favorite wins big.
            winner_diff = d if mov > 0 else -d
            mult = np.log(abs(mov) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            delta = self.k * mult * (won - expected)
            deltas[g.home_id] = deltas.get(g.home_id, 0.0) + delta
            deltas[g.away_id] = deltas.get(g.away_id, 0.0) - delta
        for t, dv in deltas.items():
            self.ratings[t] = self.rating(t) + dv

    def new_season(self) -> None:
        if self.ratings:
            mean = float(np.mean(list(self.ratings.values())))
            self.ratings = {t: mean + (r - mean) * self.carryover for t, r in self.ratings.items()}
            # Teams we haven't seen yet are programs newly moving up to D1.
            self.unknown_rating = mean - 250
