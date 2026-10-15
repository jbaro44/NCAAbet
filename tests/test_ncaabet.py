import numpy as np
import pandas as pd
import pytest

from ncaabet.backtest import accuracy_report, ats_report
from ncaabet.betting import american_to_decimal, expected_value, implied_prob, kelly, no_vig, profit
from ncaabet.lines import TeamResolver, find_edges, normalize
from ncaabet.model import Model
from ncaabet.ratings import EfficiencyRatings

# --- synthetic league -------------------------------------------------------

TRUE_NET = {1: 20.0, 2: 10.0, 3: 0.0, 4: -10.0, 5: -20.0, 6: 5.0}


def synthetic_games(seasons=(2024, 2025), rounds=6, noise=0.0, seed=0):
    """Round robins where each team's offense is half its net rating and defense the other half."""
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for season in seasons:
        day = pd.Timestamp(f"{season - 1}-11-05")
        for r in range(rounds):
            for h in TRUE_NET:
                for a in TRUE_NET:
                    if h >= a:
                        continue
                    home, away = (h, a) if r % 2 else (a, h)
                    poss = 70.0
                    hp = (104 + TRUE_NET[home] / 2 - TRUE_NET[away] / 2 + 2) * poss / 100
                    ap = (104 + TRUE_NET[away] / 2 - TRUE_NET[home] / 2 - 2) * poss / 100
                    gid += 1
                    rows.append({
                        "game_id": gid, "season": season, "date": day + pd.Timedelta(days=r * 3),
                        "neutral": False, "postseason": False, "home_id": home, "away_id": away,
                        "home_name": f"Team {home}", "away_name": f"Team {away}",
                        "home_score": round(hp + rng.normal(0, noise)), "away_score": round(ap + rng.normal(0, noise)),
                        "home_poss": poss, "away_poss": poss,
                    })
    return pd.DataFrame(rows)


def directory():
    return pd.DataFrame({
        "team_id": [1, 2, 3],
        "display_name": ["Saint Mary's Gaels", "St. John's Red Storm", "Miami (OH) RedHawks"],
        "short_name": ["Saint Mary's", "St. John's", "Miami (OH)"],
        "location": ["Saint Mary's", "St. John's", "Miami (OH)"],
        "abbreviation": ["SMC", "SJU", "M-OH"],
    })


# --- betting math -----------------------------------------------------------

def test_odds_conversions():
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert implied_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert sum(no_vig(-110, -110)) == pytest.approx(1.0)
    assert no_vig(-200, 170)[0] > 0.6


def test_ev_and_kelly():
    assert expected_value(implied_prob(-110), -110) == pytest.approx(0.0, abs=1e-12)
    assert expected_value(0.55, -110) > 0
    assert kelly(0.50, -110) == 0.0
    # Full Kelly at 55% / -110 is (0.909*0.55 - 0.45)/0.909 = 0.055.
    assert kelly(0.55, -110, fraction=1.0) == pytest.approx(0.055, abs=1e-3)
    assert profit(True, -110, 110) == pytest.approx(100)
    assert profit(False, -110) == -1
    assert profit(None, -110) == 0


# --- team names --------------------------------------------------------------

def test_name_resolution():
    r = TeamResolver(directory(), aliases={"SJU Johnnies": 2})
    assert normalize("St. John's") == normalize("St John's")
    assert r.resolve("Saint Mary's Gaels") == 1
    assert r.resolve("St Johns") == 2
    assert r.resolve("SJU Johnnies") == 2
    assert r.resolve("Miami OH") == 3
    with pytest.raises(KeyError):
        r.resolve("Gonzaga")


# --- ratings -----------------------------------------------------------------

def test_efficiency_recovers_true_ratings():
    games = synthetic_games(seasons=(2025,))
    r = EfficiencyRatings(prior_weight=0.1).fit(games, games["date"].max() + pd.Timedelta(days=1))
    for t, net in TRUE_NET.items():
        assert r.net(t) == pytest.approx(net, abs=1.0)
    hp, ap, poss = r.predict(1, 5, neutral=True)
    assert hp - ap == pytest.approx(40 * 0.70, abs=1.5)
    assert poss == pytest.approx(70, abs=0.5)
    # Home court should add roughly the 4 points per 100 built into the data.
    h_home, a_home, _ = r.predict(3, 3, neutral=False)
    assert (h_home - a_home) == pytest.approx(4 * 0.70, abs=0.6)


def test_walk_forward_never_peeks():
    games = synthetic_games()
    preds = Model().run(games)
    assert len(preds) == len(games)
    # With no history every team rates the same, so opening-day lines are just home court.
    first_day = preds[preds["date"] == preds["date"].min()]
    assert first_day["eff_margin"].nunique() == 1
    assert 1 < first_day["eff_margin"].iloc[0] < 4
    # By the second season the model has learned the ordering.
    late = preds[preds["season"] == 2025]
    assert ((late["pred_margin"] > 0) == (late["margin"] > 0)).mean() > 0.95


def test_accuracy_report_shape():
    preds = Model().run(synthetic_games(noise=5))
    rep = accuracy_report(preds, min_games=0)
    assert set(rep.index) == {2024, 2025, "all"}
    assert rep.loc["all", "games"] == len(preds)


# --- edges and ATS -------------------------------------------------------------

def _trained_model():
    model = Model()
    model.run(synthetic_games())
    return model


def test_find_edges_picks_the_right_side():
    model = _trained_model()
    names = pd.DataFrame({"team_id": list(TRUE_NET), "display_name": [f"Team {t}" for t in TRUE_NET],
                          "short_name": [f"T{t}" for t in TRUE_NET], "location": [f"Team {t}" for t in TRUE_NET],
                          "abbreviation": [f"T{t}" for t in TRUE_NET]})
    resolver = TeamResolver(names)
    p = model.predict(1, 5, True)
    lines = pd.DataFrame([{
        "date": "2025-12-01", "home": "Team 1", "away": "Team 5", "neutral": 1,
        "spread": -(p.margin - 6), "total": p.total + 8,  # line 6 pts too short; total 8 too high
        "home_odds": -110, "away_odds": -110, "over_odds": -110, "under_odds": -110,
    }])
    edges = find_edges(model, lines, resolver).set_index("market")
    assert edges.loc["spread", "pick"].startswith("Team 1")
    assert edges.loc["spread", "edge_pts"] == pytest.approx(6, abs=1e-6)
    assert edges.loc["total", "pick"].startswith("Under")
    assert (edges["ev"] > 0).all()
    # Trusting the model less shrinks the edge toward zero.
    cautious = find_edges(model, lines, resolver, trust=0.1).set_index("market")
    assert cautious.loc["spread", "prob"] < edges.loc["spread", "prob"]


def test_ats_report(tmp_path):
    preds = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"] * 3), "home_id": [1, 2, 3], "away_id": [4, 5, 6],
        "pred_margin": [10.0, -2.0, 4.0], "margin": [12, 3, 4],
    })
    pd.DataFrame({
        "date": ["2025-01-01"] * 3, "home": ["Team 1", "Team 2", "Team 3"], "away": ["Team 4", "Team 5", "Team 6"],
        "spread": [-7, -1, -4],
    }).to_csv(tmp_path / "lines.csv", index=False)
    names = pd.DataFrame({"team_id": range(1, 7), "display_name": [f"Team {t}" for t in range(1, 7)],
                          "short_name": None, "location": None, "abbreviation": None})
    rep = ats_report(preds, tmp_path / "lines.csv", TeamResolver(names), thresholds=(0, 2))
    # Game 1: model home by 3 vs line, home won by 12 -> win. Game 2: model away, home won by 3 -> loss.
    # Game 3: no edge (0) -> bets the away side, push.
    row = rep.set_index("min_edge").loc[0]
    assert (row["wins"], row["losses"], row["pushes"]) == (1, 1, 1)
    assert row["units"] == pytest.approx(100 / 110 - 1)
    assert rep.set_index("min_edge").loc[2, "bets"] == 2
