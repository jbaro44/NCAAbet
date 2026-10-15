"""Odds math: implied probability, expected value, and Kelly sizing."""

from __future__ import annotations


def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def implied_prob(odds: float) -> float:
    return 1 / american_to_decimal(odds)


def no_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Fair probabilities for a two-way market with the bookmaker's margin removed."""
    a, b = implied_prob(odds_a), implied_prob(odds_b)
    return a / (a + b), b / (a + b)


def expected_value(prob: float, odds: float) -> float:
    """Expected profit per 1 unit staked."""
    return prob * (american_to_decimal(odds) - 1) - (1 - prob)


def kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Fraction of bankroll to stake. Full Kelly is far too aggressive for a noisy model, hence fraction."""
    b = american_to_decimal(odds) - 1
    full = (b * prob - (1 - prob)) / b
    return max(0.0, full * fraction)


def profit(won: bool | None, odds: float, stake: float = 1.0) -> float:
    """Profit on a settled bet; None means push."""
    if won is None:
        return 0.0
    return stake * (american_to_decimal(odds) - 1) if won else -stake


BREAKEVEN_110 = implied_prob(-110)  # 52.38%
