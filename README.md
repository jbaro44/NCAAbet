# ncaabet

A college basketball spread and total model. It builds opponent-adjusted team ratings from every D1 game since 2021-22, projects scores for any matchup, and compares its numbers to sportsbook lines to find bets worth a look.

```
$ ncaabet predict Duke "North Carolina"
North Carolina Tar Heels vs Duke Blue Devils (at Duke Blue Devils)
  Projected score: Duke Blue Devils 80.2, North Carolina Tar Heels 67.4
  Model line:      Duke Blue Devils -12.8   Total 147.6
  Win probability: Duke Blue Devils 86.5%, North Carolina Tar Heels 13.5%
```

## Setup

```
pip install -e .[dev]
ncaabet update        # downloads ~10 MB of results into ./data and builds ratings (about 15s)
```

Game data comes from the [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) ESPN mirrors. Set `NCAABET_DATA` to keep it somewhere other than `./data`.

## Commands

| Command | What it does |
| --- | --- |
| `ncaabet ratings --top 25` | Power ratings: adjusted offense, defense, tempo, and net efficiency |
| `ncaabet predict HOME AWAY [--neutral]` | Projected score, line, total, and win probability |
| `ncaabet slate [--date 2026-11-02] [--days 3] [--csv out.csv]` | Model lines for every scheduled game |
| `ncaabet edges --lines FILE.csv` | Compare the model to your lines; shows +EV bets with Kelly stakes |
| `ncaabet edges` (with `ODDS_API_KEY` set) | Same, using live odds from [The Odds API](https://the-odds-api.com) (free tier works) |
| `ncaabet backtest [--lines historical.csv]` | Walk-forward accuracy, calibration, and ATS record if you supply past lines |

`--season 2026` before any command works in a past season instead of the current one.

### Lines file

```csv
date,home,away,neutral,spread,total,home_odds,away_odds,over_odds,under_odds
2026-11-02,Florida Gators,Miami Hurricanes,1,-7.5,152.5,-110,-110,-110,-110
2026-11-02,Kansas,Fordham,,-17.5,141.5,,,,
```

Only `home` and `away` are required. `spread` is the home team's number (-7.5 means the home team is favored by 7.5). Blank odds default to -110, and a blank `neutral` is filled in from the ESPN schedule. Team names are matched loosely ("St Johns", "UConn", and "Iowa State" all work). See `examples/lines_example.csv`.

## How it works

1. **Adjusted efficiency** (`ratings.py`). Each team gets an offensive rating, a defensive rating, and a tempo, solved jointly by weighted least squares. Older games count less (120-day half-life). Ratings start the season at 80% of last year's final numbers and update as games come in. This is the same idea KenPom and Torvik use.
2. **Elo** with margin of victory, as a second opinion. It gets 10% of the blend. The backtest showed it adds little on top of efficiency ratings.
3. **Projection.** Points = (league average + offense + opponent's defense ± home court) × expected possessions. The margin and total come from those points. Win and cover probabilities assume errors are normally distributed, with standard deviations measured in the backtest (11.6 points for the margin, 17.0 for the total).
4. **Edges** (`lines.py`). A betting line already holds more information than this model, which only sees scores and box scores. So the probability for each bet starts at the market number and moves `--trust` (default 0.3) of the way toward the model. A 4-point disagreement then becomes about 54% rather than 64%. Raise `--trust` only after the model shows an ATS profit in a backtest against real lines.

Every prediction in the backtest is made the morning of the game, using only earlier results.

## Backtest (2022-23 through 2025-26, both teams with 5+ games played)

| Season | Games | Margin MAE | Total MAE | Winner picked | Brier |
| --- | --- | --- | --- | --- | --- |
| 2023 | 4,722 | 8.92 | 13.10 | 70.9% | 0.189 |
| 2024 | 4,740 | 9.01 | 13.27 | 71.1% | 0.188 |
| 2025 | 4,772 | 8.92 | 12.96 | 71.6% | 0.182 |
| 2026 | 4,769 | 9.06 | 13.72 | 70.2% | 0.188 |

Win probabilities are well calibrated: teams given 70-80% won 74.1% of the time, and teams given 80-90% won 85.7%.

For comparison, closing lines in college basketball usually miss the final margin by about 8.5 points on average. So the model is close to the market, not better than it. **ATS profitability is unproven** until you run `ncaabet backtest --lines` with historical closing lines. The free data used here has no odds.

## Known limits / ideas

- Preseason ratings don't know about the transfer portal or recruiting. Early-season numbers are last year's team, shrunk toward average.
- No injury, travel, or rest adjustments.
- Ideas: historical closing lines (to fit `--trust` and test ATS), roster-continuity priors, a March Madness simulator built on `Model.predict`, and tracking closing-line value on your bets.

Bet responsibly. This is a hobby model, not financial advice.
