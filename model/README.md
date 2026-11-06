# Boynton extension model

Monte Carlo estimate of the probability that Mike Boynton (Michigan, interim head
coach) hits **at least 2 of 4** benchmarks that trigger an automatic extension:

1. 24+ regular-season wins
2. Top-4 finish in the Big Ten regular season
3. Big Ten Tournament title
4. Sweet 16 berth

## Run it

```bash
cd model
pip install -e .[dev]
python -m boynton                       # baseline point estimate
python -m boynton --rating-sd 3         # add season-long rating uncertainty
python -m boynton --force 2026-11-06=W  # what-if: Michigan beats UConn
python -m boynton --force 2026-11-06=+10 --json out/summary.json
python -m pytest -q
```

## Inputs

| File | What | In git? |
|---|---|---|
| `data/ratings.csv` | KenPom front page, hand-entered: `rank,team,conf,netrtg,adjt,as_of` | **No.** KenPom's terms forbid republishing |
| `data/ratings.sample.csv` | Perturbed, illustrative numbers so the pipeline runs without the real file | Yes |
| `data/michigan_schedule.csv` | All 30 games; fill in `result` (e.g. `W 84-68`, Michigan first) as games finish | Yes |
| `data/big_ten_schedule.csv` | *Optional* full 180-game league schedule (`date,home,away,home_score,away_score`) | Yes, once added |

`--ratings path.tsv` also accepts a raw copy-paste of the KenPom front-page table.
Teams missing from a partial ratings table get a fallback NetRtg of -7 (about the
250th team), and the report lists them.

## How it works

- **Games.** Expected margin = (NetRtg_home - NetRtg_away) x possessions / 100 + 3.2
  home court (0 at neutral sites), possessions = mean AdjT. Actual margin ~ Normal(expected, 11).
- **Regular season.** Michigan's 30 games are simulated, and so is the whole Big Ten
  slate, because standings depend on everyone. Until the official league schedule is
  added, other teams' games are a synthetic but valid 20-game schedule (each pair
  meets once, extra games fill out to 20). Michigan's own league games are real.
- **Top 4.** Rank by conference wins; ties broken by league point differential,
  then at random. Real Big Ten tiebreakers are a TODO.
- **Big Ten Tournament.** 15-team bracket (seeds 16 to 18 miss out), neutral site.
- **NCAA.** Michigan's resume = its rating + 1 point per win above expectation;
  other teams' resumes = rating + Normal(0, 3) noise. Top 44 in resume get an at-large
  bid (BTT champ is in automatically). Seed = ceil(rank / 4), capped at 12.
  Opponents in the R64 and R32 come from the right seed lines (seeds 12 to 16 use
  typical auto-bid strength). Make the Sweet 16 = win both.
- **Played games** use actual results. `--force DATE=W|L|+margin` sets a future
  Michigan game for what-if analysis.
- **Leverage table.** For each remaining game: P(win), P(extension | win),
  P(extension | loss), computed by conditioning the same simulations.

## Rating uncertainty and what-ifs

- **Ratings as distributions.** `--rating-sd S` gives every team one season-long
  N(0, S) shift per sim (NetRtg points). S is an assumption, not yet calibrated:
  6 is a rough early-November guess, 3 a more confident one. Results are sensitive to it.
- **Updating from results** (`boynton/update.py`). Games played *after* the ratings
  file's `as_of` date update ratings with a normal-normal (Kalman) step on the margin
  vs. expectation, with surprises capped at 25 points. Games on or before `as_of`
  are skipped because KenPom already includes them. Replace the ratings file to
  re-anchor.
- **What-ifs** (`boynton/whatif.py`). A result moves P(extension) in two ways:
  - *direct*: the win (or loss, or margin) is banked and ratings are left alone.
    This re-runs the same sims with the game forced.
  - *information*: what the result says about how good the teams are. The forced
    sims are weighted by how likely the result was given each sim's drawn ratings.

  Information is 0 when `--rating-sd` is 0.

```bash
python -m boynton --rating-sd 6 --whatif 2026-11-06=W --whatif 2026-11-06=+10
python -m boynton --rating-sd 6 --decompose 5     # W and L for the next 5 games
```

Only probabilities are ever written out (`--json`); ratings never are.
