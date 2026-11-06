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
python -m boynton                       # baseline point estimate, KenPom and Torvik
python -m boynton --source torvik       # one source only
python -m boynton --rating-sd 3         # add season-long rating uncertainty
python -m boynton --force 2026-11-06=W  # what-if: Michigan beats UConn
python -m boynton --force 2026-11-06=+10 --json out/summary.json
python -m pytest -q
```

## Inputs

| File | What | In git? |
|---|---|---|
| `data/ratings.csv` | KenPom front page, hand-entered: `rank,team,conf,netrtg,adjt,as_of` | **No.** KenPom's terms forbid republishing |
| `data/ratings.torvik.csv` | Bart Torvik T-Rank, same header (`netrtg` = AdjOE - AdjDE) | **No** |
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
- **NCAA.** Resume = rating + 1 point per win above what that sim's ratings
  predicted (the "luck"). Michigan's luck is simulated; every other team gets a
  random luck term with the same spread. Top 44 in resume get an at-large bid
  (BTT champ is in automatically). Seed = ceil(rank / 4), capped at 12.
  Opponents in the R64 and R32 come from the right seed lines (seeds 12 to 16 use
  typical auto-bid strength). Make the Sweet 16 = win both. On Torvik inputs this
  gives a mean seed of 4.8 and 92.5% to make the field (with `--rating-sd 3`);
  Torvik's own page shows 4.8 and 89.4%.
- **Played games** use actual results. `--force DATE=W|L|+margin` sets a future
  Michigan game for what-if analysis.
- **Leverage table.** For each remaining game: P(win), P(extension | win),
  P(extension | loss), computed by conditioning the same simulations.

## Uncertainty hooks

`boynton/sim.py` has two functions meant to be replaced by the Bayesian work:

- `update_ratings(ratings, played_games)`: returns updated ratings. It's a no-op
  for now, because the KenPom table already includes the games played before its date.
- `draw_ratings(base, n, rng, cfg)`: per-simulation team ratings. With
  `--rating-sd > 0`, each team gets one season-long shift per sim. Once that's on,
  the leverage table picks up the *information* in a result (a win over UConn makes
  "Michigan is good" sims more likely), on top of the win itself.

Only probabilities are ever written out (`--json`); ratings never are. The JSON's
top level is the KenPom summary (or whichever single `--source` was run), each
summary carries `source` and `sample_data`, and `by_source` holds every source run.
