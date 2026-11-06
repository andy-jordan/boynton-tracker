# Boynton extension tracker

Tracks the probability that Michigan interim head coach Mike Boynton earns his automatic
extension by hitting 2 of 4 benchmarks: 24 regular-season wins, a top-4 Big Ten finish, a
Big Ten Tournament title, or a Sweet 16 berth.

- `model/`: Python Monte Carlo model (see `model/README.md`)
- `site/`: static site (Vite + TypeScript), reads `site/public/data/*.json`
- `scripts/append_history.py`: appends one row per day to `site/public/data/history.json`
- `.github/workflows/update.yml`: runs the model, records history, builds and deploys the site

## How the site updates

The `Update tracker` workflow runs daily at 11:00 UTC, on pushes to `main` that touch the
model, site or scripts, and from the Actions tab ("Run workflow"). It tests and runs the model,
copies `model/out/summary.json` into the site, appends today's row to `history.json` (skipped
while the sample ratings are in use), commits that file back, builds the site, and deploys it
to Cloudflare Pages.

To update after a game, edit `model/data/michigan_schedule.csv` (result column, e.g. `W 84-68`)
and push; the workflow does the rest.

## One-time setup

Repo settings, Secrets and variables, Actions:

| Name | Kind | Value |
| --- | --- | --- |
| `RATINGS_CSV` | Secret | Full contents of your hand-entered ratings CSV (`rank,team,conf,netrtg,adjt,as_of`). Without it the model uses `ratings.sample.csv` and the site shows an "illustrative" banner. Kept as a secret so KenPom's numbers never sit in the public repo. |
| `CLOUDFLARE_API_TOKEN` | Secret | Cloudflare API token with the "Cloudflare Pages: Edit" permission |
| `CLOUDFLARE_ACCOUNT_ID` | Variable | Your Cloudflare account ID (dashboard sidebar) |

Then create a Pages project named `boynton-tracker` in Cloudflare (Workers & Pages, Create,
Pages, "Upload assets"; upload any placeholder once). Until the Cloudflare values exist, the
workflow still builds the site but skips the deploy.

## Local preview

```
cd model && python -m boynton --json out/summary.json && cd ..
cp model/out/summary.json site/public/data/summary.json
cd site && npm install && npm run dev
```
