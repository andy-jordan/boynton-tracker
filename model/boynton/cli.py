"""python -m boynton [--sims N] [--rating-sd 3] [--force 2026-11-06=W] [--json out.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_ratings
from .sim import Config, Model, summarize
from .whatif import next_games, what_if


def parse_pairs(items) -> list[tuple[str, object]]:
    out = []
    for it in items or []:
        date, val = it.split("=")
        val = val.strip().upper()
        out.append((date.strip(), val if val in ("W", "L") else float(val)))
    return out


def parse_force(items) -> dict:
    return dict(parse_pairs(items))


def report(s: dict) -> str:
    b = s["benchmarks"]
    lines = [
        f"Ratings as of {s['as_of']}  |  {s['n_sims']:,} sims",
        f"P(extension: 2+ of 4)     {s['p_extension']:6.1%}",
        f"  24+ regular-season wins {b['24_wins']:6.1%}   (mean {s['reg_wins_mean']:.1f} wins)",
        f"  Top-4 Big Ten finish    {b['top4']:6.1%}",
        f"  Win Big Ten Tournament  {b['btt_title']:6.1%}",
        f"  Sweet 16                {b['sweet16']:6.1%}   (make NCAA {s['p_make_ncaa']:.1%})",
        "Benchmarks hit: " + "  ".join(f"{k}:{v:.1%}" for k, v in s["hits_dist"].items()),
        "",
        "Remaining games: P(win)  P(ext | W)  P(ext | L)",
    ]
    for g in s["leverage"]:
        fmt = lambda x: "   n/a" if x is None else f"{x:6.1%}"
        lines.append(f"  {g['date']} {g['site']} {g['opponent']:<24} {g['p_win']:6.1%}"
                     f"  {fmt(g['p_ext_if_win'])}  {fmt(g['p_ext_if_loss'])}")
    if s["missing_ratings"]:
        lines.append(f"\nFallback rating used for: {', '.join(s['missing_ratings'])}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="boynton")
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ratings", type=Path, help="ratings CSV or raw KenPom paste (.tsv)")
    ap.add_argument("--hca", type=float, default=3.2)
    ap.add_argument("--game-sd", type=float, default=11.0)
    ap.add_argument("--rating-sd", type=float, default=0.0,
                    help="season-long rating uncertainty per team (0 = point estimate)")
    ap.add_argument("--force", action="append",
                    help="scenario for a Michigan game: DATE=W, DATE=L or DATE=+10 (Michigan margin)")
    ap.add_argument("--whatif", action="append",
                    help="split a result's effect into direct + information: DATE=W, DATE=L or DATE=+10")
    ap.add_argument("--decompose", type=int, default=0, metavar="K",
                    help="direct/information split for W and L in the next K games")
    ap.add_argument("--json", type=Path, help="write summary JSON (probabilities only)")
    a = ap.parse_args(argv)
    cfg = Config(n_sims=a.sims, seed=a.seed, hca=a.hca, game_sd=a.game_sd, rating_sd=a.rating_sd)
    model = Model(cfg, ratings=load_ratings(a.ratings) if a.ratings else None)
    forced = parse_force(a.force)
    res = model.simulate(forced)
    s = summarize(model, res)
    print(report(s))
    asks = parse_pairs(a.whatif)
    for d in next_games(model, a.decompose):
        asks += [(d, "W"), (d, "L")]
    if asks:
        if forced:
            ap.error("--whatif/--decompose can't be combined with --force")
        if cfg.rating_sd == 0:
            print("\nNote: --rating-sd is 0, so results carry no information about ratings.")
        print("\nWhat-if: P(ext) baseline -> direct (result banked) -> total (with what it says about ratings)")
        rows = [what_if(model, d, o, res) for d, o in asks]
        for r in rows:
            print(f"  {r.date} {r.outcome:>4}  {r.baseline:6.1%} -> {r.direct:6.1%} -> {r.total:6.1%}"
                  f"   direct {r.direct - r.baseline:+6.1%}  info {r.information:+6.1%}"
                  f"   ({r.effective_sims:,.0f} sims)")
        s["what_if"] = [r.as_dict() for r in rows]
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
