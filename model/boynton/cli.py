"""python -m boynton [--source kenpom|torvik|both] [--sims N] [--rating-sd 3] [--force 2026-11-06=W] [--json out.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import SOURCES, load_ratings, source_path
from .sim import Config, Model, summarize


def parse_force(items):
    out = {}
    for it in items or []:
        date, val = it.split("=")
        val = val.strip().upper()
        out[date] = val if val in ("W", "L") else float(val)
    return out


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
    ap.add_argument("--source", choices=[*SOURCES, "both"], default="both",
                    help="ratings source (default: run both; JSON's top level is kenpom)")
    ap.add_argument("--ratings", type=Path, help="explicit ratings CSV or raw KenPom paste (.tsv); overrides --source")
    ap.add_argument("--hca", type=float, default=3.2)
    ap.add_argument("--game-sd", type=float, default=11.0)
    ap.add_argument("--rating-sd", type=float, default=0.0,
                    help="season-long rating uncertainty per team (0 = point estimate)")
    ap.add_argument("--uncertainty", choices=["constant", "drift"], default="constant",
                    help="constant: one season-long shift of --rating-sd; drift: random walk calibrated to historical KenPom moves")
    ap.add_argument("--force", action="append",
                    help="scenario for a Michigan game: DATE=W, DATE=L or DATE=+10 (Michigan margin)")
    ap.add_argument("--json", type=Path, help="write summary JSON (probabilities only)")
    a = ap.parse_args(argv)
    cfg = Config(n_sims=a.sims, seed=a.seed, hca=a.hca, game_sd=a.game_sd, rating_sd=a.rating_sd,
                 uncertainty=a.uncertainty)
    forced = parse_force(a.force)

    def run(ratings, label):
        model = Model(cfg, ratings=ratings)
        s = summarize(model, model.simulate(forced))
        s["source"] = label
        return s

    if a.ratings:
        results = {"custom": run(load_ratings(a.ratings), "custom")}
    else:
        names = list(SOURCES) if a.source == "both" else [a.source]
        results = {}
        for name in names:
            path, real = source_path(name)
            s = run(load_ratings(path), name)
            s["sample_data"] = not real
            results[name] = s
    for name, s in results.items():
        print(f"===== {name.upper()}{' (SAMPLE DATA)' if s.get('sample_data') else ''} =====")
        print(report(s), end="\n\n")
    if a.json:
        # Top level stays the primary source's summary (backward compatible);
        # every source's summary is under by_source.
        primary = results.get("kenpom") or next(iter(results.values()))
        out = {**primary, "by_source": results}
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
