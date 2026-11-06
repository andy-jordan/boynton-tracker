"""Big Ten conference schedule (20 games per team, 18 teams = 180 games).

If data/big_ten_schedule.csv exists it is used as-is (columns: date,home,away,
home_score,away_score). Otherwise we build a synthetic schedule: Michigan's real
20 league games, plus a random-but-valid slate for everyone else (each pair plays
at least once, extra games fill each team out to 20). The synthetic file is
written to data/big_ten_schedule.synthetic.csv so it can be inspected.

TODO: replace with the official Big Ten schedule; it mainly matters for the
top-4 and BTT-seeding benchmarks.
"""
from __future__ import annotations

import csv
import itertools
import random
from collections import Counter
from pathlib import Path

from .data import BIG_TEN, DATA_DIR, Game, canon

GAMES_PER_TEAM = 20


def load_big_ten_schedule(michigan_games: list[Game], seed: int = 2027) -> list[Game]:
    real = DATA_DIR / "big_ten_schedule.csv"
    if real.exists():
        out = []
        for r in csv.DictReader(real.read_text().splitlines()):
            g = Game(r["date"], canon(r["home"]), canon(r["away"]), False, True)
            if r.get("home_score"):
                g.home_score, g.away_score = int(r["home_score"]), int(r["away_score"])
            out.append(g)
        # Michigan's results are entered in michigan_schedule.csv; carry them over.
        mich = {(g.date, g.home, g.away): g for g in michigan_games if g.conference}
        for g in out:
            m = mich.get((g.date, g.home, g.away))
            if m is not None and m.played and not g.played:
                g.home_score, g.away_score = m.home_score, m.away_score
        return out
    return _synthetic(michigan_games, seed)


def _synthetic(michigan_games: list[Game], seed: int) -> list[Game]:
    rng = random.Random(seed)
    mich_conf = [g for g in michigan_games if g.conference]
    others = [t for t in BIG_TEN if t != "Michigan"]
    vs_mich = Counter(g.away if g.home == "Michigan" else g.home for g in mich_conf)
    # Each non-Michigan pair plays once; then extra games to reach 20.
    need = {t: GAMES_PER_TEAM - vs_mich[t] - (len(others) - 1) for t in others}
    for _ in range(10_000):
        extra, left = [], dict(need)
        pool = [t for t in others for _ in range(left[t])]
        rng.shuffle(pool)
        ok = True
        while pool:
            a = pool.pop()
            cands = [i for i, b in enumerate(pool) if b != a and {a, b} not in [set(e) for e in extra]]
            if not cands:
                ok = False
                break
            b = pool.pop(rng.choice(cands))
            extra.append((a, b))
        if ok:
            break
    else:  # pragma: no cover
        raise RuntimeError("could not build synthetic schedule")

    games = list(mich_conf)
    extra_set = {frozenset(e) for e in extra}
    for a, b in itertools.combinations(others, 2):
        h, w = (a, b) if rng.random() < 0.5 else (b, a)
        games.append(Game("", h, w, False, True))
        if frozenset((a, b)) in extra_set:
            games.append(Game("", w, h, False, True))  # return game
    out = DATA_DIR / "big_ten_schedule.synthetic.csv"
    with out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["date", "home", "away", "home_score", "away_score"])
        for g in games:
            wr.writerow([g.date, g.home, g.away, g.home_score or "", g.away_score or ""])
    counts = Counter(t for g in games for t in (g.home, g.away))
    assert all(counts[t] == GAMES_PER_TEAM for t in BIG_TEN), counts
    return games
