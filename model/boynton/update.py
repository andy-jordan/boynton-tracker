"""Bayesian (normal-normal / Kalman) rating updates from game margins.

Each team's true rating ~ Normal(NetRtg, sd^2), in points per 100 possessions.
A game's margin ~ Normal((R_home - R_away) * poss/100 + HCA, game_sd^2). Seeing a
margin updates the two ratings in closed form:

    gap prior variance  V = (sd_h^2 + sd_a^2) * (poss/100)^2      (points^2)
    gain                K = V / (V + game_sd^2)
    gap shift (points)  K * clip(observed - expected, +-blowout_cap)

The shift is split between the teams in proportion to their variances and
converted back to NetRtg. Variances shrink the same way.

Don't double count: the KenPom table already includes every game through its
as_of date. Only use this for games after that date (apply_results does that),
and for hypothetical results. A fresh KenPom table replaces these posteriors.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .data import Game, Ratings


@dataclass
class Update:
    home_shift: float   # NetRtg change for the home (first-listed) team
    away_shift: float
    home_sd: float      # posterior sd, NetRtg
    away_sd: float
    gain: float
    surprise: float     # observed - expected margin, points (after the cap)


def kalman_update(net_h: float, net_a: float, sd_h: float, sd_a: float,
                  tempo_h: float, tempo_a: float, neutral: bool, observed: float,
                  hca: float = 3.2, game_sd: float = 11.0, cap: float = 25.0,
                  expected: float | None = None) -> Update:
    """Update two ratings from one observed home margin.

    `expected` overrides the rating-based expectation (e.g. a betting line).
    """
    c = (tempo_h + tempo_a) / 200            # points of margin per NetRtg point
    if expected is None:
        expected = (net_h - net_a) * c + (0 if neutral else hca)
    surprise = max(-cap, min(cap, observed - expected))
    vh, va = (sd_h * c) ** 2, (sd_a * c) ** 2  # in points^2
    V = vh + va
    if V == 0:
        return Update(0.0, 0.0, sd_h, sd_a, 0.0, surprise)
    K = V / (V + game_sd ** 2)
    gap = K * surprise
    # Split by variance share; convert points back to NetRtg.
    h_pts, a_pts = gap * vh / V, -gap * va / V
    post_vh = vh - vh ** 2 / (V + game_sd ** 2)
    post_va = va - va ** 2 / (V + game_sd ** 2)
    return Update(h_pts / c, a_pts / c, post_vh ** 0.5 / c, post_va ** 0.5 / c, K, surprise)


def _as_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s.strip())
    except (ValueError, AttributeError):
        return None


def apply_results(ratings: Ratings, played: list[Game], cfg) -> tuple[Ratings, dict[str, float]]:
    """Apply games dated after ratings.as_of. Returns (new ratings, per-team sd)."""
    sd = {t: cfg.rating_sd for t in ratings.net}
    cutoff = _as_date(ratings.as_of)
    if cfg.rating_sd <= 0 or cutoff is None:
        return ratings, sd
    net = dict(ratings.net)
    for g in sorted(played, key=lambda g: g.date):
        d = _as_date(g.date)
        if d is None or d <= cutoff:
            continue
        nh, th = ratings.get(g.home)
        na, ta = ratings.get(g.away)
        nh, na = net.get(g.home, nh), net.get(g.away, na)
        u = kalman_update(nh, na, sd.get(g.home, cfg.rating_sd), sd.get(g.away, cfg.rating_sd),
                          th, ta, g.neutral, g.home_score - g.away_score,
                          hca=cfg.hca, game_sd=cfg.game_sd, cap=cfg.blowout_cap)
        net[g.home], net[g.away] = nh + u.home_shift, na + u.away_shift
        sd[g.home], sd[g.away] = u.home_sd, u.away_sd
    out = Ratings(net=net, tempo=ratings.tempo, as_of=ratings.as_of, missing=ratings.missing)
    return out, sd
