"""Monte Carlo season simulator for the Boynton extension benchmarks.

Benchmarks (needs 2 of 4):
  1. 24+ regular-season wins (all 30 scheduled games; excludes BTT/NCAA)
  2. Top-4 finish in Big Ten regular-season standings
  3. Win the Big Ten Tournament
  4. Reach the NCAA Sweet 16

Game model: Michigan-perspective margin ~ Normal(mu, sd), with
  mu = (net_home - net_away) * possessions / 100 + HCA (0 if neutral)
  possessions = (adjT_home + adjT_away) / 2
which is the standard KenPom-style prediction from NetRtg and AdjT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np

from .bigten import load_big_ten_schedule
from .data import BIG_TEN, Game, Ratings, load_michigan_schedule, load_ratings

MICH = "Michigan"


@dataclass
class Config:
    n_sims: int = 20_000
    seed: int = 1
    hca: float = 3.2          # home-court advantage, points
    game_sd: float = 11.0     # SD of actual minus predicted margin
    rating_sd: float = 0.0    # per-sim, season-long rating uncertainty (points/100 poss)
    blowout_cap: float = 25.0 # clip |observed - expected| margin before a rating update
    resume_per_win: float = 1.0   # NCAA seeding: resume points per win above expectation
    resume_noise: float = 3.0     # noise on other teams' end-of-year resume
    at_large_cutoff: int = 44     # resume rank needed for an at-large bid
    # Typical NetRtg of seeds 12-16 (auto-bid teams), used for NCAA opponents.
    low_seed_net: dict = field(default_factory=lambda: {12: 9.0, 13: 5.0, 14: 2.0, 15: -2.0, 16: -8.0})
    wins_target: int = 24


# --------------------------------------------------------------------------
# Hooks for the uncertainty / Bayesian-updating work (separate thread).
# --------------------------------------------------------------------------

def update_ratings(ratings: Ratings, played: list[Game], cfg: "Config | None" = None
                   ) -> tuple[Ratings, dict[str, float]]:
    """Hook: return (ratings, per-team sd) updated for games played after `as_of`.

    The KenPom table already reflects games through its as_of date, so only
    games dated after it are applied (none if as_of isn't a date). Each such
    game is a normal-normal (Kalman) update on the rating gap; see update.py.
    With rating_sd = 0 there is no uncertainty to update, so it's a no-op.
    """
    from .update import apply_results
    cfg = cfg or Config()
    return apply_results(ratings, played, cfg)


def draw_ratings(base: np.ndarray, n: int, rng: np.random.Generator, cfg: Config,
                 sd: np.ndarray | None = None) -> np.ndarray:
    """Hook: per-simulation team ratings, shape (n, n_teams).

    Point ratings plus one season-long N(0, sd) shift per team per sim (sd =
    rating_sd for every team unless a per-team array is passed). With sd > 0,
    a sim where Michigan beats UConn is more likely to be a sim where Michigan is
    good, so conditioning on a result picks up the "information" effect, not just
    the win.
    """
    out = np.broadcast_to(base, (n, base.size)).copy()
    if sd is None:
        sd = np.full(base.size, cfg.rating_sd)
    if np.any(sd > 0):
        out += rng.normal(0, 1, out.shape) * sd
    return out


# --------------------------------------------------------------------------

def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


class Model:
    def __init__(self, cfg: Config | None = None, ratings: Ratings | None = None,
                 schedule: list[Game] | None = None):
        self.cfg = cfg or Config()
        self.ratings = ratings or load_ratings()
        self.mich_games = schedule or load_michigan_schedule()
        self.ratings, sd_map = update_ratings(self.ratings, [g for g in self.mich_games if g.played], self.cfg)
        self.b10_games = load_big_ten_schedule(self.mich_games)
        self.nonconf = [g for g in self.mich_games if not g.conference]
        # Team universe: every rated team plus anyone on a schedule.
        teams = list(self.ratings.ranked())
        for g in self.mich_games + self.b10_games:
            for t in (g.home, g.away):
                if t not in teams:
                    teams.append(t)
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        pairs = [self.ratings.get(t) for t in teams]
        self.base_net = np.array([p[0] for p in pairs])
        self.tempo = np.array([p[1] for p in pairs])
        self.base_sd = np.array([sd_map.get(t, self.cfg.rating_sd) for t in teams])
        self.b10_idx = np.array([self.idx[t] for t in BIG_TEN])

    # ---- single-game pieces -------------------------------------------------
    def mean_margin(self, net: np.ndarray, h: int, a: int, neutral: bool) -> np.ndarray:
        poss = (self.tempo[h] + self.tempo[a]) / 2
        return (net[..., h] - net[..., a]) * poss / 100 + (0 if neutral else self.cfg.hca)

    def point_win_prob(self, g: Game) -> float:
        """Pre-game P(home wins) from point ratings (no rating uncertainty)."""
        h, a = self.idx[g.home], self.idx[g.away]
        mu = float(self.mean_margin(self.base_net, h, a, g.neutral))
        poss = (self.tempo[h] + self.tempo[a]) / 2
        rating_var = (self.base_sd[h] ** 2 + self.base_sd[a] ** 2) * (poss / 100) ** 2
        sd = sqrt(self.cfg.game_sd ** 2 + rating_var)
        return _phi(mu / sd)

    def _play(self, net, g: Game, rng, n, forced: dict) -> np.ndarray:
        """Home margin per sim for a scheduled game."""
        if g.played:
            return np.full(n, float(g.home_score - g.away_score))
        h, a = self.idx[g.home], self.idx[g.away]
        m = self.mean_margin(net, h, a, g.neutral) + rng.normal(0, self.cfg.game_sd, n)
        key = g.date if MICH in (g.home, g.away) else None
        if key in forced:
            f = forced[key]  # Michigan perspective: "W", "L", or a signed margin
            sign = 1 if g.home == MICH else -1
            if isinstance(f, (int, float)):
                m = np.full(n, float(f) * sign)
            else:
                want_home_win = (f == "W") == (sign == 1)
                m = np.where((m > 0) == want_home_win, m, -m)  # reflect into the forced side
        return m

    def _neutral(self, net, a: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
        """Vectorized neutral-site game between team-index arrays; returns winners."""
        rows = np.arange(len(a))
        poss = (self.tempo[a] + self.tempo[b]) / 2
        mu = (net[rows, a] - net[rows, b]) * poss / 100
        m = mu + rng.normal(0, self.cfg.game_sd, len(a))
        return np.where(m > 0, a, b)

    # ---- full season -------------------------------------------------------
    def simulate(self, forced: dict | None = None) -> dict:
        cfg, forced = self.cfg, forced or {}
        n = cfg.n_sims
        rng = np.random.default_rng(cfg.seed)
        net = draw_ratings(self.base_net, n, rng, cfg, self.base_sd)

        # Regular season: Michigan non-conference + full Big Ten slate.
        mich_results = {}  # date -> Michigan-perspective win bool array
        mich_mu = {}       # date -> Michigan-perspective expected margin per sim (pre-noise)
        mich_margin = {}   # date -> Michigan-perspective simulated margin per sim

        def record(g, m):
            sign = 1 if g.home == MICH else -1
            h, a = self.idx[g.home], self.idx[g.away]
            mich_mu[g.date] = sign * self.mean_margin(net, h, a, g.neutral)
            mich_margin[g.date] = sign * m
            mich_results[g.date] = sign * m > 0

        for g in self.nonconf:
            record(g, self._play(net, g, rng, n, forced))
        T = len(BIG_TEN)
        pos = {t: i for i, t in enumerate(BIG_TEN)}
        conf_w = np.zeros((n, T))
        conf_pd = np.zeros((n, T))
        for g in self.b10_games:
            m = self._play(net, g, rng, n, forced)
            hi, ai = pos[g.home], pos[g.away]
            conf_w[:, hi] += m > 0
            conf_w[:, ai] += m < 0
            conf_pd[:, hi] += m
            conf_pd[:, ai] -= m
            if MICH in (g.home, g.away) and g.date:
                record(g, m)
        reg_wins = sum(mich_results.values()).astype(int)

        # Standings. Tiebreak approximation: conference point differential, then random.
        # TODO: real Big Ten tiebreakers (head-to-head, record vs. top teams...).
        score = conf_w + 1e-4 * conf_pd + 1e-7 * rng.random((n, T))
        order = np.argsort(-score, axis=1)                  # order[:, k] = team pos at seed k+1
        mi = pos[MICH]
        mich_finish = np.argmax(order == mi, axis=1) + 1
        seeds = self.b10_idx[order]                          # global team idx by seed

        # Big Ten Tournament (15 teams).
        s = lambda k: seeds[:, k - 1]
        play = lambda a, b: self._neutral(net, a, b, rng)
        w12_13, w10_15, w11_14 = play(s(12), s(13)), play(s(10), s(15)), play(s(11), s(14))
        w8_9, w5, w7, w6 = play(s(8), s(9)), play(s(5), w12_13), play(s(7), w10_15), play(s(6), w11_14)
        q1, q4, q2, q3 = play(s(1), w8_9), play(s(4), w5), play(s(2), w7), play(s(3), w6)
        f1, f2 = play(q1, q4), play(q2, q3)
        champ = play(f1, f2)
        mich_i = self.idx[MICH]
        won_btt = champ == mich_i
        btt_wins = sum((x == mich_i).astype(int) for x in
                       (w12_13, w10_15, w11_14, w8_9, w5, w7, w6, q1, q4, q2, q3, f1, f2, champ))

        # NCAA tournament.
        made_ncaa, seed, s16 = self._ncaa(net, reg_wins + btt_wins, won_btt, rng)

        b = {
            "24_wins": reg_wins >= cfg.wins_target,
            "top4": mich_finish <= 4,
            "btt_title": won_btt,
            "sweet16": s16,
        }
        hits = sum(v.astype(int) for v in b.values())
        return {
            "n": n, "benchmarks": b, "hits": hits, "extension": hits >= 2,
            "reg_wins": reg_wins, "finish": mich_finish, "made_ncaa": made_ncaa,
            "ncaa_seed": seed, "mich_results": mich_results,
            "mich_mu": mich_mu, "mich_margin": mich_margin,
        }

    def _ncaa(self, net, total_wins, won_btt, rng):
        cfg, n = self.cfg, self.cfg.n_sims
        mich_i = self.idx[MICH]
        # Expected total wins (regular season + ~1.5 BTT games) from point ratings.
        exp_w = sum(self.point_win_prob(g) if g.home == MICH else 1 - self.point_win_prob(g)
                    for g in self.mich_games) + 1.5
        # End-of-season resume: everyone else gets noise; Michigan gets its rating
        # (in that sim) plus credit for wins above expectation.
        resume = net + rng.normal(0, cfg.resume_noise, net.shape)
        resume[:, mich_i] = net[:, mich_i] + cfg.resume_per_win * (total_wins - exp_w)
        rank = (resume > resume[:, [mich_i]]).sum(axis=1) + 1
        made = (rank <= cfg.at_large_cutoff) | won_btt
        seed = np.where(made, np.minimum(np.ceil(rank / 4), 12), 0).astype(int)

        # Seed-line opponents: seeds 1-11 drawn from the sim's resume order
        # (excluding Michigan); 12-16 from typical auto-bid strength.
        order = np.argsort(-resume, axis=1)
        order_wo = np.where(order == mich_i, -1, order)
        rows = np.arange(n)

        def team_on_line(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return (net, tempo) of a random team on seed line k (per sim)."""
            out_net = np.empty(n)
            out_t = np.full(n, 70.0)
            hi = k <= 11
            pick = (4 * (k - 1) + rng.integers(0, 4, n)).clip(0, order.shape[1] - 2)
            # skip Michigan's own slot by shifting one if we hit it
            t = order_wo[rows, pick]
            t = np.where(t < 0, order_wo[rows, pick + 1], t)
            out_net[hi] = net[rows[hi], t[hi]]
            out_t[hi] = self.tempo[t[hi]]
            lo = ~hi
            low = np.array([cfg.low_seed_net.get(int(x), -8.0) for x in k[lo]])
            out_net[lo] = low + rng.normal(0, 2.0, lo.sum())
            return out_net, out_t

        def beat(opp_net, opp_t, my_net):
            poss = (self.tempo[mich_i] + opp_t) / 2
            mu = (my_net - opp_net) * poss / 100
            return mu + rng.normal(0, cfg.game_sd, n) > 0

        sd = np.where(made, seed, 1)
        my = net[:, mich_i]
        r64_net, r64_t = team_on_line(17 - sd)
        pod = {1: 8, 16: 9, 8: 1, 9: 16, 4: 5, 13: 12, 5: 4, 12: 13,
               3: 6, 14: 11, 6: 3, 11: 14, 2: 7, 15: 10, 7: 2, 10: 15}
        p = np.array([pod[int(x)] for x in sd])
        a_net, a_t = team_on_line(p)
        b_net, b_t = team_on_line(17 - p)
        poss_ab = (a_t + b_t) / 2
        a_wins = (a_net - b_net) * poss_ab / 100 + rng.normal(0, cfg.game_sd, n) > 0
        r32_net = np.where(a_wins, a_net, b_net)
        r32_t = np.where(a_wins, a_t, b_t)
        s16 = made & beat(r64_net, r64_t, my) & beat(r32_net, r32_t, my)
        return made, seed, s16


# --------------------------------------------------------------------------

def _record(games: list[Game]) -> dict:
    """Michigan's actual record so far (overall and Big Ten)."""
    out = {"wins": 0, "losses": 0, "conf_wins": 0, "conf_losses": 0}
    for g in games:
        if not g.played:
            continue
        won = (g.home_score > g.away_score) == (g.home == MICH)
        out["wins" if won else "losses"] += 1
        if g.conference:
            out["conf_wins" if won else "conf_losses"] += 1
    return out


def summarize(model: Model, res: dict) -> dict:
    b = res["benchmarks"]
    out = {
        "as_of": model.ratings.as_of,
        "n_sims": res["n"],
        "p_extension": float(res["extension"].mean()),
        "benchmarks": {k: float(v.mean()) for k, v in b.items()},
        "hits_dist": {k: float((res["hits"] == k).mean()) for k in range(5)},
        "reg_wins_mean": float(res["reg_wins"].mean()),
        "reg_wins_dist": {int(k): float((res["reg_wins"] == k).mean())
                          for k in range(res["reg_wins"].min(), res["reg_wins"].max() + 1)},
        "b10_finish_dist": {int(k): float((res["finish"] == k).mean()) for k in range(1, 19)},
        "p_make_ncaa": float(res["made_ncaa"].mean()),
        "missing_ratings": sorted(model.ratings.missing),
        "record": _record(model.mich_games),
    }
    # Leverage: how each remaining Michigan game moves P(extension).
    lev = []
    ext = res["extension"]
    for g in model.mich_games:
        if g.played:
            continue
        w = res["mich_results"][g.date]
        opp = g.away if g.home == MICH else g.home
        site = "N" if g.neutral else ("H" if g.home == MICH else "A")
        lev.append({
            "date": g.date, "opponent": opp, "site": site,
            "p_win": float(w.mean()),
            # Conditional estimates need enough sims on each side to mean anything.
            "p_ext_if_win": float(ext[w].mean()) if w.sum() >= 200 else None,
            "p_ext_if_loss": float(ext[~w].mean()) if (~w).sum() >= 200 else None,
        })
    out["leverage"] = lev
    return out
