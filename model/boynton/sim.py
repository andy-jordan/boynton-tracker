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
    resume_per_win: float = 1.0   # NCAA seeding: resume points per win above expectation
    at_large_cutoff: int = 44     # resume rank needed for an at-large bid
    # Typical NetRtg of seeds 12-16 (auto-bid teams), used for NCAA opponents.
    low_seed_net: dict = field(default_factory=lambda: {12: 9.0, 13: 5.0, 14: 2.0, 15: -2.0, 16: -8.0})
    wins_target: int = 24
    # Rating uncertainty mode: "constant" = one season-long N(0, rating_sd) shift
    # per team per sim (rating_sd=0 means none); "drift" = random walk calibrated
    # to how far KenPom ratings historically move between now and Selection Sunday.
    uncertainty: str = "constant"
    top4_counts_ties: bool = True  # a tie for 4th counts (Torvik convention; Andy 2026-11-07)


# --------------------------------------------------------------------------
# Hooks for the uncertainty / Bayesian-updating work (separate thread).
# --------------------------------------------------------------------------

def update_ratings(ratings: Ratings, played: list[Game]) -> Ratings:
    """Hook: return ratings updated for games already played.

    Point-estimate mode returns the input unchanged, because the KenPom table you
    enter already reflects games through its as_of date. A Bayesian version would
    treat each rating as a prior (mean=NetRtg, sd~rating_sd) and update on the
    observed margin vs. expectation.
    """
    return ratings


def draw_ratings(base: np.ndarray, n: int, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Hook: per-simulation team ratings, shape (n, n_teams).

    Default: point ratings, optionally plus one season-long N(0, rating_sd) shift
    per team per sim. Setting rating_sd > 0 makes early results informative: a sim
    where Michigan beats UConn is more likely to be a sim where Michigan is good,
    so the conditional tables pick up the "information" effect, not just the win.
    """
    out = np.broadcast_to(base, (n, base.size)).copy()
    if cfg.rating_sd > 0:
        out += rng.normal(0, cfg.rating_sd, out.shape)
    return out


# Historical KenPom (2012-2026, top-100 teams): SD of (rating on date) minus
# (rating on Selection Sunday), in NetRtg points. Mapped from POM ranks via the
# current rank->NetRtg curve; see the uncertainty thread / PR #3.
DRIFT_MISS_SD = [("11-07", 5.9), ("12-05", 5.1), ("12-25", 4.3), ("01-15", 3.6),
                 ("02-05", 2.8), ("02-25", 1.9), ("03-08", 1.1), ("03-15", 0.0)]


def _season_key(date: str) -> str:
    """Sortable within-season key: Nov-Dec before Jan-Apr."""
    mmdd = date[5:]
    return ("0" if mmdd >= "07-01" else "1") + mmdd


def drift_paths(base: np.ndarray, n: int, rng: np.random.Generator, as_of: str):
    """Random-walk ratings. Returns (knot_keys, deltas list, final_delta).

    The walk starts at 0 on the ratings date and has variance s(now)^2 - s(t)^2
    at time t, so by Selection Sunday each team's total move has the historical
    SD s(now). Increments are independent across teams and knots.
    """
    now = _season_key(as_of[:10]) if as_of[:4].isdigit() else _season_key("2026-11-07")
    knots = [(_season_key("2026-" + d), sd) for d, sd in DRIFT_MISS_SD]
    # miss SD as of the ratings date (interpolate on the table)
    def miss_at(k):
        prev = knots[0]
        for kk, sd in knots:
            if kk >= k:
                if kk == prev[0]:
                    return sd
                return prev[1] + (sd - prev[1]) * (_daynum(k) - _daynum(prev[0])) / (_daynum(kk) - _daynum(prev[0]))
            prev = (kk, sd)
        return 0.0
    s_now = miss_at(now)
    keys, deltas, d, s_prev = [now], [np.zeros((n, base.size))], np.zeros((n, base.size)), s_now
    for k, sd in knots:
        if k <= now:
            continue
        sd = min(sd, s_now)
        d = d + rng.normal(0, sqrt(max(s_prev ** 2 - sd ** 2, 0.0)), d.shape)
        keys.append(k); deltas.append(d); s_prev = sd
    return keys, deltas, d


def _daynum(key: str) -> int:
    m, dd = int(key[1:3]), int(key[4:6])
    cum = {11: 0, 12: 30, 1: 61, 2: 92, 3: 120, 4: 151, 7: -122, 8: -91, 9: -60, 10: -30}
    return cum[m] + dd


# --------------------------------------------------------------------------

def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


_vphi = np.vectorize(_phi, otypes=[float])


class Model:
    def __init__(self, cfg: Config | None = None, ratings: Ratings | None = None,
                 schedule: list[Game] | None = None):
        self.cfg = cfg or Config()
        self.ratings = ratings or load_ratings()
        self.mich_games = schedule or load_michigan_schedule()
        self.ratings = update_ratings(self.ratings, [g for g in self.mich_games if g.played])
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
        self.b10_idx = np.array([self.idx[t] for t in BIG_TEN])

    # ---- single-game pieces -------------------------------------------------
    def mean_margin(self, net: np.ndarray, h: int, a: int, neutral: bool) -> np.ndarray:
        poss = (self.tempo[h] + self.tempo[a]) / 2
        return (net[..., h] - net[..., a]) * poss / 100 + (0 if neutral else self.cfg.hca)

    def point_win_prob(self, g: Game) -> float:
        """Pre-game P(home wins) from point ratings (no rating uncertainty)."""
        h, a = self.idx[g.home], self.idx[g.away]
        mu = float(self.mean_margin(self.base_net, h, a, g.neutral))
        sd = sqrt(self.cfg.game_sd ** 2 + 2 * self.cfg.rating_sd ** 2)
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
        if cfg.uncertainty == "drift":
            # Walk only the teams that play regular-season games we simulate;
            # everyone gets a full-season move for the postseason.
            sub = sorted({self.idx[t] for g in self.mich_games + self.b10_games for t in (g.home, g.away)})
            keys, deltas, final_sub = drift_paths(self.base_net[sub], n, rng, self.ratings.as_of)
            s_now = float(np.std(final_sub)) if final_sub.size else 0.0
            net = self.base_net + rng.normal(0, s_now, (n, self.base_net.size))
            net[:, sub] = self.base_net[sub] + final_sub   # postseason (BTT, NCAA) ratings
            col = {t: j for j, t in enumerate(sub)}
            base = self.base_net

            class _KnotNet:
                def __init__(self, d): self.d = d
                def __getitem__(self, key):            # supports net[..., team]
                    t = key[-1]
                    return base[t] + self.d[:, col[t]]

            def net_for(date: str):
                k = _season_key(date) if date else keys[-1]
                i = max([j for j, kk in enumerate(keys) if kk <= k] or [0])
                return _KnotNet(deltas[i])
        else:
            net = draw_ratings(self.base_net, n, rng, cfg)
            net_for = lambda date: net

        # Regular season: Michigan non-conference + full Big Ten slate.
        mich_results = {}  # date -> Michigan-perspective win bool array
        exp_wins = np.zeros(n)  # Michigan's expected wins given each sim's ratings
        for g in self.mich_games:
            h, a = self.idx[g.home], self.idx[g.away]
            p_home = _vphi(self.mean_margin(net_for(g.date), h, a, g.neutral) / cfg.game_sd)
            exp_wins += p_home if g.home == MICH else 1 - p_home
        for g in self.nonconf:
            m = self._play(net_for(g.date), g, rng, n, forced)
            mich_results[g.date] = (m > 0) if g.home == MICH else (m < 0)
        T = len(BIG_TEN)
        pos = {t: i for i, t in enumerate(BIG_TEN)}
        conf_w = np.zeros((n, T))
        conf_pd = np.zeros((n, T))
        for g in self.b10_games:
            m = self._play(net_for(g.date), g, rng, n, forced)
            hi, ai = pos[g.home], pos[g.away]
            conf_w[:, hi] += m > 0
            conf_w[:, ai] += m < 0
            conf_pd[:, hi] += m
            conf_pd[:, ai] -= m
            if MICH in (g.home, g.away) and g.date:
                mich_results[g.date] = (m > 0) if g.home == MICH else (m < 0)
        reg_wins = sum(mich_results.values()).astype(int)

        # Standings. Tiebreak approximation: conference point differential, then random.
        # TODO: real Big Ten tiebreakers (head-to-head, record vs. top teams...).
        score = conf_w + 1e-4 * conf_pd + 1e-7 * rng.random((n, T))
        order = np.argsort(-score, axis=1)                  # order[:, k] = team pos at seed k+1
        mi = pos[MICH]
        mich_finish = np.argmax(order == mi, axis=1) + 1
        # Top 4 counting ties (Torvik's convention): conference wins at least
        # the 4th-best total. Kept for the cross-check against Torvik's page.
        fourth = -np.sort(-conf_w, axis=1)[:, 3:4]
        top4_ties = conf_w >= fourth                         # (n, T)
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
        # "Luck": wins above what this sim's ratings predicted. BTT wins are
        # measured against their across-sim average.
        luck = (reg_wins - exp_wins) + (btt_wins - btt_wins.mean())
        made_ncaa, seed, s16 = self._ncaa(net, luck, won_btt, rng)

        b = {
            "24_wins": reg_wins >= cfg.wins_target,
            # Andy (2026-11-07): a tie for 4th counts, as on Torvik's page.
            "top4": top4_ties[:, mi] if cfg.top4_counts_ties else mich_finish <= 4,
            "btt_title": won_btt,
            "sweet16": s16,
        }
        hits = sum(v.astype(int) for v in b.values())
        return {
            "n": n, "benchmarks": b, "hits": hits, "extension": hits >= 2,
            "reg_wins": reg_wins, "finish": mich_finish, "made_ncaa": made_ncaa,
            "ncaa_seed": seed, "mich_results": mich_results,
            "b10_top4_ties": top4_ties.mean(axis=0),
            "b10_top4": (np.argsort(order, axis=1) < 4).mean(axis=0),
            "b10_btt": np.array([(champ == self.idx[t]).mean() for t in BIG_TEN]),
        }

    def _ncaa(self, net, luck, won_btt, rng):
        cfg, n = self.cfg, self.cfg.n_sims
        mich_i = self.idx[MICH]
        # End-of-season resume = rating in that sim + credit for wins above what
        # that rating predicts. Michigan's luck is simulated; other teams get a
        # random luck term with the same spread, so nobody is treated differently.
        luck_sd = cfg.resume_per_win * float(np.std(luck))
        resume = net + rng.normal(0, luck_sd, net.shape)
        resume[:, mich_i] = net[:, mich_i] + cfg.resume_per_win * luck
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
    # League-wide table for cross-checking against Torvik's conference page.
    out["b10_table"] = {
        t: {"top4": float(res["b10_top4"][i]), "top4_ties": float(res["b10_top4_ties"][i]),
            "btt": float(res["b10_btt"][i])}
        for i, t in enumerate(BIG_TEN)
    }
    out["benchmarks_extra"] = {
        "top4_ties": out["b10_table"][MICH]["top4_ties"],
        "top4_seed": out["b10_table"][MICH]["top4"],
    }
    return out
