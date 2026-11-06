from math import erf, sqrt

from boynton import Config, Model, summarize
from boynton.data import BIG_TEN, load_michigan_schedule, load_ratings
from boynton.bigten import load_big_ten_schedule


def small(**kw):
    return Model(Config(n_sims=2000, **kw))


def test_schedule_and_ratings_load():
    games = load_michigan_schedule()
    assert len(games) == 30 and sum(g.conference for g in games) == 20
    assert games[0].played and games[0].home_score == 84
    r = load_ratings()
    assert all(t in r.net for t in BIG_TEN)


def test_big_ten_schedule_is_20_games_each():
    sched = load_big_ten_schedule(load_michigan_schedule())
    assert len(sched) == 180


def test_uconn_win_prob_matches_formula():
    m = small()
    g = next(g for g in m.mich_games if g.date == "2026-11-06")
    h, a = m.idx[g.home], m.idx[g.away]
    mu = (m.base_net[h] - m.base_net[a]) * (m.tempo[h] + m.tempo[a]) / 200
    assert abs(m.point_win_prob(g) - 0.5 * (1 + erf(mu / 11 / sqrt(2)))) < 1e-9


def test_probabilities_sane_and_forcing_moves_them():
    m = small()
    base = summarize(m, m.simulate())
    assert 0 <= base["p_extension"] <= 1
    assert abs(sum(base["b10_finish_dist"].values()) - 1) < 1e-9
    win = summarize(m, m.simulate({"2026-11-06": "W"}))
    loss = summarize(m, m.simulate({"2026-11-06": "L"}))
    assert win["reg_wins_mean"] > loss["reg_wins_mean"]
    assert win["p_extension"] >= loss["p_extension"]


def test_record_counts_played_games():
    m = small()
    assert summarize(m, m.simulate())["record"] == {"wins": 1, "losses": 0, "conf_wins": 0, "conf_losses": 0}


# ---- uncertainty layer ------------------------------------------------------
from boynton.data import Ratings  # noqa: E402
from boynton.update import apply_results, kalman_update  # noqa: E402
from boynton.whatif import what_if  # noqa: E402


def test_kalman_uconn_example():
    # Equal ratings uncertainty (6 NetRtg), 68 possessions, expected -7.5, won by 10.
    u = kalman_update(0, 0, 6, 6, 68, 68, True, observed=10, expected=-7.5)
    assert abs(u.surprise - 17.5) < 1e-9
    assert abs(u.home_shift * 0.68 - 1.89) < 0.02      # ~1.9 points per game
    assert abs(u.home_shift + u.away_shift) < 1e-9     # zero-sum with equal sd
    assert 5.6 < u.home_sd < 5.8                       # sd shrinks a little


def test_kalman_caps_blowouts_and_ignores_zero_sd():
    big = kalman_update(0, 0, 6, 6, 68, 68, True, observed=60, expected=0)
    assert big.surprise == 25
    none = kalman_update(0, 0, 0, 0, 68, 68, True, observed=30, expected=0)
    assert none.home_shift == 0 and none.away_shift == 0


def test_apply_results_only_after_as_of():
    games = load_michigan_schedule()          # Oakland game played 2026-11-02
    r = load_ratings()
    cfg = Config(rating_sd=6)
    before = Ratings(r.net, r.tempo, as_of="2026-11-05")
    after = Ratings(r.net, r.tempo, as_of="2026-11-01")
    same, _ = apply_results(before, [g for g in games if g.played], cfg)
    moved, sd = apply_results(after, [g for g in games if g.played], cfg)
    assert same.net["Michigan"] == r.net["Michigan"]
    assert moved.net["Michigan"] != r.net["Michigan"] and sd["Michigan"] < 6


def test_whatif_no_information_without_rating_uncertainty():
    m = small()
    w = what_if(m, "2026-11-06", "W")
    assert abs(w.information) < 1e-12 and w.direct >= w.baseline


def test_whatif_decomposition_is_consistent():
    m = Model(Config(n_sims=20000, rating_sd=6))
    base = m.simulate()
    win, loss = what_if(m, "2026-11-06", "W", base), what_if(m, "2026-11-06", "L", base)
    big, small_win = what_if(m, "2026-11-06", 10.0, base), what_if(m, "2026-11-06", 1.0, base)
    assert win.information > 0 > loss.information
    assert big.total > small_win.total
    # Law of total probability: P(W) P(ext|W) + P(L) P(ext|L) = P(ext).
    p = base["mich_results"]["2026-11-06"].mean()
    assert abs(p * win.total + (1 - p) * loss.total - win.baseline) < 0.015
