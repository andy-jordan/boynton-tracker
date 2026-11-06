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


def test_cli_both_sources_json(tmp_path):
    import json
    from boynton.cli import main
    out = tmp_path / "s.json"
    main(["--sims", "500", "--json", str(out)])
    d = json.loads(out.read_text())
    assert set(d["by_source"]) == {"kenpom", "torvik"}
    assert d["source"] == "kenpom" and 0 <= d["p_extension"] <= 1
    assert all(s["source"] == k for k, s in d["by_source"].items())
