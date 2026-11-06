"""What-if analysis: how one Michigan result moves P(extension), split in two.

- direct: re-run the same sims (same seed, same rating draws) with the game
  forced to the result. Ratings are not touched, so this is just the win (and
  margin) banked.
- total: the same forced sims, each weighted by how likely the result was given
  that sim's drawn ratings: P(win) = Phi(mu/game_sd) for W, 1 - that for L, and
  the Normal(margin; mu, game_sd) density for a margin. Sims where Michigan's
  drawn rating is high get more weight, so this adds what the result says about
  team quality. It equals P(ext | result) under the model (Bayes' rule), with
  less noise than filtering the baseline sims.
- information = total - direct.

With rating_sd = 0 every sim has the same ratings, so information is ~0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

erf = np.vectorize(__import__("math").erf, otypes=[float])

from .sim import Model


@dataclass
class WhatIf:
    date: str
    outcome: str             # "W", "L" or a signed Michigan margin like "+10"
    baseline: float
    direct: float            # P(ext) with the result banked, ratings at prior
    total: float             # P(ext | result), information included
    effective_sims: float    # sims backing `total` (ESS for margin weights)
    benchmarks_total: dict

    @property
    def information(self) -> float:
        return self.total - self.direct

    def as_dict(self) -> dict:
        d = asdict(self)
        d["information"] = self.information
        return d


def _weights(model: Model, mu: np.ndarray, outcome) -> np.ndarray:
    """Likelihood of the result given each sim's drawn ratings (Michigan view)."""
    z = mu / model.cfg.game_sd
    if outcome == "W":
        return 0.5 * (1 + erf(z / np.sqrt(2)))
    if outcome == "L":
        return 0.5 * (1 - erf(z / np.sqrt(2)))
    z = (float(outcome) - mu) / model.cfg.game_sd
    return np.exp(-0.5 * z * z)


def what_if(model: Model, date: str, outcome, base: dict | None = None) -> WhatIf:
    """outcome: "W", "L", or a float Michigan margin."""
    if date not in {g.date for g in model.mich_games if not g.played}:
        raise KeyError(f"{date} is not an unplayed Michigan game")
    base = base if base is not None else model.simulate()
    forced = model.simulate({date: outcome})   # same seed => same rating draws
    w = _weights(model, forced["mich_mu"][date], outcome)
    ess = w.sum() ** 2 / (w ** 2).sum() if w.sum() > 0 else 0.0
    wn = w / w.sum()
    label = outcome if isinstance(outcome, str) else f"{outcome:+g}"
    return WhatIf(
        date=date, outcome=label,
        baseline=float(base["extension"].mean()),
        direct=float(forced["extension"].mean()),
        total=float(wn @ forced["extension"]),
        effective_sims=float(ess),
        benchmarks_total={k: float(wn @ v) for k, v in forced["benchmarks"].items()},
    )


def next_games(model: Model, k: int) -> list[str]:
    return [g.date for g in model.mich_games if not g.played][:k]
