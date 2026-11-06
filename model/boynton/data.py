"""Loading ratings and schedules.

Ratings come from a hand-entered CSV (KenPom front page: team, conf, NetRtg, AdjT).
KenPom forbids scraping and republishing, so data/ratings.csv is gitignored and
only probabilities are ever published. data/ratings.sample.csv holds illustrative
(perturbed) numbers so the pipeline runs without the real file.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Map common alternate spellings to KenPom's names.
ALIASES = {
    "UConn": "Connecticut",
    "Ohio State": "Ohio St.",
    "Michigan State": "Michigan St.",
    "Penn State": "Penn St.",
    "Southern California": "USC",
    "Mississippi Valley State": "Mississippi Valley St.",
    "Miami (FL)": "Miami FL",
}

BIG_TEN = [
    "Illinois", "Indiana", "Iowa", "Maryland", "Michigan", "Michigan St.",
    "Minnesota", "Nebraska", "Northwestern", "Ohio St.", "Oregon", "Penn St.",
    "Purdue", "Rutgers", "UCLA", "USC", "Washington", "Wisconsin",
]

# Fallback for teams missing from a partial ratings table: roughly a
# 250th-ranked D-I team. Big Ten teams should never hit this.
FALLBACK_NET = -7.0
FALLBACK_TEMPO = 70.0


def canon(name: str) -> str:
    name = name.strip()
    return ALIASES.get(name, name)


@dataclass
class Ratings:
    net: dict[str, float]
    tempo: dict[str, float]
    as_of: str = ""
    missing: set[str] = field(default_factory=set)

    def get(self, team: str) -> tuple[float, float]:
        team = canon(team)
        if team not in self.net:
            self.missing.add(team)
            return FALLBACK_NET, FALLBACK_TEMPO
        return self.net[team], self.tempo[team]

    def ranked(self) -> list[str]:
        return sorted(self.net, key=lambda t: -self.net[t])


def _num(s: str) -> float:
    return float(s.replace("+", ""))


# Ratings sources. Each real file is hand-entered and gitignored; when it's
# missing the source falls back to the illustrative sample file.
SOURCES = {
    "kenpom": "ratings.csv",
    "torvik": "ratings.torvik.csv",
}
SAMPLE = "ratings.sample.csv"


def source_path(source: str) -> tuple[Path, bool]:
    """Path for a ratings source, plus whether it's the real file (vs. sample)."""
    real = DATA_DIR / SOURCES[source]
    return (real, True) if real.exists() else (DATA_DIR / SAMPLE, False)


def load_ratings(path: Path | None = None, source: str = "kenpom") -> Ratings:
    """Load ratings.csv (or ratings.sample.csv), or a raw KenPom paste (.tsv/.txt).

    A raw paste is the front-page table copied from the browser: tab-separated,
    header repeated every 40 rows, value/rank pairs after NetRtg.
    """
    if path is None:
        path, _ = source_path(source)
    path = Path(path)
    text = path.read_text()
    net, tempo, as_of = {}, {}, ""
    if path.suffix in (".tsv", ".txt"):
        for line in text.splitlines():
            cols = line.split("\t")
            if len(cols) < 10 or not cols[0].strip().isdigit():
                continue
            team = re.sub(r"\s*\d+$", "", cols[1]).strip()  # drop tourney seeds
            # Rk Team Conf W-L NetRtg ORtg rk DRtg rk AdjT rk ...
            net[team] = _num(cols[4])
            tempo[team] = _num(cols[9])
    else:
        for row in csv.DictReader(text.splitlines()):
            team = canon(row["team"])
            net[team] = _num(row["netrtg"])
            tempo[team] = _num(row.get("adjt") or str(FALLBACK_TEMPO))
            as_of = row.get("as_of", as_of)
    return Ratings(net=net, tempo=tempo, as_of=as_of)


@dataclass
class Game:
    date: str
    home: str  # for neutral games, "home" is just the first-listed team
    away: str
    neutral: bool
    conference: bool
    home_score: int | None = None
    away_score: int | None = None

    @property
    def played(self) -> bool:
        return self.home_score is not None


def load_michigan_schedule(path: Path | None = None) -> list[Game]:
    """Michigan's schedule. result column like 'W 84-68' (Michigan score first)."""
    path = Path(path or DATA_DIR / "michigan_schedule.csv")
    games = []
    for row in csv.DictReader(path.read_text().splitlines()):
        opp, site = canon(row["opponent"]), row["site"].strip().upper()
        home, away = ("Michigan", opp) if site in ("H", "N") else (opp, "Michigan")
        g = Game(row["date"], home, away, site == "N", row["conference_game"] == "1")
        res = (row.get("result") or "").strip()
        if res:
            m = re.match(r"([WL])\s+(\d+)-(\d+)", res)
            mich, opp_s = int(m.group(2)), int(m.group(3))
            if (m.group(1) == "W") != (mich > opp_s):
                raise ValueError(f"result {res!r} on {row['date']} is inconsistent")
            g.home_score, g.away_score = (mich, opp_s) if home == "Michigan" else (opp_s, mich)
        games.append(g)
    return games
