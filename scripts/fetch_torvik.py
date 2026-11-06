"""Fetch Bart Torvik's season team-results CSV and write the model's ratings file.

Usage: python scripts/fetch_torvik.py OUT_CSV [--year YYYY]

One request per run, to the downloadable file Torvik publishes for the season
(https://barttorvik.com/<year>_team_results.csv). robots.txt allows this path for
generic agents and asks for a 10-second crawl delay, which a once-a-day job meets.
The .json files are disallowed in robots.txt, so this never requests them.

The output has the same header as the hand-entered secret
(team,conf,netrtg,adjt,as_of) with netrtg = AdjOE - AdjDE. It stays out of git
(.gitignore); only derived probabilities are published.

Exits non-zero, without touching OUT_CSV, if the download fails or the file
does not look like a full ratings table, so the workflow can fall back to the
RATINGS_TORVIK_CSV secret.
"""
import argparse
import csv
import io
import sys
import urllib.request
from datetime import date
from pathlib import Path

USER_AGENT = "boynton-tracker/1.0 (+https://github.com/andy-jordan/boynton-tracker; once daily)"
MIN_TEAMS = 300


def season_year(today: date) -> int:
    """Torvik names a season by the year it ends: Nov 2026 is the 2027 season."""
    return today.year + 1 if today.month >= 8 else today.year


def download(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        ctype = resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8-sig", errors="replace")
    if "html" in ctype.lower() or body.lstrip().startswith("<"):
        raise ValueError(f"got an HTML page, not CSV (Content-Type {ctype!r})")
    return body


def _col(header: list[str], *names: str) -> int:
    norm = [h.strip().lower() for h in header]
    for n in names:
        if n in norm:
            return norm.index(n)
    raise ValueError(f"column {names[0]!r} not in header {header[:12]}...")


def convert(text: str, as_of: str) -> list[dict]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("empty file")
    header, data = rows[0], rows[1:]
    i_team, i_conf = _col(header, "team"), _col(header, "conf")
    i_oe, i_de = _col(header, "adjoe"), _col(header, "adjde")
    i_t = _col(header, "adjt", "adj t.", "adj_t", "adj tempo")
    out = []
    for r in data:
        if len(r) <= max(i_team, i_conf, i_oe, i_de, i_t) or not r[i_team].strip():
            continue
        oe, de, t = float(r[i_oe]), float(r[i_de]), float(r[i_t])
        if not (70 < oe < 150 and 70 < de < 150 and 55 < t < 85):
            raise ValueError(f"implausible values for {r[i_team]}: {oe}, {de}, {t}")
        out.append({
            "team": r[i_team].strip(),
            "conf": r[i_conf].strip(),
            "netrtg": f"{oe - de:.2f}",
            "adjt": f"{t:.1f}",
            "as_of": as_of,
        })
    if len(out) < MIN_TEAMS:
        raise ValueError(f"only {len(out)} teams, expected at least {MIN_TEAMS}")
    if not any(row["team"] == "Michigan" for row in out):
        raise ValueError("Michigan not found")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--year", type=int, default=season_year(date.today()))
    args = ap.parse_args()

    url = f"https://barttorvik.com/{args.year}_team_results.csv"
    try:
        rows = convert(download(url), f"{date.today().isoformat()} torvik")
    except Exception as e:  # any failure means "use the secret instead"
        print(f"Torvik fetch failed ({url}): {e}", file=sys.stderr)
        return 1

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["team", "conf", "netrtg", "adjt", "as_of"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    Path(args.out).write_text(buf.getvalue())
    print(f"Fetched {len(rows)} Torvik teams")  # no values: workflow logs are public
    return 0


if __name__ == "__main__":
    sys.exit(main())
