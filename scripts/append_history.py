"""Append one row per model run to the site's history file.

Usage: python scripts/append_history.py SUMMARY_JSON HISTORY_JSON [SOURCE]

SOURCE (kenpom or torvik) defaults to the summary's "source" field, then kenpom.
History keeps one row per day per source.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(summary_path: str, history_path: str, source: str | None = None) -> None:
    summary = json.loads(Path(summary_path).read_text())
    hist_file = Path(history_path)
    history = json.loads(hist_file.read_text()) if hist_file.exists() else []

    source = source or summary.get("source") or "kenpom"
    now = datetime.now(timezone.utc)
    row = {
        "date": now.strftime("%Y-%m-%d"),
        "source": source,
        "run_at": now.isoformat(timespec="seconds"),
        "as_of": summary.get("as_of"),
        "p_extension": summary["p_extension"],
        "benchmarks": summary.get("benchmarks", {}),
    }
    if summary.get("as_of") == "illustrative":
        # Sample ratings: stamp the summary but keep the real trend line clean.
        history_rows_written = False
    else:
        history_rows_written = True
    # One row per day: a rerun on the same day replaces that day's row.
    if history_rows_written:
        history = [
            h for h in history
            if not (h.get("date") == row["date"] and h.get("source", "kenpom") == source)
        ]
        history.append(row)
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        hist_file.write_text(json.dumps(history, indent=1) + "\n")

    # Stamp the summary the site serves with its run time.
    summary["generated_at"] = now.strftime("%Y-%m-%d %H:%M UTC")
    Path(summary_path).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
