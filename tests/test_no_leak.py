"""Definition of done: nothing outside src/data/ knows where data comes from."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"wrds|yfinance|synthetic", re.IGNORECASE)


def _engine_files():
    files = [p for p in (ROOT / "src").rglob("*.py") if "src/data" not in p.as_posix().replace(str(ROOT) + "/", "")]
    files += [ROOT / "app.py", ROOT / "run_backtest.py"]
    return files


def test_no_source_names_outside_src_data():
    offenders = []
    for p in _engine_files():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if FORBIDDEN.search(line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "the data abstraction leaked:\n" + "\n".join(offenders)


def test_engine_files_exist():
    assert len(_engine_files()) > 15
