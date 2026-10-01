"""Shared fixtures.  Test code may name concrete sources; engine code may not."""
from __future__ import annotations

import pytest

from src.data import get_source


@pytest.fixture(scope="session")
def signal_source():
    """Synthetic market with the planted signal on (default strength)."""
    return get_source("synthetic", seed=42)


@pytest.fixture(scope="session")
def placebo_source():
    """Same market, signal strength zero: fundamentals carry no information about returns."""
    return get_source("synthetic", seed=42, signal_strength=0.0)
