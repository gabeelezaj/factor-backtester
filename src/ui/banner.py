"""The data-source banner.  Driven entirely by get_metadata(); nobody should mistake a demo for research."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_banner(meta: Dict[str, Any]) -> None:
    caveats = "\n".join(f"- {c}" for c in meta.get("caveats", []))
    coverage = f"{meta.get('coverage_start', '?')} → {meta.get('coverage_end', '?')}"
    if meta.get("is_simulated"):
        st.error(f"### ⚠️ SIMULATED DATA — {meta.get('display_name')}\n"
                 f"Every number on this page comes from a randomly generated market ({coverage}). "
                 f"**This is not a historical result** and says nothing about real stocks.\n\n{caveats}")
    elif meta.get("survivorship_bias"):
        st.warning(f"### ⚠️ DEMO DATA WITH SURVIVORSHIP BIAS — {meta.get('display_name')}\n"
                   f"Real prices ({coverage}), but the universe is today's survivors, so **results are optimistic**. "
                   f"Demo, not research.\n\n{caveats}")
    else:
        st.info(f"### Data source: {meta.get('display_name')} ({coverage})\n{caveats}")
