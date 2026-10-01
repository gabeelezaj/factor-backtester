"""The Streamlit app runs headlessly, shows the banner from get_metadata(), and renders every tab."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.ui.banner import render_banner


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    assert not app.exception, app.exception


def test_banner_reads_as_simulated(app):
    texts = [e.value for e in app.error]
    assert texts and "SIMULATED DATA" in texts[0] and "not a historical result" in texts[0]


def test_headline_metrics_and_tabs(app):
    labels = [m.label for m in app.metric]
    for want in ("Annual return, net of costs", "vs equal-weighted universe", "Luck check", "Signal captured"):
        assert want in labels
    assert [t.label for t in app.tabs] == ["Performance", "Metrics", "Calendar years", "Deciles", "Holdings", "Config & export"]
    assert any("configuration(s) tried" in w.value for w in app.warning)


def test_sidebar_controls_present(app):
    labels = {s.label for s in app.sidebar.slider} | {n.label for n in app.sidebar.number_input}
    for want in ("Backtest years", "Number of stocks to hold", "Largest N companies by market cap", "Minimum market cap ($ millions)",
                 "Planted signal strength", "Earnings yield", "Return on capital"):
        assert want in labels, want
    assert any(m.label == "Factors" for m in app.sidebar.multiselect)
    assert any(r.label == "Rebalance" for r in app.sidebar.radio)
    assert any(sb.label == "Preset" for sb in app.sidebar.selectbox)


def test_changing_weights_reruns(app):
    w = [s for s in app.sidebar.slider if s.label == "Earnings yield"][0]
    before = [m.value for m in app.metric if m.label == "Annual return, net of costs"][0]
    w.set_value(100).run()
    assert not app.exception
    after = [m.value for m in app.metric if m.label == "Annual return, net of costs"][0]
    assert after != before


def test_no_code_names_in_visible_labels(app):
    """Factor code names with underscores must not appear in any control label or metric."""
    visible = [s.label for s in app.sidebar.slider] + [m.label for m in app.metric] + [t.label for t in app.tabs]
    assert not any("_" in v for v in visible), [v for v in visible if "_" in v]


def test_banner_variants(monkeypatch):
    captured = {}
    import src.ui.banner as b
    monkeypatch.setattr(b.st, "error", lambda msg: captured.setdefault("error", msg))
    monkeypatch.setattr(b.st, "warning", lambda msg: captured.setdefault("warning", msg))
    monkeypatch.setattr(b.st, "info", lambda msg: captured.setdefault("info", msg))
    render_banner({"display_name": "X", "is_simulated": True, "caveats": ["c1"]})
    render_banner({"display_name": "Y", "is_simulated": False, "survivorship_bias": True, "caveats": ["c2"]})
    render_banner({"display_name": "Z", "is_simulated": False, "survivorship_bias": False, "caveats": []})
    assert "SIMULATED" in captured["error"] and "c1" in captured["error"]
    assert "SURVIVORSHIP" in captured["warning"] and "optimistic" in captured["warning"]
    assert "Z" in captured["info"]
