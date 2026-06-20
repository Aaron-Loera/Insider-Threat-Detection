"""Headless smoke test: dashboard/app.py boots and every page renders.

This is the regression tripwire for the Phase 7 dashboard decomposition. It runs
against the synthetic data tree from conftest.py and asserts the app reaches an
authenticated, rendered state on each of the four nav pages without raising. It
deliberately exercises navigation only — not dialog/fragment interaction, which is
brittle under AppTest on current Streamlit.
"""

def test_app_boots_authenticated(app_test):
    at = app_test.run()
    assert not at.exception, f"App raised on boot: {at.exception}"
    assert at.session_state["authenticated"] is True
    # `not at.exception` is necessary but not sufficient: a failed data load is
    # *caught* by the app, which renders an error page and st.stop()s before the
    # sidebar. The nav radio only exists once DATA_LOADED is True, so its presence
    # is what actually proves the app booted healthy (not just without raising).
    assert "nav_page" in {w.key for w in at.radio}, "nav radio missing — data load failed"


def test_nav_radio_default_is_alerts(app_test):
    at = app_test.run()
    assert not at.exception
    nav = at.radio(key="nav_page")
    assert nav.value == "Alerts"


def test_all_pages_render_without_exception(app_test):
    at = app_test.run()
    assert not at.exception
    for page in ["Alerts", "Overview", "Investigation", "Channels"]:
        at.radio(key="nav_page").set_value(page).run()
        assert not at.exception, f"Page '{page}' raised: {at.exception}"


def test_filter_session_state_initialized(app_test):
    at = app_test.run()
    assert not at.exception
    # Filters bootstrap on first render (see init_filter_state in the plan).
    assert "flt_risk" in at.session_state
