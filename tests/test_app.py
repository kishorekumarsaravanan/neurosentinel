"""Smoke tests: every page of the Streamlit app renders without raising, in light and dark mode.

Pages that only need the committed artifacts/ always run; pages that read raw EEG are skipped
when data/raw is empty (e.g. on CI).
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import needs_artifacts, needs_data

ARTIFACT_PAGES = ["monitor", "game", "arena"]
DATA_PAGES = ["prologue", "atlas", "signal_lab", "xray", "diagnose"]
# absolute path: newer Streamlit resolves relative AppTest paths against the calling test file, not the cwd
ENTRY = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def run_page(name: str, theme: str) -> AppTest:
    """Open a page through the real entry point, so st.navigation / st.page_link behave as in the app."""
    at = AppTest.from_file(ENTRY, default_timeout=180)
    at.session_state["ns_theme"] = theme
    at.run()
    return at.switch_page(f"nsapp/views/{name}.py").run()


def assert_clean(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.app
@needs_artifacts
@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("page", ARTIFACT_PAGES)
def test_artifact_pages_render(page, theme):
    at = run_page(page, theme)
    assert_clean(at)
    assert at.markdown, "page rendered no content"


@pytest.mark.app
@needs_artifacts
@needs_data
@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("page", DATA_PAGES)
def test_data_pages_render(page, theme):
    assert_clean(run_page(page, theme))


@pytest.mark.app
@needs_artifacts
def test_model_comparison_scope_switch():
    at = run_page("arena", "light")
    assert any("4/4 seizures" in m.value for m in at.markdown)       # both patients: 4 test seizures
    at.button_group[0].set_value(["chb03"]).run()   # single-select control: value passed as a list
    assert_clean(at)
    assert any("2/2 seizures" in m.value for m in at.markdown)       # one patient: 2 test seizures


@pytest.mark.app
@needs_artifacts
@needs_data
def test_diagnose_scans_a_library_recording():
    at = run_page("diagnose", "light")
    at.selectbox[0].set_value("chb01_21").run()
    at.button[0].click().run()
    assert_clean(at)
    dx = at.session_state["dx"]
    assert dx["name"] == "chb01_21" and dx["known"]
    assert len(dx["times"]) > 3000                       # one decision per second over the hour


@pytest.mark.app
@needs_artifacts
def test_entry_point_navigation_renders():
    at = AppTest.from_file(ENTRY, default_timeout=180).run()
    assert_clean(at)
