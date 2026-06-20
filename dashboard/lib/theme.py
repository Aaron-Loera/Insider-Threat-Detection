"""Visual theme for the dashboard — palette, Plotly defaults, plot caps, base CSS.

Constants are import-side-effect-free. The risk palette and order are re-exported
from ueba.risk (the single source of truth shared with the offline pipeline) so the
dashboard never duplicates band colors. inject_base_css() loads the global stylesheet
from assets/styles.css and is the only function that touches the Streamlit runtime.
"""

from pathlib import Path

import streamlit as st

from ueba.risk import BAND_COLORS, BAND_ORDER

# assets/ sits next to the dashboard package (dashboard/assets), one level up from
# this module (dashboard/lib/theme.py).
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_STYLES_CSS = _ASSETS_DIR / "styles.css"

# Risk palette / order. RISK_TIERS is CRITICAL→LOW (reverse of BAND_ORDER's
# LOW→CRITICAL) because the dashboard renders highest severity first.
RISK_TIERS: list[str] = list(reversed(BAND_ORDER))
RISK_COLORS: dict[str, str] = dict(BAND_COLORS)

# Explicit channel→color mapping so each channel keeps its color regardless of
# which channels are present in the filtered data.
CHANNEL_COLOR_MAP: dict[str, str] = {
    "Authentication":  "#00b4d8",
    "File Access":     "#e84545",
    "Removable Media": "#d4a017",
    "Email":           "#3a86a8",
    "HTTP Activity":   "#9b59b6",
    "PC Activity":     "#e67e22",
}

# Shared Plotly layout defaults (dark theme, muted grid).
PLOTLY_LAYOUT: dict = dict(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#0a0a0a",
    font=dict(family="Inter, sans-serif", color="#999999", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#1a1a1a"),
    yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#1a1a1a"),
)

# Downsample cap for scatter/hist plots so large frames stay responsive.
MAX_PLOT_POINTS: int = 50_000


@st.cache_data(show_spinner=False)
def _read_styles_css() -> str:
    """Read the global stylesheet once per process (cached by content path)."""
    return _STYLES_CSS.read_text(encoding="utf-8")


def inject_base_css() -> None:
    """Inject the global dashboard stylesheet.

    The file holds the exact ``\\n<style>…</style>\\n`` string that used to be an
    inline literal in app.py, so the rendered output is byte-identical.
    """
    st.markdown(_read_styles_css(), unsafe_allow_html=True)
