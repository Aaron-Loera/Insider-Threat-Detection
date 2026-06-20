"""Sidebar, page chrome, and shared section-header / KPI-carousel UI.

render_sidebar() draws the sidebar (logo, nav radio, live status, sign-out) and
returns the active page. render_chrome() injects the page-title badge and the
mobile sidebar-toggle script. section_header() and _render_card_carousel() are the
shared layout helpers used by the page modules. NAV_PAGES is the canonical page
list, also used by app.py's dispatch dict.
"""

import streamlit as st

from lib.live import _cached_live_file_stats

NAV_PAGES = [
    "Alerts",
    "Overview",
    "Investigation",
    "Channels",
]


def render_sidebar() -> str:
    """Draw the sidebar and return the selected page (st.radio key="nav_page")."""
    with st.sidebar:
        # ── InsiderGuard AI Logo ──
        st.markdown(
            "<div class='sidebar-branding'>"
            "<svg width='44' height='44' viewBox='0 0 100 100' fill='none' xmlns='http://www.w3.org/2000/svg' style='flex-shrink:0;'>"
            "<path d='M25 85 L25 40 L15 15 L30 30 L50 25 L70 30 L85 15 L75 40 L75 85 Z' "
            "fill='#e84545' opacity='0.9'/>"
            "<circle cx='38' cy='50' r='5' fill='#000'/>"
            "<circle cx='62' cy='50' r='5' fill='#000'/>"
            "<path d='M45 60 Q50 65 55 60' stroke='#000' stroke-width='2' fill='none'/>"
            "<line x1='20' y1='55' x2='38' y2='52' stroke='#000' stroke-width='1.5'/>"
            "<line x1='20' y1='60' x2='38' y2='58' stroke='#000' stroke-width='1.5'/>"
            "<line x1='62' y1='52' x2='80' y2='55' stroke='#000' stroke-width='1.5'/>"
            "<line x1='62' y1='58' x2='80' y2='60' stroke='#000' stroke-width='1.5'/>"
            "</svg>"
            "<div style='line-height:1; white-space:nowrap;'>"
            "<div style='font-family:JetBrains Mono,monospace; font-size:13px; letter-spacing:1.5px; "
            "color:#ffffff; font-weight:700;'>InsiderGuard AI</div>"
            "<div style='font-family:JetBrains Mono,monospace; font-size:9px; letter-spacing:1.5px; "
            "color:#555555; text-transform:uppercase; margin-top:3px;'>Data Structure Kittens</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Navigation ──
        # The radio's key="nav_page" writes the selection into session_state; the
        # function returns it below (the old `active_page = ...` binding is gone).
        st.markdown("<p class='sidebar-section-label'>Navigation</p>", unsafe_allow_html=True)
        st.radio(
            "Nav",
            NAV_PAGES,
            index=0,
            label_visibility="collapsed",
            key="nav_page",
        )

        st.markdown("<div style='border-top:1px solid #1a1a1a; margin:8px 0 0 0;'></div>", unsafe_allow_html=True)

        # ── Global live status (visible on every page) ──
        _live_rows_received = 0
        _stream_done = False
        _live_session_active = bool(st.session_state.live_mode or st.session_state.live_paused)
        if _live_session_active:
            _live_rows_received, _stream_done = _cached_live_file_stats()

        _proc = st.session_state.live_proc
        _proc_running = _proc is not None and _proc.poll() is None
        if not _live_session_active:
            _status_color = "#666"
            _status_label = "IDLE"
        elif st.session_state.live_paused:
            _status_color = "#f5a623"
            _status_label = "PAUSED"
        elif _proc_running:
            _status_color = "#3c9"
            _status_label = "RUNNING"
        else:
            _status_color = "#e84545"
            _status_label = "COMPLETE" if _stream_done else "STOPPED"

        st.markdown(
            f"<div style='margin:12px 0 6px 0;'>"
            f"<span style='font-family:JetBrains Mono,monospace; font-size:11px; "
            f"color:{_status_color}; letter-spacing:1.5px;'>● {_status_label}</span>"
            f"<span style='font-family:JetBrains Mono,monospace; font-size:10px; "
            f"color:#555; margin-left:10px;'>{_live_rows_received:,} rows received</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='border-top:1px solid #1a1a1a; margin:16px 0;'></div>", unsafe_allow_html=True)

        # ── Signed-in user + logout (fixed to bottom of sidebar) ──
        _signed_in_email = st.session_state.get("auth_user_email", "")
        _email_html = (
            f"<p class='so-email'>{_signed_in_email}</p>"
            if _signed_in_email else ""
        )
        st.markdown(
            f"<div class='signout-panel'>"
            f"{_email_html}"
            f"<a class='so-btn' href='?logout=true' target='_self'>SIGN OUT</a>"
            f"</div>",
            unsafe_allow_html=True,
        )
    return st.session_state.nav_page


def render_chrome() -> None:
    """Inject the page-title badge and the mobile sidebar-toggle script."""
    st.markdown("<div class='project-title-badge'>Insider Threat Detection</div>", unsafe_allow_html=True)

    # ── Mobile sidebar toggle (injected once into parent document) ──
    st.html(
        """
        <script>
        (function(){
            var p = document;
            if (p.getElementById('mob-sidebar-btn')) return;
            var btn = p.createElement('button');
            btn.id = 'mob-sidebar-btn';
            btn.innerHTML = '&#9776;';
            btn.style.cssText = [
                'position:fixed','top:12px','left:12px','z-index:99999',
                'background:#0a0a0a','border:1px solid #1a1a1a','color:#aaa',
                'width:38px','height:38px','font-size:18px','cursor:pointer',
                'display:none','align-items:center','justify-content:center',
                'padding:0'
            ].join(';');
            p.body.appendChild(btn);
            function show(){
                btn.style.display = window.innerWidth <= 768 ? 'flex' : 'none';
            }
            window.addEventListener('resize', show);
            show();
            btn.addEventListener('click', function(){
                var sidebar = p.querySelector('[data-testid="stSidebar"]');
                var collapse = p.querySelector('[data-testid="stSidebarCollapseButton"]');
                var expand   = p.querySelector('[data-testid="stExpandSidebarButton"]');
                if (sidebar) {
                    var rect = sidebar.getBoundingClientRect();
                    if (rect.width > 50) {
                        if (collapse) collapse.click();
                    } else {
                        if (expand) expand.click();
                    }
                } else if (expand) { expand.click(); }
            });
        })();
        </script>
        """
    )


_SECTION_INFO = {
    "Risk Distribution": (
        "**Risk Distribution**\n\n"
        "A donut chart showing what proportion of activity records fall into each risk tier:\n\n"
        "- **HIGH** — anomaly score in the top percentile; warrants immediate review\n"
        "- **MEDIUM** — elevated but not critical; worth monitoring\n"
        "- **LOW** — behavior consistent with typical baseline activity\n\n"
        "Risk level is assigned per record based on the anomaly score percentile "
        "produced by the Isolation Forest model."
    ),
    "Alert Trend Over Time": (
        "**Alert Trend Over Time**\n\n"
        "An area chart showing how many alerts were generated each day, stacked by risk level.\n\n"
        "Use this to identify **spikes** or **sustained elevations** in suspicious activity "
        "within the selected date range. A sudden spike often correlates with a specific "
        "incident (e.g. large file exfiltration, off-hours access)."
    ),
    "Top 10 Riskiest Users": (
        "**Top 10 Riskiest Users**\n\n"
        "A ranked list of the users with the highest overall risk score.\n\n"
        "Each entry shows:\n"
        "- **Percentile** — how the user ranks relative to all users (100 = most anomalous)\n"
        "- **High-risk days** — number of days flagged HIGH by the model\n"
        "- **Badge** — CRITICAL (≥80th pct), HIGH (≥60th), or ELEVATED (below 60th)\n\n"
        "Click **Investigate →** to jump directly to that user's full behavioral profile."
    ),
    "Anomaly Score Distribution": (
        "**Anomaly Score Distribution**\n\n"
        "A histogram of raw anomaly scores across all records, coloured by risk level.\n\n"
        "- Scores **near 0** indicate behavior very close to the normal baseline\n"
        "- Scores **toward 1** represent increasingly anomalous activity\n\n"
        "The Isolation Forest model assigns each record a score based on how easily it "
        "can be isolated from the rest of the dataset. Outliers require fewer splits and "
        "therefore receive higher scores."
    ),
    "Cross-Channel Risk Flags (Global)": (
        "**Cross-Channel Risk Flags**\n\n"
        "Counts of records where suspicious activity co-occurred across multiple data channels:\n\n"
        "- **USB + File Write** — removable device use paired with large file write activity\n"
        "- **Off-Hours Activity** — logins or actions recorded outside normal working hours\n"
        "- **External Communication** — significant outbound traffic to external endpoints\n"
        "- **Job Site + USB** — job-site browsing paired with USB insertion on the same day\n"
        "- **Suspicious Upload** — HTTP upload to a suspicious or uncategorized domain\n"
        "- **Cloud Upload** — HTTP upload to a known cloud storage service\n"
        "- **Non-Primary PC** — sensitive activity (file copy, USB, HTTP) from an atypical endpoint\n\n"
        "These compound flags are stronger indicators of insider threat than any single "
        "channel signal alone."
    ),
    "Anomaly Score Timeline": (
        "**Anomaly Score Timeline**\n\n"
        "A day-by-day line chart of the selected user's anomaly score.\n\n"
        "- A **persistent elevation** suggests consistently unusual behavior\n"
        "- A **sudden spike** may point to a discrete incident on that date\n\n"
        "Use the date filter to narrow the window and correlate peaks with raw activity records below."
    ),
    "Behavioral Profile (Avg Activity)": (
        "**Behavioral Profile**\n\n"
        "A radar chart comparing the selected user's **average feature values** (solid line) "
        "against the **global population average** (dashed line).\n\n"
        "Each axis represents one of six behavioral channels: Authentication, File Access, "
        "Removable Media, Email, HTTP Activity, and PC Activity. "
        "Axes where the user extends significantly beyond the population average indicate "
        "dimensions of behavior worth investigating."
    ),
    "Daily Feature Activity": (
        "**Daily Feature Activity**\n\n"
        "A heatmap of the user's raw feature values over time.\n\n"
        "- Each **row** is one behavioral feature\n"
        "- Each **column** is one day\n"
        "- **Darker cells** = higher-than-usual activity on that day and feature\n\n"
        "This lets you pinpoint exactly which features drove an anomaly spike on a given date."
    ),
    "Cross-Channel Risk Indicators": (
        "**Cross-Channel Risk Indicators**\n\n"
        "A summary of whether this user triggered any multi-channel co-occurrence flags:\n\n"
        "- **USB + File Write** — device use coincided with large file write events\n"
        "- **Off-Hours Activity** — actions occurred outside normal business hours\n"
        "- **External Communication** — outbound connections to external hosts were detected\n"
        "- **Job Site + USB** — job-site browsing paired with USB insertion on the same day\n"
        "- **Suspicious Upload** — HTTP upload to a suspicious or uncategorized domain\n"
        "- **Cloud Upload** — HTTP upload to a known cloud storage service\n"
        "- **Non-Primary PC** — sensitive activity from an atypical endpoint\n\n"
        "Combinations of multiple flags substantially increase the likelihood of an insider threat."
    ),
    "Raw Activity Records": (
        "**Raw Activity Records**\n\n"
        "A full table of every aggregated daily record for the selected user within the "
        "current filter window.\n\n"
        "Each row represents one day and includes all behavioral features (email, file, "
        "HTTP, logon, device activity), the computed anomaly score, and the assigned risk level. "
        "Use this to audit exactly what the model saw on any particular date."
    ),
    "Channel Activity Volume": (
        "**Channel Activity Volume**\n\n"
        "A line chart showing the total number of events recorded per day across each data channel "
        "(Authentication, File Access, Removable Media, Email, HTTP Activity, PC Activity) "
        "within the selected filters.\n\n"
        "Channels with disproportionately high volumes relative to peers can indicate "
        "a data exfiltration path that warrants deeper investigation."
    ),
    "Channel Volume Share": (
        "**Channel Volume Share**\n\n"
        "A donut chart showing each channel's **percentage share** of all recorded events "
        "across Authentication, File Access, Removable Media, Email, HTTP Activity, and PC Activity.\n\n"
        "This gives a quick sense of which channels dominate activity organizationally. "
        "A sudden shift in these proportions between time periods may indicate an attack campaign."
    ),
    "Feature Distributions by Risk Level": (
        "**Feature Distributions by Risk Level**\n\n"
        "Box plots for each numeric feature, grouped by risk level (HIGH / MEDIUM / LOW).\n\n"
        "The box shows the interquartile range (25th–75th percentile); the line inside is the median. "
        "Features where the HIGH box sits far above LOW are the **strongest predictors** "
        "of anomalous behavioral in this dataset."
    ),
    "Feature Correlation Matrix": (
        "**Feature Correlation Matrix**\n\n"
        "A heatmap of Pearson correlations between all numeric features.\n\n"
        "- **+1 (dark red)** — features rise and fall together\n"
        "- **−1 (dark blue)** — features move in opposite directions\n"
        "- **~0** — no linear relationship\n\n"
        "Highly correlated features may be redundant for modelling, while unexpected "
        "correlations can reveal undocumented behavioral patterns."
    ),
}


def section_header(title: str, key: str) -> None:
    """Render a section header with an optional ⓘ info popover."""
    info = _SECTION_INFO.get(title, "")
    if info:
        _hdr_left, _hdr_right = st.columns([9, 1], vertical_alignment="bottom")
        with _hdr_left:
            st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
        with _hdr_right:
            with st.popover("ⓘ", use_container_width=True):
                st.markdown(info, unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def _render_card_carousel(cards: list[str], state_key: str, visible_count: int = 4) -> None:
    """Render KPI cards with native previous/next controls."""
    if not cards:
        return

    visible_count = max(1, min(visible_count, len(cards)))
    max_start = max(len(cards) - visible_count, 0)
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    start = min(max(int(st.session_state[state_key]), 0), max_start)

    left_col, card_col, right_col = st.columns([0.45, 8, 0.45], vertical_alignment="center")
    with left_col:
        prev_clicked = st.button("←", key=f"{state_key}_prev", use_container_width=True, disabled=(start == 0))
    with right_col:
        next_clicked = st.button("→", key=f"{state_key}_next", use_container_width=True, disabled=(start == max_start))

    if prev_clicked:
        start = max(0, start - visible_count)
    if next_clicked:
        start = min(max_start, start + visible_count)
    st.session_state[state_key] = start

    with card_col:
        visible_cards = cards[start : start + visible_count]
        cols = st.columns(visible_count)
        for col, card_html in zip(cols, visible_cards):
            col.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
