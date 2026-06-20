"""Dashboard support library — extracted helpers for dashboard/app.py.

Phase 7 decomposes the dashboard monolith into focused modules under this package.

Import convention (load-bearing): app.py and these modules are run with the
*dashboard directory* on sys.path (Streamlit puts it there for the entrypoint; the
sys.path guard at the top of app.py and the test fixture replicate it). So always
import as `from lib.<mod> import …` / `import lib.<mod>` — NEVER as
`from dashboard.lib.<mod> import …`. The two spellings resolve to *distinct* module
objects (same dual-identity pitfall as `db` vs `dashboard.db`), which would split
caches and constants. Tests import the pure helpers via `dashboard.lib` only
because they never boot the Streamlit runtime.

These modules must be import-side-effect-free: constants, pure functions, and
`@st.cache_*`-decorated definitions only. All module-level execution (page config,
CSS injection, auth gate, data bootstrap, sidebar, dispatch, the refresh loop)
stays in app.py, in its current order.
"""
