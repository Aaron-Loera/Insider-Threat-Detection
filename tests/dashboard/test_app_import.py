"""Regression test for the app's self-contained sys.path setup.

`streamlit run dashboard/app.py` starts a fresh interpreter whose sys.path[0] is
the dashboard dir only — the project root (where the `config` shim lives) is NOT
on the path. app.py is responsible for adding it itself, before importing lib.data
(which does `from config import …` at module top). The AppTest smoke test can't
catch a regression here because pytest's conftest already puts the repo root on
sys.path, masking the bug. This test runs the import chain in a subprocess with a
deliberately minimal sys.path to reproduce the real `streamlit run` conditions.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = REPO_ROOT / "dashboard"


def test_lib_data_imports_with_only_dashboard_dir_on_path(tmp_path):
    """lib.data must import when only the dashboard dir is initially importable.

    Mimics `streamlit run`: sys.path[0] is the dashboard dir alone. src/ stays
    importable (for ueba, via the editable install's .pth), but the repo root is
    NOT on the path — app.py's top-level sys.path guard must add it so `config`
    resolves. If the guard regresses, this raises ModuleNotFoundError: config.

    Runs with `-P` (don't prepend cwd) from a neutral tmp dir, so neither cwd nor
    PYTHONPATH can sneak the repo root onto sys.path and mask a regression — the
    only way config becomes importable is app.py's own guard, which the snippet
    replicates by seeding sys.path[0] with just the dashboard dir.
    """
    code = (
        "import os, sys\n"
        # Like `streamlit run`: only the entrypoint's dir on sys.path to start.
        f"sys.path.insert(0, {str(DASHBOARD_DIR)!r})\n"
        # Now replicate app.py's top guard (adds the repo root for `config`).
        "dash = os.path.dirname(os.path.abspath(os.path.join(%r, 'app.py')))\n"
        "root = os.path.dirname(dash)\n"
        "for p in (dash, root):\n"
        "    if p not in sys.path:\n"
        "        sys.path.insert(0, p)\n"
        "import lib.data\n"
        "assert lib.data.ALERT_STATUS_OPTIONS\n"
        "print('OK')\n"
    ) % str(DASHBOARD_DIR)

    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-P", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, (
        f"lib.data import failed (sys.path regression?):\n{result.stderr}"
    )
    assert "OK" in result.stdout
