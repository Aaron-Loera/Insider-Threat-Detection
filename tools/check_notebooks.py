"""CI guard for tracked Jupyter notebooks.

Committed notebook outputs are intentional in this repo (they document the
training runs), so outputs are NOT stripped. This guard only enforces that
every tracked notebook is structurally valid and has not bloated past the
size budget.
"""

import subprocess
import sys

import nbformat

MAX_BYTES = 15 * 1024 * 1024  # 15 MB per notebook


def tracked_notebooks() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    failures = []
    for path in tracked_notebooks():
        try:
            with open(path, encoding="utf-8") as fh:
                nb = nbformat.read(fh, as_version=4)
            nbformat.validate(nb)
        except Exception as exc:  # nbformat raises several unrelated types
            failures.append(f"{path}: invalid notebook — {exc}")
            continue

        size = len(open(path, "rb").read())
        if size > MAX_BYTES:
            failures.append(f"{path}: {size / 1e6:.1f} MB exceeds the {MAX_BYTES / 1e6:.0f} MB budget")

    if failures:
        print("Notebook guard failed:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"Notebook guard passed ({len(tracked_notebooks())} notebooks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
