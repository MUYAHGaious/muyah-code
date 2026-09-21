import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checklib import pytest, run  # noqa: E402

leftovers = [str(p) for p in Path(".").rglob("*.py") if "calc_total" in p.read_text(encoding="utf-8")]
if leftovers:
    sys.exit(f"FAIL: calc_total still referenced in {leftovers}")
tests = pytest("test_shop.py")
report = run("report.py").stdout.strip()
print(report)
sys.exit(0 if tests == 0 and report == "33.55" else 1)
