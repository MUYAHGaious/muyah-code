import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checklib import pytest, unchanged  # noqa: E402

unchanged("test_stats.py", "ff93bad6942be302aabaac6d9b9c637af5ae01c55236eb2d68455ef5ea2d463f")
sys.exit(pytest("test_stats.py"))
