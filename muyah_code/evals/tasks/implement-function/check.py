import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checklib import pytest, unchanged  # noqa: E402

unchanged("test_slugify.py", "f6f0e345cb692f1dfed8dc3c87457655ecd0566a787fd7758eb9a26b0914ab2f")
sys.exit(pytest("test_slugify.py"))
