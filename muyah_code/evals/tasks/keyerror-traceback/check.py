import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checklib import run  # noqa: E402

default = run("app.py").stdout.strip()
Path("config.json").write_text(json.dumps({"host": "db", "timeout": 5}))
explicit = run("app.py").stdout.strip()
print(repr(default), repr(explicit))
sys.exit(0 if (default, explicit) == ("host=localhost timeout=30", "host=db timeout=5") else 1)
