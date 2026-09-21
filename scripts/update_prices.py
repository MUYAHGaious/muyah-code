"""Refresh the bundled price table (muyah_code/data/prices.json) from LiteLLM's public price list.

    python scripts/update_prices.py            # download and trim
    python scripts/update_prices.py FILE.json  # trim a file you already downloaded

The table is trimmed to the providers MUYAH-CODE can talk to. The CLI also refreshes its own copy once a
day into ~/.muyah/cache/prices.json (see muyah_code/pricing.py), so this only keeps the fallback current.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muyah_code.pricing import LITELLM_URL, trim_litellm  # noqa: E402

OUT = ROOT / "muyah_code" / "data" / "prices.json"


def main() -> int:
    if len(sys.argv) > 1:
        raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(LITELLM_URL, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    table = trim_litellm(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(table['models'])} models -> {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
