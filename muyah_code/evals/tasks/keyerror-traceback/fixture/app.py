import json
from pathlib import Path


def load_config(path="config.json"):
    return json.loads(Path(path).read_text())


def describe(cfg):
    return f"host={cfg['host']} timeout={cfg['timeout']}"


if __name__ == "__main__":
    print(describe(load_config()))
