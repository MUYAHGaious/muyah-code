import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checklib import run  # noqa: E402

plain = run("greet.py", "Bob").stdout.strip()
shout = run("greet.py", "Bob", "--shout").stdout.strip()
print(repr(plain), repr(shout))
sys.exit(0 if (plain, shout) == ("Hello, Bob!", "HELLO, BOB!") else 1)
