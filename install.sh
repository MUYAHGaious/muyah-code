#!/bin/sh
# MUYAH-CODE installer for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/MUYAHGaious/muyah-code/main/install.sh | sh
#
# Installs with uv or pipx when you have one. Otherwise it makes a private virtual environment in
# ~/.local/share/muyah-code (so it never collides with system Python packages, and works where pip refuses
# to install for the user). Then it makes sure `muyah` works from any folder by adding the launcher's
# folder to PATH in your shell start-up files.
set -eu

SOURCE="git+https://github.com/MUYAHGaious/muyah-code"
has() { command -v "$1" >/dev/null 2>&1; }

echo "Installing MUYAH-CODE..."

if has uv; then
    uv tool install --force "$SOURCE"
    uv tool update-shell || true
    BIN="$(uv tool dir --bin)"
elif has pipx; then
    pipx install --force "$SOURCE"
    pipx ensurepath || true
    BIN="$HOME/.local/bin"
else
    PY=""
    for candidate in python3 python; do
        if has "$candidate" && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
            PY="$candidate"
            break
        fi
    done
    if [ -z "$PY" ]; then
        echo "MUYAH-CODE needs Python 3.10 or newer (https://www.python.org/downloads/), then run this again."
        exit 1
    fi
    VENV="$HOME/.local/share/muyah-code/venv"
    "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade --quiet pip
    "$VENV/bin/python" -m pip install --upgrade --quiet "$SOURCE"
    "$VENV/bin/python" -m muyah_code path
    BIN="$VENV/bin"
fi

echo ""
if has muyah; then
    echo "Done. Type 'muyah' in any folder to start."
else
    echo "Installed. Open a new terminal (or run: export PATH=\"$BIN:\$PATH\") and type 'muyah'."
fi
