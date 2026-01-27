#!/usr/bin/env bash
set -euo pipefail

echo "[Info] Running post-create setup..."

# Ensure git works inside container
git config --global --add safe.directory /workspaces/*

# Sync Python dependencies
echo "[Info] Installing Python dependencies..."
uv sync --locked

# Install pre-commit hooks (if available)
if uv run --quiet pre-commit --version >/dev/null 2>&1; then
  echo "[Info] Installing pre-commit hooks..."
  uv run pre-commit install
fi

# Optional sanity check
python -c "import sys; print(f'[Info] Python {sys.version}')"

echo "[Info] Post-create setup complete!"
