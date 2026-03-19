#!/usr/bin/env bash
# Helper script for developers: builds the csdecomp wheel from source
# and installs it into the examples venv, replacing the PyPI version.
#
# Usage: cd examples && bash dev_install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Determine the Python version configured in Bazel
PY_VERSION=$(cd "$REPO_ROOT" && python3 -c "
import re
with open('tools/my_python_version.bzl') as f:
    m = re.search(r'return \"(\d+\.\d+)\"', f.read())
    print(m.group(1) if m else '3.10')
")
echo "Configured Python version: $PY_VERSION"

echo "Building csdecomp wheel..."
cd "$REPO_ROOT"
bazel build //csdecomp/src/pybind/csdecomp:csdecomp_wheel

WHEEL=$(find bazel-bin/csdecomp/src/pybind/csdecomp -name "csdecomp-*.whl" | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: No wheel found after build."
    exit 1
fi
echo "Found wheel: $WHEEL"

cd "$SCRIPT_DIR"

# Ensure the venv exists with the matching Python version
if [ ! -d ".venv" ]; then
    echo "Creating venv with uv sync (Python $PY_VERSION)..."
    uv sync --python "$PY_VERSION"
fi

echo "Installing local wheel (replacing PyPI version)..."
uv pip install --reinstall "$REPO_ROOT/$WHEEL"

echo "Done! Run examples with: uv run python <script.py>"
