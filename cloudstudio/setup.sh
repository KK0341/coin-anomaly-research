#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep the course image's Python and PyTorch available; never upgrade them implicitly.
python3 -m venv --system-site-packages .venv-cloud
source .venv-cloud/bin/activate
python -m pip install -r cloudstudio/requirements.txt
python cloudstudio/check_env.py
