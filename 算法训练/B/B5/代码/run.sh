#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
PY=${B5_PYTHON:-/workspace/b5-venv/bin/python}
DATA=${B5_DATA:-/workspace/b5-assets}
OUT=${B5_OUT:-/workspace/B5}
export CUDA_VISIBLE_DEVICES=${B5_GPU:-0}
export OMP_NUM_THREADS=4
test "${B5_STORAGE_CONFIRMED:-}" = yes || { echo 'Confirm this workspace storage retention policy, then set B5_STORAGE_CONFIRMED=yes'; exit 1; }
test "${B5_EXPECTED_COMMIT:-}" = "$(git -C "$HERE" rev-parse HEAD)" || { echo 'Set B5_EXPECTED_COMMIT to the reviewed full commit'; exit 1; }
test -z "$(git -C "$HERE" status --porcelain --untracked-files=no)" || { echo 'Tracked cloud code is modified; inspect differences first'; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" || { echo 'GPU has active compute processes'; exit 1; }
SMOKE="${OUT}-smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/train.log") 2>&1
"$PY" -u "$HERE/train.py" --data "$DATA" --out "$SMOKE" --batch "${B5_BATCH:-4}" --smoke
cp "$SMOKE/smoke.json" "$OUT/preflight_smoke.json"
"$PY" -u "$HERE/train.py" --data "$DATA" --out "$OUT" --batch "${B5_BATCH:-4}" "$@"
"$PY" -u "$HERE/evaluate.py" --data "$DATA" --out "$OUT" --final-test
tar -czf "${OUT}-results.tar.gz" --exclude=last.pt --exclude=training.lock -C "$OUT" .
sha256sum "${OUT}-results.tar.gz" > "${OUT}-results.tar.gz.sha256"
echo 'Complete. Download results and verify locally before stopping workspace compute. Do not delete the workspace.'
