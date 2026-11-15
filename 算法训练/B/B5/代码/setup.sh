#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ENV_DIR=${B5_ENV_DIR:-/workspace/b5-venv}
test ! -e "$ENV_DIR" || { echo 'Choose a fresh B5_ENV_DIR; refusing to modify an existing environment'; exit 1; }
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
# Choose cu126 for a CUDA 12.6-capable driver; override only after checking nvidia-smi.
"$ENV_DIR/bin/python" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url "https://download.pytorch.org/whl/${B5_CUDA_WHEEL:-cu126}"
"$ENV_DIR/bin/python" -m pip install -r "$HERE/requirements.txt"
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -c 'import sys,torch,torchvision; print(sys.executable,torch.__version__,torchvision.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(),torch.cuda.is_bf16_supported())'
