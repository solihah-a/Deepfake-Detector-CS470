#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -q -r requirements-colab.txt

python - <<'PY'
import torch

print("setup complete")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("gpu: none")
PY
