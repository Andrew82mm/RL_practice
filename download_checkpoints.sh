#!/usr/bin/env bash
set -e

mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"andrey821024","key":"ВСТАВЬ_ТОКЕН_СЮДА"}
EOF
chmod 600 ~/.kaggle/kaggle.json

pip install -q kaggle

python3 << 'PYEOF'
import kaggle.api as api

api.authenticate()

kernels = {
    "vit": "andrey821024/notebook7a01605639",
    "cnn": "andrey821024/notebookc5318547d8",
}

for name, kernel in kernels.items():
    print(f"\n=== {name} ===")
    resp = api.kernels_list_files(kernel)
    for f in resp:
        print(f)
PYEOF
