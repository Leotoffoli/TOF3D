#!/bin/bash
# Runs every check of the package; each writes its JSON next to it. Usage: bash tests/run_all.sh [python]
cd "$(dirname "$0")"
PY=${1:-python3}
TOF3D_FP64=1 $PY check_gradients.py cpu > check_gradients.log 2>&1; echo "check_gradients rc=$?"
TOF3D_FP64=1 $PY check_mg.py exato cpu > check_mg_exato.log 2>&1; echo "check_mg exato rc=$?"
$PY check_mg.py iter cuda > check_mg_iter.log 2>&1; echo "check_mg iter rc=$?"
$PY check_mma.py > check_mma.log 2>&1; echo "check_mma rc=$?"
$PY check_mma_gpu.py > check_mma_gpu.log 2>&1; echo "check_mma_gpu rc=$?"
$PY check_filters.py > check_filters.log 2>&1; echo "check_filters rc=$?"
echo FIM_TESTES
