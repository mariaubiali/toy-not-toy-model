#!/usr/bin/env bash
set -e

HERA="../data/out/hera_nc_ep_575.npz"
OUT="out/chi2_result.npz"

MODELS=(
  # "../results/nn/chi2_th_full/nn_summary.npz"
  # "../results/nn/mse_th_full/nn_summary.npz"
  # "../results/nn/chi2_full/nn_summary.npz"
  # "../results/nn/mse_full/nn_summary.npz"
  # "../results/ntk/init/nn_summary.npz"
  # "../results/ntk/post_full/nn_summary.npz"
  # "../results/gp/rbf_full3/gp_summary.npz"
)

for model in "${MODELS[@]}"; do
  echo "========================================"
  echo "Running (exp only): $model"
  python chi2_hera.py \
    --hera "$HERA" \
    --model "$model" \
    --out "$OUT"

  echo "----------------------------------------"
  echo "Running (exp + ens): $model"
  python chi2_hera.py \
    --cov \
    --hera "$HERA" \
    --model "$model" \
    --out "$OUT"
done