#!/usr/bin/env bash
# run_sweep_yelp_compressible.sh
# Sweep dimensions:
#   curvature_mode  : cosine, forman_ricci
#   target_fraction : 0.25, 0.5, 0.75, 1.0
#   min_shared      : 1, 3, 5
# Fixed: models=lightgcn,gcn,gat,graphsage  epochs=50  patience=10  clustering=hem
# Output dirs: output/sweep_yelp/<mode>_frac<tag>_ms<N>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON=python
RUNNER="${SCRIPT_DIR}/run_yelp_compressible.py"
BASE_OUT="${REPO_ROOT}/output/sweep_yelp"

MODELS="lightgcn,gcn,gat,graphsage"
EPOCHS=50
PATIENCE=10
CURVATURE_MODES=("cosine" "forman_ricci")
FRACTIONS=("0.25" "0.5" "0.75" "1.0")
MIN_SHARED_VALS=(1 3 5)

echo "========================================================"
echo " Yelp compressible sweep"
echo "  models    : ${MODELS}"
echo "  epochs    : ${EPOCHS}  (early_stopping_patience=${PATIENCE})"
echo "  modes     : ${CURVATURE_MODES[*]}"
echo "  fractions : ${FRACTIONS[*]}"
echo "  min_shared: ${MIN_SHARED_VALS[*]}"
echo "  clustering: hem"
echo "  base_out  : ${BASE_OUT}"
echo "========================================================"

total=$(( ${#CURVATURE_MODES[@]} * ${#FRACTIONS[@]} * ${#MIN_SHARED_VALS[@]} ))
run_idx=0

for mode in "${CURVATURE_MODES[@]}"; do
    for frac in "${FRACTIONS[@]}"; do
        for ms in "${MIN_SHARED_VALS[@]}"; do
            run_idx=$(( run_idx + 1 ))
            frac_tag=$(echo "$frac" | tr -d '.')
            out_dir="${BASE_OUT}/${mode}_frac${frac_tag}_ms${ms}"

            echo ""
            echo "──────────────────────────────────────────────────────"
            echo " Run ${run_idx}/${total}: mode=${mode}  fraction=${frac}  min_shared=${ms}"
            echo " Output → ${out_dir}"
            echo "──────────────────────────────────────────────────────"
            mkdir -p "$out_dir"

            "$PYTHON" "$RUNNER" \
                --models                  "$MODELS" \
                --epochs                  "$EPOCHS" \
                --early_stopping_patience "$PATIENCE" \
                --curvature_mode          "$mode" \
                --target_fraction         "$frac" \
                --min_shared              "$ms" \
                --clustering_method       hem \
                --output_dir              "$out_dir" \
                2>&1 | tee "${out_dir}_run.log" || {
                    echo "[WARN] Run ${run_idx} (mode=${mode}, frac=${frac}, ms=${ms}) exited with error. Continuing sweep."
                }

            echo "[DONE] Run ${run_idx}/${total}: mode=${mode}  fraction=${frac}  min_shared=${ms}"
        done
    done
done

echo ""
echo "========================================================"
echo " Sweep complete. Results under ${BASE_OUT}"
echo "========================================================"
