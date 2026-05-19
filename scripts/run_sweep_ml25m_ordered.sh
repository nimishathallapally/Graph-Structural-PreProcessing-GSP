#!/usr/bin/env bash
# run_sweep_ml25m_ordered.sh
# Custom ordered sweep for MovieLens-25M:
#   1. forman_ricci @ 1.0
#   2. cosine       @ 1.0
#   3. forman_ricci @ 0.75
#   4. forman_ricci @ 0.5
#   5. forman_ricci @ 0.25
#   6. cosine       @ 0.75
#   7. cosine       @ 0.5
# min_shared: 1, 3, 5 for each combination
# Fixed: models=lightgcn,gcn,gat,graphsage  epochs=50  patience=10  clustering=hem

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON=python
RUNNER="${SCRIPT_DIR}/run_ml25m.py"
BASE_OUT="${REPO_ROOT}/output/sweep_ml25m_ordered"

MODELS="lightgcn,gcn,gat,graphsage"
EPOCHS=50
PATIENCE=10
MIN_SHARED_VALS=(1 3 5)

# Ordered list of (mode, fraction) pairs
ORDERED_RUNS=(
    "forman_ricci 1.0"
    "cosine 1.0"
    "forman_ricci 0.75"
    "forman_ricci 0.5"
    "forman_ricci 0.25"
    "cosine 0.75"
    "cosine 0.5"
)

echo "========================================================"
echo " MovieLens-25M ordered sweep"
echo "  models    : ${MODELS}"
echo "  epochs    : ${EPOCHS}  (early_stopping_patience=${PATIENCE})"
echo "  clustering: hem"
echo "  base_out  : ${BASE_OUT}"
echo "  run order : forman_ricci@1.0, cosine@1.0, forman_ricci@0.75,"
echo "              forman_ricci@0.5, forman_ricci@0.25, cosine@0.75, cosine@0.5"
echo "  min_shared: ${MIN_SHARED_VALS[*]} (for each)"
echo "========================================================"

total=$(( ${#ORDERED_RUNS[@]} * ${#MIN_SHARED_VALS[@]} ))
run_idx=0

for run_spec in "${ORDERED_RUNS[@]}"; do
    mode=$(echo "$run_spec" | awk '{print $1}')
    frac=$(echo "$run_spec" | awk '{print $2}')
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

echo ""
echo "========================================================"
echo " Sweep complete. Results under ${BASE_OUT}"
echo "========================================================"
