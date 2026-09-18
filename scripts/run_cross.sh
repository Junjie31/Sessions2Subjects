#!/bin/bash
# Serial training of all six directed cross-domain pairs (36 runs), preventing OOM.
#
#   for each direction (6) x mode (none, osc_real) x seed (42, 100, 202):
#       python rppg/main.py --config_file <cfg> --seed <seed>
#
# Skips a config whose Epoch29 checkpoint already exists (resumable). Run from
# the repo root after activating the environment and generating configs:
#
#   conda activate <env>
#   python scripts/gen_configs.py --data-root /path/to/Data
#   bash scripts/run_cross.sh
cd "$(dirname "$0")/.." || exit 1

PY="${PYTHON:-python}"

CONFIG_DIR=rppg/configs/train_configs
LOG_DIR=logs
mkdir -p "$LOG_DIR"

DIRECTIONS="PURE_MMPD UBFC_MMPD MMPD_PURE MMPD_UBFC PURE_UBFC UBFC_PURE"
MODES="none osc_real"
SEEDS="42 100 202"

echo "=============================================================="
echo "Six-direction cross-domain training"
echo "Start: $(date)"
echo "=============================================================="

run_one() {
  local cfg=$1
  local name seed log mae
  name=$(basename "$cfg" .yaml)
  seed=$(echo "$name" | grep -oP 'S\K\d+')
  log="$LOG_DIR/${name}.log"

  if find PreTrainedModels -name "${name}_Epoch29.pth" 2>/dev/null | grep -q .; then
    echo "SKIP (checkpoint exists): ${name}"
    return 0
  fi

  echo "--- ${name} @ $(date '+%H:%M') ---"
  if ! "$PY" rppg/main.py --config_file "$cfg" --seed "$seed" > "$log" 2>&1; then
    echo "FAILED: ${name} @ $(date '+%H:%M')"
    return 1
  fi
  mae=$(grep -a "FFT MAE (FFT Label)" "$log" | tail -1 | grep -oP '[\d.]+(?= \+/\-)' | head -1)
  echo "DONE MAE=${mae:-?} @ $(date '+%H:%M')"
}

failures=0
for dir in $DIRECTIONS; do
  for mode in $MODES; do
    for seed in $SEEDS; do
      run_one "${CONFIG_DIR}/${dir}_${mode}_S${seed}.yaml" \
        || failures=$((failures + 1))
    done
  done
done

if [ "$failures" -gt 0 ]; then
  echo "Completed with ${failures} failed run(s)."
  exit 1
fi
echo "All 36 runs completed successfully."

echo ""
echo "=== All done: $(date) ==="
