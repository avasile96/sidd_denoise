#!/usr/bin/env bash
# Full experiment queue, core results first. On an M4 Pro (MPS) the whole queue takes
# ~13-16 h; on a CUDA GPU with AMP it is several times faster. Every run is resumable,
# so re-running this script skips finished evaluations and resumes interrupted training.
#
#   bash scripts/run_all.sh            # everything
#   STAGES="zero core" bash scripts/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
STAGES=${STAGES:-"baselines zero core ablations sweep extra"}
LOG=runs/logs
mkdir -p "$LOG" results
export PYTHONWARNINGS=ignore

has() { [[ " $STAGES " == *" $1 "* ]]; }

train() {  # train <config> <run-name> [overrides...]
  local cfg=$1 name=$2; shift 2
  local extra=(); [[ $# -gt 0 ]] && extra=(--set "$@")
  if [[ -f runs/$name/summary.json ]]; then echo "[skip] train $name (done)"; return 0; fi
  echo "[train] $name  ($(date '+%H:%M'))"
  $PY -u scripts/train.py --config "configs/$cfg.yaml" --name "$name" ${extra[@]+"${extra[@]}"} > "$LOG/$name.train.log" 2>&1 \
    || echo "[FAIL] train $name, see $LOG/$name.train.log"
}

evaluate() {  # evaluate <run-name> [result-name] [extra args...]
  local name=$1 res=${2:-$1}; shift; [[ $# -gt 0 ]] && shift
  if [[ -f results/$res.json ]]; then echo "[skip] eval $res (done)"; return 0; fi
  if [[ ! -f runs/$name/best.pt ]]; then echo "[FAIL] eval $res: no checkpoint"; return 0; fi
  echo "[eval] $res  ($(date '+%H:%M'))"
  $PY -u scripts/evaluate.py --ckpt "runs/$name/best.pt" --name "$res" --dists "$@" > "$LOG/$res.eval.log" 2>&1 \
    && tail -1 "$LOG/$res.eval.log" || echo "[FAIL] eval $res, see $LOG/$res.eval.log"
}

# CPU-only classical baselines run in the background, at lower priority, alongside the GPU queue.
if has baselines && [[ ! -f results/bm3d_tuned.json ]]; then
  echo "[bg] baselines -> $LOG/baselines.log"
  nice -n 10 $PY -u scripts/run_baselines.py --workers 6 > "$LOG/baselines.log" 2>&1 &
  BASELINE_PID=$!
fi

if has zero && [[ ! -f results/ref_nafnet_sidd_w64.json ]]; then
  echo "[zero-shot] ($(date '+%H:%M'))"
  $PY -u scripts/zero_shot.py > "$LOG/zero_shot.log" 2>&1 || echo "[FAIL] zero-shot"
  grep -E "PSNR|sigma" "$LOG/zero_shot.log" | tail -12
fi

if has core; then
  train scunet_ft_full scunet_ft_full;          evaluate scunet_ft_full
  train nafnet_w32_scratch nafnet_w32_scratch;  evaluate nafnet_w32_scratch
fi

if has ablations; then
  train scunet_ft_lora scunet_ft_lora;          evaluate scunet_ft_lora
  train scunet_ft_headtail scunet_ft_headtail;  evaluate scunet_ft_headtail
  train unet_scratch unet_scratch;              evaluate unet_scratch
fi

if has sweep; then  # data efficiency: 16 pairs vs all 144, equal compute per point
  train sweep_nafnet sweep_nafnet_p16 data.num_pairs=16;       evaluate sweep_nafnet_p16
  train sweep_nafnet sweep_nafnet_p144;                        evaluate sweep_nafnet_p144
  train scunet_ft_full scunet_ft_full_p16 data.num_pairs=16;   evaluate scunet_ft_full_p16
  train scunet_ft_lora scunet_ft_lora_p16 data.num_pairs=16;   evaluate scunet_ft_lora_p16
fi

if has extra; then
  evaluate nafnet_w32_scratch nafnet_w32_scratch_tta --tta
  evaluate scunet_ft_full scunet_ft_full_tta --tta
  train drunet_ft_full drunet_ft_full;          evaluate drunet_ft_full
fi

if [[ -n "${BASELINE_PID:-}" ]]; then echo "waiting for baselines..."; wait "$BASELINE_PID"; tail -8 "$LOG/baselines.log"; fi
echo "[done] $(date '+%H:%M')"
