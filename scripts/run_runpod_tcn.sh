#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT="${PERSIST_ROOT:-/workspace}"
INPUT_ROOT="${INPUT_ROOT:-$PERSIST_ROOT/input}"
CODE_ARCHIVE="${CODE_ARCHIVE:-$INPUT_ROOT/trade_bot_code.tgz}"
PREPARED_ARCHIVE="${PREPARED_ARCHIVE:-$INPUT_ROOT/prepared_target.tgz}"
WORKDIR="${WORKDIR:-$PERSIST_ROOT/trade_bot}"
PREPARED_ROOT_REL="${PREPARED_ROOT_REL:-data/prepared/eurusd_1min_ote_full}"
TARGET="${TARGET:-long_reversal}"
PROFILE="${PROFILE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEQUENCE_MEMORY_AUTO_FRACTION="${SEQUENCE_MEMORY_AUTO_FRACTION:-0.78}"
PRELOAD_TO_DEVICE="${PRELOAD_TO_DEVICE:-1}"
ALLOW_TF32="${ALLOW_TF32:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
USE_AMP="${USE_AMP:-0}"
TORCH_NUM_WORKERS="${TORCH_NUM_WORKERS:-8}"
TORCH_EVAL_NUM_WORKERS="${TORCH_EVAL_NUM_WORKERS:-4}"
TORCH_PREFETCH_FACTOR="${TORCH_PREFETCH_FACTOR:-4}"
TORCH_PERSISTENT_WORKERS="${TORCH_PERSISTENT_WORKERS:-1}"
SKIP_APT="${SKIP_APT:-0}"
CLEAN_WORKDIR="${CLEAN_WORKDIR:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT_REL="${OUTPUT_ROOT_REL:-models/runpod_1min_tcn_${TARGET}_${RUN_ID}}"

set_default() {
  local var_name="$1"
  local default_value="$2"
  if [[ -z "${!var_name:-}" ]]; then
    printf -v "$var_name" "%s" "$default_value"
  fi
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  "$@"
}

resolve_profile() {
  if [[ "$PROFILE" != "auto" ]]; then
    return
  fi

  case "$TARGET" in
    *reversal*)
      PROFILE="reversal"
      ;;
    *breakout*)
      PROFILE="breakout"
      ;;
    *continuation*)
      PROFILE="continuation"
      ;;
    *ote*)
      PROFILE="ote"
      ;;
    *)
      PROFILE="balanced"
      ;;
  esac
}

apply_profile_defaults() {
  case "$PROFILE" in
    reversal)
      set_default N_TRIALS 48
      set_default CV_INITIAL_TRAIN_ROWS 400000
      set_default CV_VAL_ROWS 100000
      set_default CV_STEP_ROWS 100000
      set_default CV_MAX_TRAIN_ROWS 700000
      set_default CV_MIN_FOLDS 3
      set_default MAX_LOADED_FEATURES 96
      set_default TOP_FEATURE_MIN 24
      set_default TOP_FEATURE_MAX 96
      set_default WINDOW_MIN 20
      set_default WINDOW_MAX 28
      set_default EPOCHS 48
      set_default BATCH_SIZE 256
      set_default HIDDEN_SIZE 64
      set_default NUM_LAYERS 2
      set_default LEARNING_RATE 0.001
      set_default THRESHOLD_EVENT_FBETA_WEIGHT 0.45
      set_default THRESHOLD_EVENT_PRECISION_WEIGHT 0.55
      set_default THRESHOLD_TURNOVER_PENALTY_WEIGHT 0.35
      set_default THRESHOLD_TURNOVER_TARGET_RATIO 0.90
      set_default OBJECTIVE_AVERAGE_PRECISION_WEIGHT 0.35
      set_default OBJECTIVE_THRESHOLD_SCORE_WEIGHT 0.55
      set_default OBJECTIVE_BRIER_PENALTY_WEIGHT 0.10
      set_default FOCAL_ALPHA_MIN 0.70
      set_default FOCAL_ALPHA_MAX 0.78
      set_default FOCAL_GAMMA_MIN 2.60
      set_default FOCAL_GAMMA_MAX 3.00
      set_default HARD_NEGATIVE_RADIUS_MIN 1
      set_default HARD_NEGATIVE_RADIUS_MAX 1
      set_default HARD_NEGATIVE_MULTIPLIER_MIN 1.90
      set_default HARD_NEGATIVE_MULTIPLIER_MAX 2.20
      set_default TORCH_WARMUP_EPOCHS 6
      set_default TORCH_MAIN_EPOCHS 22
      set_default TORCH_FINE_EPOCHS 12
      set_default TORCH_TAIL_EPOCHS 8
      set_default TORCH_FINE_LR_SCALE 0.20
      set_default TORCH_TAIL_LR_SCALE 0.07
      ;;
    breakout)
      set_default N_TRIALS 36
      set_default CV_INITIAL_TRAIN_ROWS 320000
      set_default CV_VAL_ROWS 100000
      set_default CV_STEP_ROWS 100000
      set_default CV_MAX_TRAIN_ROWS 600000
      set_default CV_MIN_FOLDS 3
      set_default MAX_LOADED_FEATURES 96
      set_default TOP_FEATURE_MIN 24
      set_default TOP_FEATURE_MAX 96
      set_default WINDOW_MIN 24
      set_default WINDOW_MAX 36
      set_default EPOCHS 48
      set_default BATCH_SIZE 256
      set_default HIDDEN_SIZE 64
      set_default NUM_LAYERS 4
      set_default LEARNING_RATE 0.0008
      set_default THRESHOLD_EVENT_FBETA_WEIGHT 0.35
      set_default THRESHOLD_EVENT_PRECISION_WEIGHT 0.65
      set_default THRESHOLD_TURNOVER_PENALTY_WEIGHT 0.75
      set_default THRESHOLD_TURNOVER_TARGET_RATIO 0.70
      set_default OBJECTIVE_AVERAGE_PRECISION_WEIGHT 0.25
      set_default OBJECTIVE_THRESHOLD_SCORE_WEIGHT 0.65
      set_default OBJECTIVE_BRIER_PENALTY_WEIGHT 0.10
      set_default FOCAL_ALPHA_MIN 0.70
      set_default FOCAL_ALPHA_MAX 0.95
      set_default FOCAL_GAMMA_MIN 1.40
      set_default FOCAL_GAMMA_MAX 2.60
      set_default HARD_NEGATIVE_RADIUS_MIN 1
      set_default HARD_NEGATIVE_RADIUS_MAX 3
      set_default HARD_NEGATIVE_MULTIPLIER_MIN 1.00
      set_default HARD_NEGATIVE_MULTIPLIER_MAX 1.60
      set_default TORCH_WARMUP_EPOCHS 6
      set_default TORCH_MAIN_EPOCHS 22
      set_default TORCH_FINE_EPOCHS 12
      set_default TORCH_TAIL_EPOCHS 8
      set_default TORCH_FINE_LR_SCALE 0.20
      set_default TORCH_TAIL_LR_SCALE 0.07
      ;;
    continuation)
      set_default N_TRIALS 32
      set_default CV_INITIAL_TRAIN_ROWS 300000
      set_default CV_VAL_ROWS 100000
      set_default CV_STEP_ROWS 100000
      set_default CV_MAX_TRAIN_ROWS 500000
      set_default CV_MIN_FOLDS 3
      set_default MAX_LOADED_FEATURES 128
      set_default TOP_FEATURE_MIN 24
      set_default TOP_FEATURE_MAX 96
      set_default WINDOW_MIN 16
      set_default WINDOW_MAX 28
      set_default EPOCHS 32
      set_default BATCH_SIZE 256
      set_default HIDDEN_SIZE 64
      set_default NUM_LAYERS 2
      set_default LEARNING_RATE 0.001
      set_default THRESHOLD_EVENT_FBETA_WEIGHT 0.50
      set_default THRESHOLD_EVENT_PRECISION_WEIGHT 0.50
      set_default THRESHOLD_TURNOVER_PENALTY_WEIGHT 0.45
      set_default THRESHOLD_TURNOVER_TARGET_RATIO 0.85
      set_default OBJECTIVE_AVERAGE_PRECISION_WEIGHT 0.40
      set_default OBJECTIVE_THRESHOLD_SCORE_WEIGHT 0.50
      set_default OBJECTIVE_BRIER_PENALTY_WEIGHT 0.10
      set_default FOCAL_ALPHA_MIN 0.75
      set_default FOCAL_ALPHA_MAX 0.90
      set_default FOCAL_GAMMA_MIN 1.80
      set_default FOCAL_GAMMA_MAX 3.20
      set_default HARD_NEGATIVE_RADIUS_MIN 1
      set_default HARD_NEGATIVE_RADIUS_MAX 4
      set_default HARD_NEGATIVE_MULTIPLIER_MIN 1.20
      set_default HARD_NEGATIVE_MULTIPLIER_MAX 2.00
      set_default TORCH_WARMUP_EPOCHS 4
      set_default TORCH_MAIN_EPOCHS 16
      set_default TORCH_FINE_EPOCHS 8
      set_default TORCH_TAIL_EPOCHS 4
      set_default TORCH_FINE_LR_SCALE 0.25
      set_default TORCH_TAIL_LR_SCALE 0.10
      ;;
    ote)
      set_default N_TRIALS 40
      set_default CV_INITIAL_TRAIN_ROWS 400000
      set_default CV_VAL_ROWS 120000
      set_default CV_STEP_ROWS 120000
      set_default CV_MAX_TRAIN_ROWS 700000
      set_default CV_MIN_FOLDS 3
      set_default MAX_LOADED_FEATURES 128
      set_default TOP_FEATURE_MIN 24
      set_default TOP_FEATURE_MAX 96
      set_default WINDOW_MIN 16
      set_default WINDOW_MAX 32
      set_default EPOCHS 40
      set_default BATCH_SIZE 256
      set_default HIDDEN_SIZE 64
      set_default NUM_LAYERS 3
      set_default LEARNING_RATE 0.001
      set_default THRESHOLD_EVENT_FBETA_WEIGHT 0.55
      set_default THRESHOLD_EVENT_PRECISION_WEIGHT 0.45
      set_default THRESHOLD_TURNOVER_PENALTY_WEIGHT 0.40
      set_default THRESHOLD_TURNOVER_TARGET_RATIO 0.85
      set_default OBJECTIVE_AVERAGE_PRECISION_WEIGHT 0.45
      set_default OBJECTIVE_THRESHOLD_SCORE_WEIGHT 0.45
      set_default OBJECTIVE_BRIER_PENALTY_WEIGHT 0.10
      set_default FOCAL_ALPHA_MIN 0.70
      set_default FOCAL_ALPHA_MAX 0.95
      set_default FOCAL_GAMMA_MIN 1.25
      set_default FOCAL_GAMMA_MAX 3.75
      set_default HARD_NEGATIVE_RADIUS_MIN 1
      set_default HARD_NEGATIVE_RADIUS_MAX 8
      set_default HARD_NEGATIVE_MULTIPLIER_MIN 1.00
      set_default HARD_NEGATIVE_MULTIPLIER_MAX 2.50
      set_default TORCH_WARMUP_EPOCHS 4
      set_default TORCH_MAIN_EPOCHS 18
      set_default TORCH_FINE_EPOCHS 10
      set_default TORCH_TAIL_EPOCHS 8
      set_default TORCH_FINE_LR_SCALE 0.35
      set_default TORCH_TAIL_LR_SCALE 0.10
      ;;
    balanced)
      set_default N_TRIALS 32
      set_default CV_INITIAL_TRAIN_ROWS 320000
      set_default CV_VAL_ROWS 100000
      set_default CV_STEP_ROWS 100000
      set_default CV_MAX_TRAIN_ROWS 600000
      set_default CV_MIN_FOLDS 3
      set_default MAX_LOADED_FEATURES 96
      set_default TOP_FEATURE_MIN 24
      set_default TOP_FEATURE_MAX 96
      set_default WINDOW_MIN 16
      set_default WINDOW_MAX 32
      set_default EPOCHS 32
      set_default BATCH_SIZE 256
      set_default HIDDEN_SIZE 64
      set_default NUM_LAYERS 2
      set_default LEARNING_RATE 0.001
      set_default THRESHOLD_EVENT_FBETA_WEIGHT 0.50
      set_default THRESHOLD_EVENT_PRECISION_WEIGHT 0.50
      set_default THRESHOLD_TURNOVER_PENALTY_WEIGHT 0.40
      set_default THRESHOLD_TURNOVER_TARGET_RATIO 0.85
      set_default OBJECTIVE_AVERAGE_PRECISION_WEIGHT 0.40
      set_default OBJECTIVE_THRESHOLD_SCORE_WEIGHT 0.50
      set_default OBJECTIVE_BRIER_PENALTY_WEIGHT 0.10
      set_default FOCAL_ALPHA_MIN 0.70
      set_default FOCAL_ALPHA_MAX 0.90
      set_default FOCAL_GAMMA_MIN 1.50
      set_default FOCAL_GAMMA_MAX 3.25
      set_default HARD_NEGATIVE_RADIUS_MIN 1
      set_default HARD_NEGATIVE_RADIUS_MAX 4
      set_default HARD_NEGATIVE_MULTIPLIER_MIN 1.10
      set_default HARD_NEGATIVE_MULTIPLIER_MAX 2.20
      set_default TORCH_WARMUP_EPOCHS 4
      set_default TORCH_MAIN_EPOCHS 16
      set_default TORCH_FINE_EPOCHS 8
      set_default TORCH_TAIL_EPOCHS 4
      set_default TORCH_FINE_LR_SCALE 0.25
      set_default TORCH_TAIL_LR_SCALE 0.10
      ;;
    *)
      echo "Unsupported PROFILE=$PROFILE" >&2
      exit 1
      ;;
  esac
}

maybe_install_prereqs() {
  if [[ "$SKIP_APT" == "1" ]]; then
    return
  fi

  run_privileged apt-get update
  run_privileged apt-get install -y python3-venv build-essential tmux
}

prepare_workspace() {
  mkdir -p "$PERSIST_ROOT" "$INPUT_ROOT"

  if [[ "$CLEAN_WORKDIR" == "1" ]]; then
    rm -rf "$WORKDIR"
  fi

  mkdir -p "$WORKDIR"
  tar -xzf "$CODE_ARCHIVE" -C "$WORKDIR" --strip-components=1
  cd "$WORKDIR"

  "$PYTHON_BIN" -m venv .venv --system-site-packages
  # shellcheck disable=SC1091
  source .venv/bin/activate

  pip install --upgrade pip setuptools wheel
  pip install --no-cache-dir numpy pandas scipy scikit-learn xgboost optuna joblib psutil tqdm pyyaml

  if ! python - <<'PY'
import torch
print(torch.__version__)
PY
  then
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  fi
}

extract_prepared_data() {
  local prepared_root_abs="$WORKDIR/$PREPARED_ROOT_REL"

  mkdir -p "$prepared_root_abs"
  tar -xzf "$PREPARED_ARCHIVE" -C "$prepared_root_abs"

  if [[ ! -d "$prepared_root_abs/$TARGET" ]]; then
    echo "Prepared target directory was not found after extraction: $prepared_root_abs/$TARGET" >&2
    exit 1
  fi
}

print_runtime_summary() {
  local gpu_name=""
  local gpu_mem=""
  local cpu_count=""
  local ram_available_gib=""

  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)"
  gpu_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 || true)"
  cpu_count="$(nproc || echo unknown)"
  ram_available_gib="$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"

  echo "=================================================="
  echo "RunPod 1-Minute TCN Full Run"
  echo "Target: $TARGET"
  echo "Profile: $PROFILE"
  echo "Persistent root: $PERSIST_ROOT"
  echo "Input root: $INPUT_ROOT"
  echo "Workdir: $WORKDIR"
  echo "Prepared root: $PREPARED_ROOT_REL"
  echo "Output root: $OUTPUT_ROOT_REL"
  echo "GPU: ${gpu_name:-unknown} (${gpu_mem:-unknown} MiB)"
  echo "vCPU: ${cpu_count}"
  echo "Available RAM: ${ram_available_gib:-unknown} GiB"
  echo "RunPod Pod ID: ${RUNPOD_POD_ID:-unknown}"
  echo "RunPod Public IP: ${RUNPOD_PUBLIC_IP:-unknown}"
  echo "RunPod SSH Port: ${RUNPOD_TCP_PORT_22:-unknown}"
  echo "Sequence memory auto fraction: $SEQUENCE_MEMORY_AUTO_FRACTION"
  echo "Torch preload to device: $PRELOAD_TO_DEVICE"
  echo "Torch TF32 enabled: $ALLOW_TF32"
  echo "Torch cuDNN benchmark: $CUDNN_BENCHMARK"
  echo "Torch AMP enabled: $USE_AMP"
  echo "Torch loader workers: train=$TORCH_NUM_WORKERS eval=$TORCH_EVAL_NUM_WORKERS prefetch=$TORCH_PREFETCH_FACTOR persistent=$TORCH_PERSISTENT_WORKERS"
  echo "Trials: $N_TRIALS"
  echo "CV rows: train=$CV_INITIAL_TRAIN_ROWS val=$CV_VAL_ROWS step=$CV_STEP_ROWS cap=$CV_MAX_TRAIN_ROWS"
  echo "Features: max=$MAX_LOADED_FEATURES top_min=$TOP_FEATURE_MIN top_max=$TOP_FEATURE_MAX"
  echo "Window search: min=$WINDOW_MIN max=$WINDOW_MAX"
  echo "Epoch schedule: total=$EPOCHS warmup=$TORCH_WARMUP_EPOCHS main=$TORCH_MAIN_EPOCHS fine=$TORCH_FINE_EPOCHS tail=$TORCH_TAIL_EPOCHS"
  echo "=================================================="

  if [[ "$TARGET" == "short_breakout" ]]; then
    echo "Note: historical short-breakout TCNs had excellent CV but weak policy backtests. Treat this as a research run until policy metrics confirm it."
  fi
  if [[ "$TARGET" == *continuation* ]]; then
    echo "Note: continuation-family TCN evidence is thinner than reversal and OTE. Expect to review this target more critically after training."
  fi
}

run_training() {
  cd "$WORKDIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate

  mkdir -p logs
  local log_file="logs/runpod_tcn_${TARGET}_${RUN_ID}.log"

  export PYTHONUNBUFFERED=1

  local -a cmd=(
    python
    -m
    model_training.ote_training.ote_xgboost_pipeline
    --prepared-root "$PREPARED_ROOT_REL"
    --output-root "$OUTPUT_ROOT_REL"
    --backend torch
    --model-type tcn
    --targets "$TARGET"
    --trials "$N_TRIALS"
    --cv-initial-train-rows "$CV_INITIAL_TRAIN_ROWS"
    --cv-val-rows "$CV_VAL_ROWS"
    --cv-step-rows "$CV_STEP_ROWS"
    --cv-max-train-rows "$CV_MAX_TRAIN_ROWS"
    --cv-min-folds "$CV_MIN_FOLDS"
    --max-loaded-features "$MAX_LOADED_FEATURES"
    --top-feature-min "$TOP_FEATURE_MIN"
    --top-feature-max "$TOP_FEATURE_MAX"
    --window-min "$WINDOW_MIN"
    --window-max "$WINDOW_MAX"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --hidden-size "$HIDDEN_SIZE"
    --num-layers "$NUM_LAYERS"
    --learning-rate "$LEARNING_RATE"
    --sequence-memory-auto-fraction "$SEQUENCE_MEMORY_AUTO_FRACTION"
    --threshold-event-fbeta-weight "$THRESHOLD_EVENT_FBETA_WEIGHT"
    --threshold-event-precision-weight "$THRESHOLD_EVENT_PRECISION_WEIGHT"
    --threshold-turnover-penalty-weight "$THRESHOLD_TURNOVER_PENALTY_WEIGHT"
    --threshold-turnover-target-ratio "$THRESHOLD_TURNOVER_TARGET_RATIO"
    --objective-average-precision-weight "$OBJECTIVE_AVERAGE_PRECISION_WEIGHT"
    --objective-threshold-score-weight "$OBJECTIVE_THRESHOLD_SCORE_WEIGHT"
    --objective-brier-penalty-weight "$OBJECTIVE_BRIER_PENALTY_WEIGHT"
    --focal-alpha-min "$FOCAL_ALPHA_MIN"
    --focal-alpha-max "$FOCAL_ALPHA_MAX"
    --focal-gamma-min "$FOCAL_GAMMA_MIN"
    --focal-gamma-max "$FOCAL_GAMMA_MAX"
    --hard-negative-radius-min "$HARD_NEGATIVE_RADIUS_MIN"
    --hard-negative-radius-max "$HARD_NEGATIVE_RADIUS_MAX"
    --hard-negative-multiplier-min "$HARD_NEGATIVE_MULTIPLIER_MIN"
    --hard-negative-multiplier-max "$HARD_NEGATIVE_MULTIPLIER_MAX"
    --torch-warmup-epochs "$TORCH_WARMUP_EPOCHS"
    --torch-main-epochs "$TORCH_MAIN_EPOCHS"
    --torch-fine-epochs "$TORCH_FINE_EPOCHS"
    --torch-tail-epochs "$TORCH_TAIL_EPOCHS"
    --torch-fine-lr-scale "$TORCH_FINE_LR_SCALE"
    --torch-tail-lr-scale "$TORCH_TAIL_LR_SCALE"
    --torch-num-workers "$TORCH_NUM_WORKERS"
    --torch-eval-num-workers "$TORCH_EVAL_NUM_WORKERS"
    --torch-prefetch-factor "$TORCH_PREFETCH_FACTOR"
    --seed 42
  )

  if [[ "$PRELOAD_TO_DEVICE" == "1" ]]; then
    cmd+=(--torch-preload-to-device)
  fi
  if [[ "$ALLOW_TF32" == "1" ]]; then
    cmd+=(--torch-allow-tf32)
  fi
  if [[ "$CUDNN_BENCHMARK" == "1" ]]; then
    cmd+=(--torch-cudnn-benchmark)
  fi
  if [[ "$USE_AMP" == "1" ]]; then
    cmd+=(--use-amp)
  else
    cmd+=(--no-use-amp)
  fi
  if [[ "$TORCH_PERSISTENT_WORKERS" == "1" ]]; then
    cmd+=(--torch-persistent-workers)
  else
    cmd+=(--no-torch-persistent-workers)
  fi

  printf "Training command:\n%s\n\n" "${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$log_file"

  echo
  echo "Training complete."
  echo "Artifacts: $WORKDIR/$OUTPUT_ROOT_REL"
  echo "Log file: $WORKDIR/$log_file"
}

main() {
  if [[ ! -f "$CODE_ARCHIVE" ]]; then
    echo "Code archive not found: $CODE_ARCHIVE" >&2
    exit 1
  fi
  if [[ ! -f "$PREPARED_ARCHIVE" ]]; then
    echo "Prepared archive not found: $PREPARED_ARCHIVE" >&2
    exit 1
  fi

  resolve_profile
  apply_profile_defaults
  maybe_install_prereqs
  prepare_workspace
  extract_prepared_data
  print_runtime_summary
  run_training
}

main "$@"
