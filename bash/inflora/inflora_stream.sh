#!/usr/bin/env bash
# InfLoRA continual learning over a stream of LIBERO tasks (lerobot_lsy/src/lerobot/scripts/inflora.py), with the
# same data, steps, batch size, optimizer and evaluation as the CLARE stream, but no adapter expansion and no
# routing: each task's LoRA update is merged into the base weights after the task.
#
# Usage, from the root of the repo:
#   SUITES="libero_goal" OUT_ROOT=./outputs/libero_goal/inflora bash bash/inflora/inflora_stream.sh [SEED ...]
#
# Settings (environment variables):
#   SUITES          suites learned in order, space separated (required)
#   OUT_ROOT        where each stage writes its output folder (required)
#   PRETRAIN_PATH   base policy (default: continuallearning/dit_flow_mt_libero_90_pretrain)
#   TARGET_MODULES  regex of the nn.Linear layers that get a LoRA branch (default: the layer CLARE adapts,
#                   .*velocity_net\.cond_proj)
#   RANK            LoRA rank per task (default 32)
#   ORTHOGONAL      true = InfLoRA; false = sequential LoRA without the projection (ablation). Default true
#   LAMB, LAME      DualGPM thresholds (default 0.95, 1.0, as the official InfLoRA configs)
#   STEPS           training steps per task (default 20000, as CLARE's adapter phase)
#   FEATURE_STEPS   batches used for the input covariance of each layer (default 200)
#   N_EVAL          evaluation rollouts per learned task after each stage (default 100)
#   BS_EVAL         parallel LIBERO envs during evaluation (default 10)
#   NUM_WORKERS     data loader workers (default 16)
#   TASK_IDS        task ids learned in each suite (default "0 1 2 3 4 5 6 7 8 9")
#   WANDB_ENABLE    true to log to Weights & Biases (default false)
#
# A stage whose inflora_state.pt already exists is skipped, so rerunning the same command resumes the stream.
# Each stage's output is also saved to $OUT_ROOT/logs/<run name>.log, in the format clare_metrics.py reads.
set -euo pipefail

: "${SUITES:?set SUITES, e.g. SUITES=libero_goal}"
: "${OUT_ROOT:?set OUT_ROOT, e.g. OUT_ROOT=./outputs/libero_goal/inflora}"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
SEEDS=("${@:-42}")

PRETRAIN_PATH="${PRETRAIN_PATH:-continuallearning/dit_flow_mt_libero_90_pretrain}"
TARGET_MODULES="${TARGET_MODULES:-.*velocity_net\.cond_proj}"
RANK="${RANK:-32}"
ORTHOGONAL="${ORTHOGONAL:-true}"
LAMB="${LAMB:-0.95}"
LAME="${LAME:-1.0}"
STEPS="${STEPS:-20000}"
FEATURE_STEPS="${FEATURE_STEPS:-200}"
LOG_STEPS=100
N_EVAL="${N_EVAL:-100}"
BS_EVAL="${BS_EVAL:-10}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a task_array <<< "$TASK_IDS"
N_SUITES=$(wc -w <<< "$SUITES")
TOTAL_TASKS=$(( ${#task_array[@]} * N_SUITES ))
METHOD_TAG=$([ "$ORTHOGONAL" = true ] && echo inflora || echo seqlora)

task_prefix() {   # suite -> task name prefix used by gym-libero, e.g. libero_goal -> Libero_Goal
  case "$1" in
    libero_10) echo Libero_10 ;;
    libero_goal) echo Libero_Goal ;;
    libero_spatial) echo Libero_Spatial ;;
    libero_object) echo Libero_Object ;;
    *) echo "unknown suite: $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$OUT_ROOT/logs"
for SEED in "${SEEDS[@]}"; do
  seen_tasks=""   # every task learned so far; each stage evaluates all of them
  prev_state=""   # stage k starts from the state saved at the end of stage k-1
  for suite in $SUITES; do
    for task_id in "${task_array[@]}"; do
      name="dit_flow_mt_cl_seed_${SEED}_${suite}_task_${task_id}_${METHOD_TAG}_rank_${RANK}"
      seen_tasks="${seen_tasks:+$seen_tasks,}$(task_prefix "$suite")_Task_${task_id}"
      state_file="$OUT_ROOT/$name/inflora_state.pt"

      state_args=()
      if [ -n "$prev_state" ]; then
        state_args=(--inflora_state_path="$prev_state")
      fi

      if [ -f "$state_file" ]; then
        echo "=== skip $name (state exists)"
      else
        echo "=== $name"
        rm -rf "${OUT_ROOT:?}/$name"   # leftovers of an interrupted stage
        python ./lerobot_lsy/src/lerobot/scripts/inflora.py \
          --seed="$SEED" \
          --job_name="$name" \
          --output_dir="$OUT_ROOT/$name" \
          --dataset.repo_id="continuallearning/${suite}_image_task_${task_id}" \
          --policy.path="$PRETRAIN_PATH" \
          --policy.push_to_hub=false \
          --batch_size=32 \
          --num_workers="$NUM_WORKERS" \
          --steps="$STEPS" \
          --env.type=libero \
          --env.benchmark="$suite" \
          --env.task="$seen_tasks" \
          --eval.batch_size="$BS_EVAL" \
          --eval.n_episodes="$N_EVAL" \
          --eval.max_episodes_rendered=4 \
          --eval_freq=0 \
          --log_freq="$LOG_STEPS" \
          --target_modules="$TARGET_MODULES" \
          --rank="$RANK" \
          --orthogonal="$ORTHOGONAL" \
          --lamb="$LAMB" \
          --lame="$LAME" \
          --total_tasks="$TOTAL_TASKS" \
          --feature_steps="$FEATURE_STEPS" \
          --feature_batch_size=32 \
          "${state_args[@]}" \
          --wandb.enable="${WANDB_ENABLE:-false}" 2>&1 | tee "$OUT_ROOT/logs/$name.log"
      fi
      prev_state="$state_file"
    done
  done
done
