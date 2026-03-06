#!/usr/bin/env bash
# SMoE GRPO with explicit async rollout mode.
# Uses vLLM async engine (AsyncLLM) and ROCm async flags (VLLM_ROCM_USE_AITER=1).
# Same setup as run_smoe_midtrain_phase2_decay_30036_math.sh but with rollout.mode=async
# set explicitly and experiment name / log file suffixed with _async for comparison.

set -x

# Paths: run script is inside verl/; workdir = parent of verl/ (repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Detect verl_workdir as parent of verl/ folder; override with export VERL_WORKDIR=/path if needed
VERL_WORKDIR="${VERL_WORKDIR:-$REPO_ROOT}"
DATA_ROOT="${VERL_WORKDIR}/tas/train/data"

# Install required packages if not already available
pip install tensordict wandb omegaconf torchdata codetiming
pip install hydra-core --upgrade

# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - CCA_Decode vLLM's
# memory pool is incompatible with it (AssertionError in device_allocator/cumem.py).

# Log to wandb offline to avoid BrokenPipeError in atexit when process exits (sync later with: wandb sync <run_dir>)
# export WANDB_MODE=offline

# Avoid "dubious ownership" git error when wandb/verl probe git root (e.g. in containers or different user)
git config --global --add safe.directory "$REPO_ROOT" 2>/dev/null || true
git config --global --add safe.directory "$SCRIPT_DIR" 2>/dev/null || true

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
# vLLM v1 async engine (optional; required for some async rollout backends)
# export VLLM_USE_V1=1
# ROCm async inference (aiter) for vLLM rollout
export VLLM_ROCM_USE_AITER=1
export USE_ROCM_AITER_ROPE_BACKEND=0

ray stop --force

# Download and preprocess GSM8k and MATH if parquet files are missing
mkdir -p "$DATA_ROOT/gsm8k" "$DATA_ROOT/math"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
if [[ ! -f "$DATA_ROOT/gsm8k/train.parquet" ]] || [[ ! -f "$DATA_ROOT/gsm8k/test.parquet" ]]; then
  echo "Preparing GSM8k dataset under $DATA_ROOT/gsm8k ..."
  python3 "$SCRIPT_DIR/examples/data_preprocess/gsm8k.py" --local_save_dir "$DATA_ROOT/gsm8k"
fi
if [[ ! -f "$DATA_ROOT/math/train.parquet" ]] || [[ ! -f "$DATA_ROOT/math/test.parquet" ]]; then
  echo "Preparing MATH dataset under $DATA_ROOT/math ..."
  python3 "$SCRIPT_DIR/examples/data_preprocess/math_dataset.py" --local_save_dir "$DATA_ROOT/math"
fi

gsm8k_train_path=$DATA_ROOT/gsm8k/train.parquet
gsm8k_test_path=$DATA_ROOT/gsm8k/test.parquet
math_train_path=$DATA_ROOT/math/train.parquet
math_test_path=$DATA_ROOT/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# Log file: grpo_smoe_..._async_<timestamp>.log
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/grpo_smoe_midtrain_phase2_decay_30036_async_gsm8k_math_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Zyphra-staging/smoe-midtrain_phase2_decay-30036 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='smoe_midtrain_phase2_decay_30036_async' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_training_steps=20 \
    trainer.total_epochs=16 "$@" 2>&1 | tee "$LOG_FILE"
