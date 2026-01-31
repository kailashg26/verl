MODEL_PATH="ibm-granite/granite-4.0-h-small"
train_files="./data/gsm8k/train.parquet"
test_files="./data/gsm8k/test.parquet"

# export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
GPUS_PER_NODE=8


TP_VALUE=4 #If deepseek, set TP_VALUE=4
INFERENCE_BATCH_SIZE=32 #If deepseek, set INFERENCE_BATCH_SIZE=32
GPU_MEMORY_UTILIZATION=0.4 #If deepseek, set GPU_MEMORY_UTILIZATION=0.4
#export RAY_memory_monitor_refresh_ms=0

python3 -m verl.trainer.main_ppo  \
    algorithm.adv_estimator=grpo \
	data.train_files=$train_files  \
	data.val_files=$test_files  \
	data.train_batch_size=1024 \
	data.max_prompt_length=1024 \
	data.max_response_length=512 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.actor.strategy="fsdp2" \
	actor_rollout_ref.actor.ppo_mini_batch_size=256 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.fsdp_config.param_offload=False \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_VALUE \
	actor_rollout_ref.rollout.name=vllm  \
	actor_rollout_ref.nccl_timeout=3600 \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.fsdp_config.param_offload=False \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.critic_warmup=0 \
	trainer.logger=['console','wandb'] \
	trainer.project_name='ppo_qwen_llm' \
	trainer.experiment_name='ppo_trainer/run_qwen2-7b.sh_default' \
	trainer.n_gpus_per_node=8 \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=10 \
	trainer.total_epochs=50 | tee log.txt
