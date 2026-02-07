#!/bin/bash

#SBATCH --job-name=verl-ray-on-slurm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=200G
#SBATCH --time=30-00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
##SBATCH --nodelist=useocpm2m-097-[008,038,039,041]
##SBATCH --nodelist=useocpm2m-097-[008,032]


# load necessary modules
### Run this setup
# [Cluster]: Use docker
# docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4


##########################################################################
###The following setting should be set in different project and cluster###
##########################################################################
CONTAINER_NAME="multinode_verl_training"
verl_workdir="${HOME}/verl"

### Cluster Network Setting
export NCCL_DEBUG=TRACE
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
# export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
##########################################################################

### For rocm and training script
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES

export HF_HOME="${HOME}/.cache/huggingface"
export HF_TOKEN="your_huggingface_token"

export TIKTOKEN_RS_CACHE_DIR="${HOME}/tiktoken"

# Build and launch the Docker container
srun bash -c "
    # Exit on any error
    set -e

    # Need to pull the docker first
    docker pull docker.io/tasimage/primus:verl-torch2.9-pr-7

    # Kill and remove any existing containers (clean slate before launch).
    # Ignore errors so we don't abort the setup if nothing is running or a kill fails.
    docker ps -q | xargs -r docker kill || true
    docker ps -aq | xargs -r docker rm || true

    # Checking network devices
    ibdev2netdev

    # Launch the docker
    docker run --rm -d \
    -e HYDRA_FULL_ERROR=1 \
    -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
    -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    -e NOSET_CUDA_VISIBLE_DEVICES=1 \
    -e NOSET_HIP_VISIBLE_DEVICES=1 \
    -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
    -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    -e NCCL_DEBUG=${NCCL_DEBUG} \
    -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES} \
    -e TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY} \
    -e NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE} \
    -e NCCL_IB_HCA=${NCCL_IB_HCA} \
    -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
    -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC} \
    -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \
    -e NCCL_PROTO=${NCCL_PROTO} \
    -e RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE} \
    -e TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} \
    -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM} \
    -e HF_HOME=${HF_HOME} \
    -e HF_TOKEN=${HF_TOKEN} \
    -e TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR} \
    --network host \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v \${HOME}:\${HOME} \
    -v \${HOME}/.ssh:/root/.ssh \
    --shm-size 128G \
    --name \"${CONTAINER_NAME}\" \
    docker.io/tasimage/primus:verl-torch2.9-pr-7 \
    tail -f /dev/null

    echo \"Container setup completed\"
"

### Ray launch the nodes before training

# Getting the node names
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
else
    head_node_ip=${ADDR[0]}
fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization

# Print out all env variables
printenv

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    docker exec "${CONTAINER_NAME}" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --dashboard-port=8266 \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}"
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Debug: Starting worker on node_i = ${node_i}"
    if [ -z "$node_i" ]; then
        echo "Error: Empty node name for worker $i"
        continue
    fi
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        docker exec "${CONTAINER_NAME}" \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
done
sleep 10


# Ray initlization test (See whether any error in the above execution)
echo "Testing Ray initialization in the slurm nodes..."
docker exec "${CONTAINER_NAME}" python3 -c '
import ray
try:
    ray.init(address="auto")
    print("\n=== Ray Cluster Status ===")
    print(f"Number of nodes: {len(ray.nodes())}")
    for node in ray.nodes():
        print("Node: {}, Status: {}".format(node["NodeManagerHostname"], node["Alive"]))
        # print(f"Node: {node}")
    ray.shutdown()
    print("Ray initialization successful!")
except Exception as e:
    print(f"Ray initialization failed: {str(e)}")
'
echo "=== Ray test completed ==="
######




# Run data preprocessing

# echo "Starting data preprocessing..."
# docker exec "${CONTAINER_NAME}" \
#     python3 "examples/data_preprocess/gsm8k.py" "--local_save_dir" "../data/gsm8k"

# echo "Starting data preprocessing..."
# docker exec "${CONTAINER_NAME}" \
#     python3 "examples/data_preprocess/math_dataset.py" "--local_dir" "../data/math"

train_files="${verl_workdir}/tas/train/data/gsm8k/train.parquet"
val_files="${verl_workdir}/tas/train/data/gsm8k/test.parquet"

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"

echo "Start to train..."

# docker exec "${CONTAINER_NAME}" \
#     python3 -c "import transformers; transformers.pipeline('text-generation', model='$MODEL_PATH')"

PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
    docker exec "${CONTAINER_NAME}" sh -c " \
    ls -la ${TIKTOKEN_RS_CACHE_DIR} && \
    python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=32768 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    2>&1 | tee log.txt "