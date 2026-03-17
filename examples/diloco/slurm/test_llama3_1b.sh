#!/bin/bash
#SBATCH --job-name=megatron-llama3-1b-test
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Llama 3 1B smoke test — mock data, 100 iterations, 4 GPUs (all DP).
# No real dataset or tokenizer needed.
#
# Submit:
#   sbatch examples/diloco/slurm/test_llama3_1b.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

NUM_NODES="${SLURM_NNODES:-1}"
NUM_GPUS_PER_NODE=4
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=29500

# Llama 3 1B architecture
NUM_LAYERS=16
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=8192
NUM_ATTN_HEADS=32
NUM_QUERY_GROUPS=8
SEQ_LEN=4096
MAX_POS_EMB=8192

# Training
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=16
TRAIN_ITERS=100
LR=3e-4

CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}/test_llama3_1b"
RUN_LOG_PATH="${LOG_PATH}/test_llama3_1b"
mkdir -p "${CHECKPOINT_PATH}" "${RUN_LOG_PATH}"

echo "================================================"
echo "Llama 3 1B smoke test — mock data, 100 iters"
echo "  Nodes: ${NUM_NODES} x ${NUM_GPUS_PER_NODE} GPUs (all DP)"
echo "================================================"

torchrun \
    --nproc_per_node="${NUM_GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${MEGATRON_ROOT}/pretrain_gpt.py" \
    \
    --use-mcore-models \
    --num-layers          "${NUM_LAYERS}" \
    --hidden-size         "${HIDDEN_SIZE}" \
    --ffn-hidden-size     "${FFN_HIDDEN_SIZE}" \
    --num-attention-heads "${NUM_ATTN_HEADS}" \
    --group-query-attention \
    --num-query-groups    "${NUM_QUERY_GROUPS}" \
    --seq-length          "${SEQ_LEN}" \
    --max-position-embeddings "${MAX_POS_EMB}" \
    --position-embedding-type rope \
    --rotary-base         500000 \
    --rotary-percent      1.0 \
    --normalization       RMSNorm \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --attention-dropout   0.0 \
    --hidden-dropout      0.0 \
    --no-position-embedding \
    \
    --tokenizer-type      NullTokenizer \
    --vocab-size          128256 \
    --mock-data \
    \
    --micro-batch-size    "${MICRO_BATCH_SIZE}" \
    --global-batch-size   "${GLOBAL_BATCH_SIZE}" \
    --train-iters         "${TRAIN_ITERS}" \
    --lr                  "${LR}" \
    --min-lr              1e-5 \
    --lr-decay-style      cosine \
    --lr-warmup-iters     10 \
    --weight-decay        0.1 \
    --clip-grad           1.0 \
    --bf16 \
    \
    --tensor-model-parallel-size   1 \
    --pipeline-model-parallel-size 1 \
    \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    \
    --save          "${CHECKPOINT_PATH}" \
    --save-interval 100 \
    --use-dist-ckpt \
    --ckpt-format   torch_dist \
    \
    --log-interval  10 \
    --eval-interval 50 \
    --eval-iters    5 \
    \
    2>&1 | tee "${RUN_LOG_PATH}/train.log"
