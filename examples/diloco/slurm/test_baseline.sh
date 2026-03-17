#!/bin/bash
#SBATCH --job-name=megatron-test
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
# Quick smoke test — tiny model, mock data, 100 iterations.
# No real dataset needed. Use this to verify the stack works end-to-end.
#
# Submit:
#   sbatch examples/diloco/slurm/test_baseline.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

NUM_NODES="${SLURM_NNODES:-1}"
NUM_GPUS_PER_NODE=4
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=29500

# Tiny model for fast iteration
NUM_LAYERS=4
HIDDEN_SIZE=256
NUM_ATTN_HEADS=4
SEQ_LEN=512

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16
TRAIN_ITERS=100
LR=3e-4

CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}/test_baseline"
RUN_LOG_PATH="${LOG_PATH}/test_baseline"
mkdir -p "${CHECKPOINT_PATH}" "${RUN_LOG_PATH}"

echo "================================================"
echo "Smoke test — tiny GPT, mock data, 100 iters"
echo "  Nodes: ${NUM_NODES} x ${NUM_GPUS_PER_NODE} GPUs"
echo "================================================"

torchrun \
    --nproc_per_node="${NUM_GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${MEGATRON_ROOT}/pretrain_gpt.py" \
    \
    --num-layers          "${NUM_LAYERS}" \
    --hidden-size         "${HIDDEN_SIZE}" \
    --num-attention-heads "${NUM_ATTN_HEADS}" \
    --seq-length          "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --tokenizer-type      NullTokenizer \
    --vocab-size          32768 \
    --mock-data \
    \
    --micro-batch-size  "${MICRO_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters       "${TRAIN_ITERS}" \
    --lr                "${LR}" \
    --min-lr            1e-5 \
    --lr-decay-style    cosine \
    --lr-warmup-iters   10 \
    --weight-decay      0.1 \
    --clip-grad         1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    \
    --tensor-model-parallel-size   1 \
    --pipeline-model-parallel-size 1 \
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
