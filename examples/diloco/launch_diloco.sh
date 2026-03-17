#!/bin/bash
# =============================================================================
# DiLoCo Multi-Replica Launch Script
# =============================================================================
#
# This script launches ONE Megatron replica group (one per datacenter/cluster).
# Run this script independently on each cluster. A shared Lighthouse server
# must be running before starting any replicas.
#
# Architecture:
#   Cluster 0 (DC 0):  bash launch_diloco.sh 0
#   Cluster 1 (DC 1):  bash launch_diloco.sh 1
#   Cluster 2 (DC 2):  bash launch_diloco.sh 2
#                               |
#                    Lighthouse (shared, accessible by all replicas)
#
# Usage:
#   # 1. Start the Lighthouse on a publicly reachable machine:
#   #    python -m torchft.lighthouse --bind [::]:29510
#   #
#   # 2. On each cluster, run this script with the replica index:
#   #    LIGHTHOUSE_ADDR=lighthouse-host:29510 bash launch_diloco.sh <REPLICA_ID>
#
# =============================================================================

set -euo pipefail

# ---- Required: set these per-cluster before running ----
REPLICA_ID="${1:-0}"                         # Unique index: 0, 1, 2, ...
LIGHTHOUSE_ADDR="${LIGHTHOUSE_ADDR:-localhost:29510}"

# ---- Intra-replica cluster config ----
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-$((29500 + REPLICA_ID))}"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"

# ---- Model config (GPT-3 small for demonstration) ----
MODEL_SIZE="${MODEL_SIZE:-small}"   # small | medium | large

case "${MODEL_SIZE}" in
  small)
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTN_HEADS=12
    ;;
  medium)
    NUM_LAYERS=24
    HIDDEN_SIZE=1024
    NUM_ATTN_HEADS=16
    ;;
  large)
    NUM_LAYERS=24
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    ;;
esac

# ---- Parallelism config ----
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE=$(( NUM_NODES * NUM_GPUS_PER_NODE / TP_SIZE / PP_SIZE ))

# ---- Training config ----
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
SEQ_LEN="${SEQ_LEN:-2048}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
LR="${LR:-3e-4}"

# ---- DiLoCo config ----
DILOCO_SYNC_EVERY="${DILOCO_SYNC_EVERY:-500}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.7}"
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
DILOCO_MIN_REPLICAS="${DILOCO_MIN_REPLICAS:-1}"
DILOCO_QUORUM_TIMEOUT="${DILOCO_QUORUM_TIMEOUT:-600}"  # seconds

# ---- Paths ----
VOCAB_FILE="${VOCAB_FILE:-/data/gpt2-vocab.json}"
MERGE_FILE="${MERGE_FILE:-/data/gpt2-merges.txt}"
DATA_PATH="${DATA_PATH:-/data/my_text_document}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/checkpoints/diloco_replica_${REPLICA_ID}}"
LOG_PATH="${LOG_PATH:-/logs/diloco_replica_${REPLICA_ID}}"

mkdir -p "${CHECKPOINT_PATH}" "${LOG_PATH}"

# ---- Distributed args ----
DISTRIBUTED_ARGS=(
    --nproc_per_node "${NUM_GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --max_restarts 3        # allow torchrun restarts on failure
)

# ---- GPT model args ----
MODEL_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTN_HEADS}"
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --tokenizer-type GPT2BPETokenizer
    --vocab-file "${VOCAB_FILE}"
    --merge-file "${MERGE_FILE}"
)

# ---- Training args ----
TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --lr "${LR}"
    --min-lr 1e-5
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --use-flash-attn
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

# ---- Parallelism args ----
PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
)

# ---- Data args ----
DATA_ARGS=(
    --data-path "${DATA_PATH}"
    --split 949,50,1
)

# ---- Checkpoint args ----
CKPT_ARGS=(
    --save "${CHECKPOINT_PATH}"
    --load "${CHECKPOINT_PATH}"
    --save-interval 1000
    --use-dist-ckpt
    --ckpt-format torch_dist
)

# ---- Logging args ----
LOG_ARGS=(
    --log-interval 10
    --eval-interval 1000
    --eval-iters 10
    --tensorboard-dir "${LOG_PATH}/tensorboard"
)

# ---- DiLoCo args ----
DILOCO_ARGS=(
    --diloco
    --diloco-sync-every "${DILOCO_SYNC_EVERY}"
    --diloco-outer-lr "${DILOCO_OUTER_LR}"
    --diloco-outer-momentum "${DILOCO_OUTER_MOMENTUM}"
    --diloco-outer-nesterov
    --diloco-backup-device cpu
    --diloco-pin-memory
    --diloco-lighthouse-addr "${LIGHTHOUSE_ADDR}"
    --diloco-replica-id "replica_${REPLICA_ID}"
    --diloco-min-replica-size "${DILOCO_MIN_REPLICAS}"
    --diloco-quorum-timeout-sec "${DILOCO_QUORUM_TIMEOUT}"
)

echo "=============================================="
echo "Starting DiLoCo replica ${REPLICA_ID}"
echo "  Lighthouse: ${LIGHTHOUSE_ADDR}"
echo "  Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Nodes:      ${NUM_NODES} x ${NUM_GPUS_PER_NODE} GPUs"
echo "  TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE}"
echo "  sync_every: ${DILOCO_SYNC_EVERY}"
echo "=============================================="

torchrun "${DISTRIBUTED_ARGS[@]}" \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${DILOCO_ARGS[@]}" \
    2>&1 | tee "${LOG_PATH}/replica_${REPLICA_ID}.log"
