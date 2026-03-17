#!/bin/bash
#SBATCH --job-name=megatron-diloco
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j-replica${REPLICA_ID}.out
#SBATCH --error=%x-%j-replica${REPLICA_ID}.err

# =============================================================================
# Megatron DiLoCo replica — one job per datacenter/partition.
#
# Submit as:
#   REPLICA_ID=0 LIGHTHOUSE_ADDR=<host>:29510 sbatch replica.sh
#   REPLICA_ID=1 LIGHTHOUSE_ADDR=<host>:29510 sbatch replica.sh
#
# Or with auto-discovery (if lighthouse.sh wrote the address file):
#   REPLICA_ID=0 sbatch replica.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

# ---- Replica identity -------------------------------------------------------
REPLICA_ID="${REPLICA_ID:-0}"

# Auto-discover lighthouse if not set explicitly
if [[ -z "${LIGHTHOUSE_ADDR:-}" ]]; then
    LIGHTHOUSE_ADDR_FILE="${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt"
    if [[ -f "${LIGHTHOUSE_ADDR_FILE}" ]]; then
        LIGHTHOUSE_ADDR=$(cat "${LIGHTHOUSE_ADDR_FILE}")
        echo "Auto-discovered lighthouse at: ${LIGHTHOUSE_ADDR}"
    else
        echo "ERROR: LIGHTHOUSE_ADDR not set and ${LIGHTHOUSE_ADDR_FILE} not found."
        echo "Start lighthouse.sh first, or set LIGHTHOUSE_ADDR manually."
        exit 1
    fi
fi

# ---- Cluster topology -------------------------------------------------------
NUM_NODES="${SLURM_NNODES:-1}"
NUM_GPUS_PER_NODE=4
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
# Use different master port per replica to avoid collisions on shared clusters
MASTER_PORT=$(( 29500 + REPLICA_ID ))

# ---- Model config -----------------------------------------------------------
# Defaults: GPT-3 small (117M). Override via env vars.
MODEL_SIZE="${MODEL_SIZE:-small}"
case "${MODEL_SIZE}" in
  small)   NUM_LAYERS=12; HIDDEN_SIZE=768;  NUM_ATTN_HEADS=12 ;;
  medium)  NUM_LAYERS=24; HIDDEN_SIZE=1024; NUM_ATTN_HEADS=16 ;;
  large)   NUM_LAYERS=24; HIDDEN_SIZE=2048; NUM_ATTN_HEADS=16 ;;
  xl)      NUM_LAYERS=32; HIDDEN_SIZE=4096; NUM_ATTN_HEADS=32 ;;
  *) echo "Unknown MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

# ---- Parallelism ------------------------------------------------------------
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE=$(( NUM_NODES * NUM_GPUS_PER_NODE / TP_SIZE / PP_SIZE ))

# ---- Training hyperparams --------------------------------------------------
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
SEQ_LEN="${SEQ_LEN:-2048}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
LR="${LR:-3e-4}"

# ---- DiLoCo hyperparams ----------------------------------------------------
DILOCO_SYNC_EVERY="${DILOCO_SYNC_EVERY:-500}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.7}"
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
DILOCO_MIN_REPLICAS="${DILOCO_MIN_REPLICAS:-1}"
DILOCO_QUORUM_TIMEOUT="${DILOCO_QUORUM_TIMEOUT:-600}"

# ---- W&B config ------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-megatron-diloco}"
WANDB_ENTITY="${WANDB_ENTITY:-}"   # <EDIT: your team/org, or leave empty>
WANDB_RUN_NAME="${WANDB_RUN_NAME:-diloco_${MODEL_SIZE}_replica_${REPLICA_ID}}"

# ---- Paths -----------------------------------------------------------------
CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}/diloco_replica_${REPLICA_ID}"
RUN_LOG_PATH="${LOG_PATH}/diloco_replica_${REPLICA_ID}"
mkdir -p "${CHECKPOINT_PATH}" "${RUN_LOG_PATH}"

echo "================================================"
echo "DiLoCo Replica ${REPLICA_ID}"
echo "  Lighthouse:  ${LIGHTHOUSE_ADDR}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Nodes:       ${NUM_NODES} x ${NUM_GPUS_PER_NODE} GPUs"
echo "  Model:       ${MODEL_SIZE} (${NUM_LAYERS}L ${HIDDEN_SIZE}H)"
echo "  TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE}"
echo "  sync_every:  ${DILOCO_SYNC_EVERY}"
echo "  Checkpoint:  ${CHECKPOINT_PATH}"
echo "================================================"

torchrun \
    --nproc_per_node="${NUM_GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=3 \
    "${MEGATRON_ROOT}/pretrain_gpt.py" \
    \
    --num-layers       "${NUM_LAYERS}" \
    --hidden-size      "${HIDDEN_SIZE}" \
    --num-attention-heads "${NUM_ATTN_HEADS}" \
    --seq-length       "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --tokenizer-type   GPT2BPETokenizer \
    --vocab-file       "${VOCAB_FILE}" \
    --merge-file       "${MERGE_FILE}" \
    \
    --micro-batch-size  "${MICRO_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters       "${TRAIN_ITERS}" \
    --lr                "${LR}" \
    --min-lr            1e-5 \
    --lr-decay-style    cosine \
    --lr-warmup-iters   2000 \
    --weight-decay      0.1 \
    --clip-grad         1.0 \
    --bf16 \
    --attention-dropout 0.0 \
    --hidden-dropout    0.0 \
    \
    --tensor-model-parallel-size  "${TP_SIZE}" \
    --pipeline-model-parallel-size "${PP_SIZE}" \
    \
    --data-path "${DATA_PATH}" \
    --split     949,50,1 \
    \
    --save          "${CHECKPOINT_PATH}" \
    --load          "${CHECKPOINT_PATH}" \
    --save-interval 1000 \
    --use-dist-ckpt \
    --ckpt-format   torch_dist \
    \
    --log-interval  10 \
    --eval-interval 1000 \
    --eval-iters    10 \
    --wandb-project  "${WANDB_PROJECT}" \
    --wandb-exp-name "${WANDB_RUN_NAME}" \
    ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
    --wandb-save-dir "${RUN_LOG_PATH}/wandb" \
    \
    --diloco \
    --diloco-sync-every          "${DILOCO_SYNC_EVERY}" \
    --diloco-outer-lr            "${DILOCO_OUTER_LR}" \
    --diloco-outer-momentum      "${DILOCO_OUTER_MOMENTUM}" \
    --diloco-outer-nesterov \
    --diloco-backup-device       cpu \
    --diloco-pin-memory \
    --diloco-lighthouse-addr     "${LIGHTHOUSE_ADDR}" \
    --diloco-replica-id          "replica_${REPLICA_ID}" \
    --diloco-min-replica-size    "${DILOCO_MIN_REPLICAS}" \
    --diloco-quorum-timeout-sec  "${DILOCO_QUORUM_TIMEOUT}" \
    \
    2>&1 | tee "${RUN_LOG_PATH}/train.log"
