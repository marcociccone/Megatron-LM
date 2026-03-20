#!/bin/bash
#SBATCH --job-name=megatron-diloco-small
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j-replica${REPLICA_ID}.out
#SBATCH --error=%x-%j-replica${REPLICA_ID}.err

# =============================================================================
# Megatron DiLoCo replica — small model (~100M params) for correctness testing.
#
# Use this BEFORE the full 1.5B run to verify that:
#   - DiLoCo sync commits at steps 10, 20, …
#   - Fault-tolerance path works (kill one replica mid-run)
#   - No CUDA OOM (100M model: <3 GB/GPU for params + optimizer states)
#
# Submit as:
#   REPLICA_ID=0 LIGHTHOUSE_ADDR=<host>:29510 sbatch replica_small.sh
#   REPLICA_ID=1 LIGHTHOUSE_ADDR=<host>:29510 sbatch replica_small.sh
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
NUM_GPUS_PER_NODE=1
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=$(( 29500 + REPLICA_ID ))

# ---- Replicas & GBS ---------------------------------------------------------
N_REPLICAS="${N_REPLICAS:-2}"
TOTAL_GLOBAL_BATCH_SIZE="${TOTAL_GLOBAL_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE=$(( TOTAL_GLOBAL_BATCH_SIZE / N_REPLICAS ))

# ---- Model config (~100M params) --------------------------------------------
# 8 layers, hidden=1024, ffn=4096 → ~100M parameters
NUM_LAYERS=8
HIDDEN_SIZE=1024
FFN_HIDDEN_SIZE=4096
NUM_ATTN_HEADS=16
NUM_QUERY_GROUPS=8
SEQ_LEN=2048
MAX_POS_EMB=4096

# ---- Parallelism ------------------------------------------------------------
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE=$(( NUM_NODES * NUM_GPUS_PER_NODE / TP_SIZE / PP_SIZE ))

# ---- Training hyperparams --------------------------------------------------
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
LR="${LR:-3e-4}"

# ---- DiLoCo hyperparams ----------------------------------------------------
# Sync every 10 steps for fast first-sync validation
DILOCO_SYNC_EVERY="${DILOCO_SYNC_EVERY:-10}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.7}"
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
DILOCO_MIN_REPLICAS="${DILOCO_MIN_REPLICAS:-2}"
DILOCO_QUORUM_TIMEOUT="${DILOCO_QUORUM_TIMEOUT:-300}"
DILOCO_TIMEOUT="${DILOCO_TIMEOUT:-300}"

# ---- W&B config ------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-megatron-diloco-small}"
WANDB_ENTITY="${WANDB_ENTITY:-mciccone}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-diloco_small_replica_${REPLICA_ID}}"

# ---- Paths -----------------------------------------------------------------
# Checkpoints are shared across runs (resume from latest).
# Logs are per-job to avoid overwriting previous runs.
CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}/diloco_small_replica_${REPLICA_ID}"
RUN_LOG_PATH="${LOG_PATH}/diloco_small_replica_${REPLICA_ID}/${SLURM_JOB_ID}"
mkdir -p "${CHECKPOINT_PATH}" "${RUN_LOG_PATH}"

# ---- Data blend -------------------------------------------------------------
DATA_BLEND=()
for f in "${DATA_DIR}"/fineweb_edu_10bt_*.bin; do
    DATA_BLEND+=("1" "${f%.bin}")
done
if [[ ${#DATA_BLEND[@]} -eq 0 ]]; then
    echo "ERROR: No .bin files found in ${DATA_DIR}. Run preprocess_fineweb_edu.sh first."
    exit 1
fi

echo "================================================"
echo "DiLoCo Small Replica ${REPLICA_ID} / ${N_REPLICAS}"
echo "  Lighthouse:  ${LIGHTHOUSE_ADDR}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Nodes:       ${NUM_NODES} x ${NUM_GPUS_PER_NODE} GPUs"
echo "  Model:       ~100M (${NUM_LAYERS}L ${HIDDEN_SIZE}H)"
echo "  TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE}"
echo "  local_GBS:   ${GLOBAL_BATCH_SIZE}  (total_GBS=${TOTAL_GLOBAL_BATCH_SIZE})"
echo "  sync_every:  ${DILOCO_SYNC_EVERY}"
echo "  backup_device: cuda (GPU snapshots — safe at 100M scale)"
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
    --use-mcore-models \
    --transformer-impl transformer_engine \
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
    --no-position-embedding \
    --attention-dropout   0.0 \
    --hidden-dropout      0.0 \
    \
    --tokenizer-type      HuggingFaceTokenizer \
    --tokenizer-model     "meta-llama/Llama-3.2-1B" \
    \
    --micro-batch-size    "${MICRO_BATCH_SIZE}" \
    --global-batch-size   "${GLOBAL_BATCH_SIZE}" \
    --train-iters         "${TRAIN_ITERS}" \
    --lr                  "${LR}" \
    --min-lr              1e-5 \
    --lr-decay-style      cosine \
    --lr-warmup-iters     20 \
    --weight-decay        0.1 \
    --clip-grad           1.0 \
    --bf16 \
    \
    --tensor-model-parallel-size   "${TP_SIZE}" \
    --pipeline-model-parallel-size "${PP_SIZE}" \
    \
    --data-path "${DATA_BLEND[@]}" \
    --split     949,50,1 \
    \
    --save          "${CHECKPOINT_PATH}" \
    --load          "${CHECKPOINT_PATH}" \
    --save-interval 500 \
    --use-dist-ckpt \
    --ckpt-format   torch_dist \
    \
    --log-throughput \
    --log-interval  5 \
    --eval-interval 500 \
    --eval-iters    5 \
    --tensorboard-dir "${RUN_LOG_PATH}/tensorboard" \
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
    --diloco-backup-device       cuda \
    --diloco-pin-memory \
    --diloco-lighthouse-addr     "${LIGHTHOUSE_ADDR}" \
    --diloco-replica-id          "replica_${REPLICA_ID}" \
    --diloco-min-replica-size    "${DILOCO_MIN_REPLICAS}" \
    --diloco-timeout-sec         "${DILOCO_TIMEOUT}" \
    --diloco-quorum-timeout-sec  "${DILOCO_QUORUM_TIMEOUT}" \
    --diloco-num-replicas        "${N_REPLICAS}" \
    --diloco-replica-index       "${REPLICA_ID}" \
    \
    2>&1 | tee "${RUN_LOG_PATH}/train.log"
