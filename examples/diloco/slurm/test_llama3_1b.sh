#!/bin/bash
#SBATCH --job-name=megatron-llama3-1b-test
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Llama 3 1B baseline — no DiLoCo, 2 nodes × 4 GPUs (DP=8).
# Settings mirror replica.sh exactly (FSDP2, same batch/seq/lr config) so
# throughput and loss can be compared directly against a single DiLoCo replica.
#
# Use launch_baseline.sh instead of calling sbatch directly:
#   bash launch_baseline.sh --name my_baseline
#   bash launch_baseline.sh --resume my_baseline
# =============================================================================

set -euo pipefail
SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

# ---- Cluster topology -------------------------------------------------------
NUM_NODES="${SLURM_NNODES:-1}"
NUM_GPUS_PER_NODE=4
RDZV_PORT=29500
head_node=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)

# ---- Model config (Llama 3 1B) — identical to replica.sh -------------------
NUM_LAYERS=16
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=8192
NUM_ATTN_HEADS=32
NUM_QUERY_GROUPS=8
SEQ_LEN=4096
MAX_POS_EMB=8192

# ---- Parallelism ------------------------------------------------------------
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE=$(( NUM_NODES * NUM_GPUS_PER_NODE / TP_SIZE / PP_SIZE ))

# ---- Training hyperparams — identical to replica.sh ------------------------
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
LR="${LR:-3e-4}"

# ---- W&B -------------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-megatron-diloco}"
WANDB_ENTITY="${WANDB_ENTITY:-mciccone}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-baseline_llama3_1b}"

# ---- Paths (set by launch_baseline.sh via CHECKPOINT_OVERRIDE) -------------
CHECKPOINT_PATH="${CHECKPOINT_OVERRIDE:-${BASE_CHECKPOINT_PATH}/baseline_llama3_1b}"
RUN_LOG_PATH="${LOG_PATH}/baseline/${SLURM_JOB_ID}"
mkdir -p "${CHECKPOINT_PATH}" "${RUN_LOG_PATH}"

# ---- Data blend (same as replica.sh) ----------------------------------------
DATA_BLEND=()
for f in "${DATA_DIR}"/fineweb_edu_10bt_*.bin; do
    DATA_BLEND+=("1" "${f%.bin}")
done
if [[ ${#DATA_BLEND[@]} -eq 0 ]]; then
    echo "ERROR: No .bin files found in ${DATA_DIR}. Run preprocess_fineweb_edu.sh first."
    exit 1
fi

echo "================================================"
echo "Llama 3 1B baseline (no DiLoCo)"
echo "  Head node:   ${head_node} (${head_node_ip}:${RDZV_PORT})"
echo "  Nodes:       ${NUM_NODES} x ${NUM_GPUS_PER_NODE} GPUs"
echo "  TP=${TP_SIZE} PP=${PP_SIZE} DP=${DP_SIZE}"
echo "  GBS=${GLOBAL_BATCH_SIZE}  MBS=${MICRO_BATCH_SIZE}  SEQ=${SEQ_LEN}"
echo "  Checkpoint:  ${CHECKPOINT_PATH}"
echo "  Log:         ${RUN_LOG_PATH}/train.log"
echo "================================================"

srun --kill-on-bad-exit=1 \
  torchrun \
    --nproc_per_node="${NUM_GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${head_node_ip}:${RDZV_PORT}" \
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
    --lr-warmup-iters     0 \
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
    --use-torch-fsdp2 \
    --no-gradient-accumulation-fusion \
    \
    --save          "${CHECKPOINT_PATH}" \
    --load          "${CHECKPOINT_PATH}" \
    --save-interval 100 \
    --use-dist-ckpt \
    --ckpt-format   torch_dist \
    \
    --log-throughput \
    --log-interval  10 \
    --eval-interval 1000 \
    --eval-iters    10 \
    --tensorboard-dir "${RUN_LOG_PATH}/tensorboard" \
    --wandb-project  "${WANDB_PROJECT}" \
    --wandb-exp-name "${WANDB_RUN_NAME}" \
    ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
    --wandb-save-dir "${RUN_LOG_PATH}/wandb" \
    \
    2>&1 | tee "${RUN_LOG_PATH}/train.log"
