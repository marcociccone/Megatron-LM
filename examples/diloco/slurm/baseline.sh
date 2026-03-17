#!/bin/bash
#SBATCH --job-name=megatron-baseline
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Baseline Megatron training — same config as replica.sh but without DiLoCo.
# Use this as the comparison run to evaluate DiLoCo convergence.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

NUM_NODES="${SLURM_NNODES:-1}"
NUM_GPUS_PER_NODE=4
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL_SIZE="${MODEL_SIZE:-small}"
case "${MODEL_SIZE}" in
  small)   NUM_LAYERS=12; HIDDEN_SIZE=768;  NUM_ATTN_HEADS=12 ;;
  medium)  NUM_LAYERS=24; HIDDEN_SIZE=1024; NUM_ATTN_HEADS=16 ;;
  large)   NUM_LAYERS=24; HIDDEN_SIZE=2048; NUM_ATTN_HEADS=16 ;;
  xl)      NUM_LAYERS=32; HIDDEN_SIZE=4096; NUM_ATTN_HEADS=32 ;;
esac

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
SEQ_LEN="${SEQ_LEN:-2048}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
LR="${LR:-3e-4}"

# ---- W&B config ------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-megatron-diloco}"
WANDB_ENTITY="${WANDB_ENTITY:-}"   # <EDIT: your team/org, or leave empty>
WANDB_RUN_NAME="${WANDB_RUN_NAME:-baseline_${MODEL_SIZE}}"

CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}/baseline"
RUN_LOG_PATH="${LOG_PATH}/baseline"
mkdir -p "${CHECKPOINT_PATH}" "${RUN_LOG_PATH}"

echo "================================================"
echo "Baseline run (no DiLoCo)"
echo "  Model: ${MODEL_SIZE}  TP=${TP_SIZE} PP=${PP_SIZE}"
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
    --tokenizer-type      GPT2BPETokenizer \
    --vocab-file          "${VOCAB_FILE}" \
    --merge-file          "${MERGE_FILE}" \
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
    --tensor-model-parallel-size   "${TP_SIZE}" \
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
    2>&1 | tee "${RUN_LOG_PATH}/train.log"
