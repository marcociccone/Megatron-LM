#!/bin/bash
# =============================================================================
# launch_baseline.sh — submit a single Llama 3 1B baseline job (no DiLoCo)
#
# Usage:
#   bash launch_baseline.sh                          # new run, auto-named
#   bash launch_baseline.sh --name my_baseline       # new run, custom name
#   bash launch_baseline.sh --resume my_baseline     # resume existing run
#
# Checkpoint dirs:
#   ${BASE_CHECKPOINT_PATH}/runs/${RUN_NAME}/baseline/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Parse args -------------------------------------------------------------
RUN_NAME=""
RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            RUN_NAME="$2"; shift 2 ;;
        --resume)
            RESUME=true
            RUN_NAME="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash launch_baseline.sh [--name NAME | --resume NAME]"
            exit 1 ;;
    esac
done

# Source env to get BASE_CHECKPOINT_PATH and LOG_PATH
source "${SCRIPT_DIR}/setup_env.sh" 2>/dev/null || true
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints}"

# ---- Resolve run name -------------------------------------------------------
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="baseline_llama3_1b_$(date +%Y%m%d_%H%M%S)"
    echo "Auto-generated run name: ${RUN_NAME}"
fi

RUNS_DIR="${BASE_CHECKPOINT_PATH}/runs"
RUN_CKPT_DIR="${RUNS_DIR}/${RUN_NAME}/baseline"

# ---- Validate resume/new ----------------------------------------------------
if [[ "${RESUME}" == true ]]; then
    if [[ ! -d "${RUN_CKPT_DIR}" ]]; then
        echo "ERROR: --resume requested but checkpoint dir not found: ${RUN_CKPT_DIR}"
        echo "Available runs:"
        ls "${RUNS_DIR}" 2>/dev/null || echo "  (none)"
        exit 1
    fi
    echo "Resuming run: ${RUN_NAME}"
    latest="${RUN_CKPT_DIR}/latest_checkpointed_iteration.txt"
    [[ -f "${latest}" ]] && echo "  iter $(cat ${latest})"
else
    if [[ -d "${RUN_CKPT_DIR}" ]]; then
        echo "ERROR: Run '${RUN_NAME}' already exists at ${RUN_CKPT_DIR}"
        echo "Use --resume ${RUN_NAME} to continue it, or choose a different name."
        exit 1
    fi
    echo "New run: ${RUN_NAME}"
    mkdir -p "${RUN_CKPT_DIR}"
fi

# ---- Submit -----------------------------------------------------------------
JOB=$(CHECKPOINT_OVERRIDE="${RUN_CKPT_DIR}" \
      WANDB_RUN_NAME="${RUN_NAME}" \
      sbatch "${SCRIPT_DIR}/test_llama3_1b.sh" | awk '{print $NF}')

echo ""
echo "================================================"
echo "Run:        ${RUN_NAME}"
echo "Job:        ${JOB}"
echo "Checkpoint: ${RUN_CKPT_DIR}"
echo "Log:        ${LOG_PATH:-...}/baseline/${JOB}/train.log"
echo "Resume with:"
echo "  bash launch_baseline.sh --resume ${RUN_NAME}"
echo "================================================"
