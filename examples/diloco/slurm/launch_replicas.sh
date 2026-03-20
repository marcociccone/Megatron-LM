#!/bin/bash
# =============================================================================
# launch_replicas.sh — submit both DiLoCo replica jobs
#
# Usage:
#   bash launch_replicas.sh                          # new run, auto-named
#   bash launch_replicas.sh --name my_exp            # new run, custom name
#   bash launch_replicas.sh --resume llama3_1b_20260320_fsdp2  # resume run
#
# The run name becomes the checkpoint directory key:
#   ${BASE_CHECKPOINT_PATH}/runs/${RUN_NAME}/replica_0/
#   ${BASE_CHECKPOINT_PATH}/runs/${RUN_NAME}/replica_1/
#
# The lighthouse must already be running. Its address is auto-discovered
# from ${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt (written by lighthouse.sh).
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
            echo "Usage: bash launch_replicas.sh [--name NAME | --resume NAME]"
            exit 1 ;;
    esac
done

# Source env to get BASE_CHECKPOINT_PATH and LOG_PATH
source "${SCRIPT_DIR}/setup_env.sh" 2>/dev/null || true
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints}"

# ---- Resolve run name -------------------------------------------------------
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="llama3_1b_$(date +%Y%m%d_%H%M%S)"
    echo "Auto-generated run name: ${RUN_NAME}"
fi

RUNS_DIR="${BASE_CHECKPOINT_PATH}/runs"
RUN_CKPT_DIR="${RUNS_DIR}/${RUN_NAME}"

# ---- Validate resume/new ----------------------------------------------------
if [[ "${RESUME}" == true ]]; then
    if [[ ! -d "${RUN_CKPT_DIR}" ]]; then
        echo "ERROR: --resume requested but checkpoint dir not found: ${RUN_CKPT_DIR}"
        echo "Available runs:"
        ls "${RUNS_DIR}" 2>/dev/null || echo "  (none)"
        exit 1
    fi
    echo "Resuming run: ${RUN_NAME}"
    echo "  Checkpoints: ${RUN_CKPT_DIR}"
    for i in 0 1; do
        latest="${RUN_CKPT_DIR}/replica_${i}/latest_checkpointed_iteration.txt"
        if [[ -f "${latest}" ]]; then
            echo "  replica_${i}: iter $(cat ${latest})"
        fi
    done
else
    if [[ -d "${RUN_CKPT_DIR}" ]]; then
        echo "ERROR: Run '${RUN_NAME}' already exists at ${RUN_CKPT_DIR}"
        echo "Use --resume ${RUN_NAME} to continue it, or choose a different name."
        exit 1
    fi
    echo "New run: ${RUN_NAME}"
    echo "  Checkpoints: ${RUN_CKPT_DIR}"
    mkdir -p "${RUN_CKPT_DIR}/replica_0" "${RUN_CKPT_DIR}/replica_1"
fi

# ---- Check lighthouse -------------------------------------------------------
LIGHTHOUSE_ADDR_FILE="${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt"
if [[ ! -f "${LIGHTHOUSE_ADDR_FILE}" ]]; then
    echo "ERROR: Lighthouse address file not found: ${LIGHTHOUSE_ADDR_FILE}"
    echo "Run: sbatch ${SCRIPT_DIR}/lighthouse.sh"
    exit 1
fi
LIGHTHOUSE_ADDR=$(cat "${LIGHTHOUSE_ADDR_FILE}")

# Verify lighthouse is reachable (retry a few times in case it just started)
LIGHTHOUSE_HOST=$(echo "${LIGHTHOUSE_ADDR}" | sed 's|http://||' | cut -d: -f1)
LIGHTHOUSE_PORT=$(echo "${LIGHTHOUSE_ADDR}" | sed 's|http://||' | cut -d: -f2)
REACHABLE=false
for i in 1 2 3; do
    if timeout 5 bash -c "cat < /dev/null > /dev/tcp/${LIGHTHOUSE_HOST}/${LIGHTHOUSE_PORT}" 2>/dev/null; then
        REACHABLE=true; break
    fi
    [[ $i -lt 3 ]] && echo "  Lighthouse not reachable yet, retrying (${i}/3)..." && sleep 5
done
if [[ "${REACHABLE}" == false ]]; then
    echo "ERROR: Lighthouse at ${LIGHTHOUSE_ADDR} is not reachable."
    echo "  The address file may be stale. Start a new one with:"
    echo "  sbatch ${SCRIPT_DIR}/lighthouse.sh"
    exit 1
fi
echo "Lighthouse: ${LIGHTHOUSE_ADDR} (reachable)"

# ---- Submit replicas --------------------------------------------------------
echo ""
JOB_0=$(REPLICA_ID=0 \
    CHECKPOINT_OVERRIDE="${RUN_CKPT_DIR}/replica_0" \
    WANDB_RUN_NAME="${RUN_NAME}_replica_0" \
    sbatch "${SCRIPT_DIR}/replica.sh" | awk '{print $NF}')
echo "Submitted replica 0: job ${JOB_0}"

JOB_1=$(REPLICA_ID=1 \
    CHECKPOINT_OVERRIDE="${RUN_CKPT_DIR}/replica_1" \
    WANDB_RUN_NAME="${RUN_NAME}_replica_1" \
    sbatch "${SCRIPT_DIR}/replica.sh" | awk '{print $NF}')
echo "Submitted replica 1: job ${JOB_1}"

echo ""
echo "================================================"
echo "Run:        ${RUN_NAME}"
echo "Jobs:       ${JOB_0} (replica 0)  ${JOB_1} (replica 1)"
echo "Checkpoints: ${RUN_CKPT_DIR}/replica_{0,1}/"
echo "Logs:"
echo "  ${LOG_PATH:-...}/diloco_replica_0/${JOB_0}/train.log"
echo "  ${LOG_PATH:-...}/diloco_replica_1/${JOB_1}/train.log"
echo "Resume with:"
echo "  bash launch_replicas.sh --resume ${RUN_NAME}"
echo "================================================"
