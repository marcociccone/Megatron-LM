#!/bin/bash
#SBATCH --job-name=torchft-lighthouse
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Lighthouse coordinator — start this BEFORE any replica jobs.
#
# After this job starts, get the hostname with:
#   squeue -j <JOB_ID> -o "%R"   or look at the .out log
# Then set LIGHTHOUSE_ADDR=<that-hostname>:29510 in replica.sh.
# =============================================================================

SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

LIGHTHOUSE_PORT="${LIGHTHOUSE_PORT:-29510}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"   # can proceed with 1 replica (safe default)

echo "================================================"
echo "Lighthouse starting on $(hostname):${LIGHTHOUSE_PORT}"
echo "Min replicas: ${MIN_REPLICAS}"
echo "================================================"

# Write hostname to a shared file so replica jobs can discover it automatically
LIGHTHOUSE_ADDR_FILE="${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt"
mkdir -p "${BASE_CHECKPOINT_PATH}"
echo "$(hostname):${LIGHTHOUSE_PORT}" > "${LIGHTHOUSE_ADDR_FILE}"
echo "Address written to: ${LIGHTHOUSE_ADDR_FILE}"

torchft_lighthouse \
    --bind "[::]:${LIGHTHOUSE_PORT}" \
    --min_replicas "${MIN_REPLICAS}" \
    --quorum_tick_ms 500 \
    --join_timeout_ms 30000
