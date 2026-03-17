#!/bin/bash
# =============================================================================
# setup_env.sh  —  source this at the top of every SLURM script
# Leonardo @ CINECA
# =============================================================================

# ---- Modules ---------------------------------------------------------------
module purge
module load cuda/12.6
module load python/3.11.7

# ---- Python environment ----------------------------------------------------
source /leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/megatron/bin/activate

# ---- Internet proxy (required on compute nodes for HuggingFace, wandb, etc.)
full_hostname=$(hostname)
if [[ "$full_hostname" == *"lrdn"* ]]; then
    if ! [[ -d /tmp/tmux-29441 ]]; then
        mkdir /tmp/tmux-29441
    fi
    port=$(cat $HOME/scripts/.leonardo_port)
    echo "Waiting for reverse proxy on port ${port}..."
    while ! netstat -an | grep $port &> /dev/null; do sleep 1; done
    export HTTP_PROXY=socks5://127.0.0.1:$port
    export HTTPS_PROXY=socks5://127.0.0.1:$port
    export SOCK_PROXY=socks5://127.0.0.1:$port
    export ALL_PROXY=socks5://127.0.0.1:$port
    echo "Reverse proxy is up and running!"
fi

# ---- Paths -----------------------------------------------------------------
export MEGATRON_ROOT="${MEGATRON_ROOT:-/leonardo/home/userexternal/mciccone/exp/Megatron-LM}"
export DATA_PATH="${DATA_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/fineweb_text_document}"
export VOCAB_FILE="${VOCAB_FILE:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/gpt2-vocab.json}"
export MERGE_FILE="${MERGE_FILE:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/gpt2-merges.txt}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints}"
export LOG_PATH="${LOG_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs}"

# ---- NCCL / networking -----------------------------------------------------
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export GLOO_SOCKET_IFNAME=ib0      # InfiniBand on Leonardo
export MASTER_PORT="${MASTER_PORT:-29500}"
