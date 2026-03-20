#!/bin/bash
# =============================================================================
# setup_env.sh  —  source this at the top of every SLURM script
# Leonardo @ CINECA
# =============================================================================

# ---- Modules ---------------------------------------------------------------
module purge
module load python/3.11.7
# Note: no cuda module — PyTorch cu126 bundles its own CUDA runtime.
# Loading a cuda module can cause NVML driver/library version mismatch
# if the node's GPU driver is older than the loaded toolkit.

# ---- Python environment ----------------------------------------------------
source /leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/megatron/bin/activate

# ---- Internet access via reverse SOCKS5 proxy (optional) -------------------
# The proxy is set up by SSH tunneling from your local machine to the login node,
# then forwarded to compute nodes. Port is stored in $HOME/scripts/.leonardo_port.
#
# If the proxy is available, use it for WandB/HF (HTTP/HTTPS only).
# IMPORTANT: Do NOT set ALL_PROXY or SOCK_PROXY — Gloo and torchft use raw TCP
# sockets that don't speak SOCKS5, and routing them through the tunnel breaks
# inter-replica communication. HTTP_PROXY/HTTPS_PROXY are safe because Gloo/NCCL
# ignore those (they're only read by HTTP clients like requests/curl).
_PROXY_PORT_FILE="$HOME/scripts/.leonardo_port"
_proxy_up=false
if [[ -f "${_PROXY_PORT_FILE}" ]]; then
    _PROXY_PORT=$(cat "${_PROXY_PORT_FILE}")
    # Wait up to 60s for the reverse tunnel to appear (the tunnel script polls
    # every 10s and creates the SSH reverse forward after the job becomes RUNNING).
    echo "[setup_env] Waiting for proxy on port ${_PROXY_PORT}..."
    for _i in $(seq 1 30); do
        if ss -tlnp 2>/dev/null | grep -q ":${_PROXY_PORT}" || \
           netstat -tlnp 2>/dev/null | grep -q ":${_PROXY_PORT}"; then
            _proxy_up=true; break
        fi
        sleep 2
    done
fi

if [[ "${_proxy_up}" == true ]]; then
    # Verify the SOCKS5 proxy actually works before enabling it.
    # curl with --socks5 tests the full chain (tunnel → login node → internet).
    if curl -sf --socks5 "127.0.0.1:${_PROXY_PORT}" --connect-timeout 5 \
            "https://huggingface.co" -o /dev/null 2>/dev/null; then
        export HTTP_PROXY="socks5://127.0.0.1:${_PROXY_PORT}"
        export HTTPS_PROXY="socks5://127.0.0.1:${_PROXY_PORT}"
        # Standard NO_PROXY: domain suffix (.leonardo.local) and localhost.
        # Note: gRPC is disabled from using any proxy via GRPC_PROXY_OVERRIDE.
        export NO_PROXY="localhost,127.0.0.1,.leonardo.local"
        export no_proxy="${NO_PROXY}"
        # Disable gRPC proxy — gRPC (used by torchft) does not reliably respect
        # NO_PROXY and would try to route internal lighthouse connections through
        # the SOCKS5 tunnel, breaking inter-replica communication.
        export GRPC_PROXY_OVERRIDE=""
        export WANDB_MODE=online
        export HF_DATASETS_OFFLINE=0
        export TRANSFORMERS_OFFLINE=0
        echo "[setup_env] Proxy active on port ${_PROXY_PORT} — WandB/HF online"
    else
        echo "[setup_env] Proxy port listening but SOCKS5 connection failed — running offline"
        echo "[setup_env] Check your local SOCKS5 server and SSH tunnel chain."
        export WANDB_MODE=offline
        export HF_DATASETS_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    fi
else
    export WANDB_MODE=offline
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[setup_env] No proxy — running offline"
fi

# ---- HuggingFace cache (pre-downloaded models on scratch) ------------------
export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/huggingface}"

# ---- Paths -----------------------------------------------------------------
export MEGATRON_ROOT="${MEGATRON_ROOT:-/leonardo/home/userexternal/mciccone/exp/Megatron-LM}"
export DATA_DIR="${DATA_DIR:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/fineweb_edu_10bt}"
export VOCAB_FILE="${VOCAB_FILE:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/gpt2-vocab.json}"
export MERGE_FILE="${MERGE_FILE:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/gpt2-merges.txt}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints}"
export LOG_PATH="${LOG_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs}"

# ---- torchft debug ----------------------------------------------------------
export RUST_LOG="${RUST_LOG:-info}"

# ---- NCCL / networking -----------------------------------------------------
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
# GLOO_SOCKET_IFNAME intentionally not set — let Gloo auto-detect the interface.
# torchft's ProcessGroupGloo delegates to PyTorch's C++ backend for interface selection.
export NCCL_IB_HCA=mlx5              # Mellanox HCA on Leonardo A100 nodes
export CUDA_DEVICE_MAX_CONNECTIONS=1024
export MASTER_PORT="${MASTER_PORT:-29500}"
