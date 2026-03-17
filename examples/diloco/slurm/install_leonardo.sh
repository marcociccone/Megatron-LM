#!/bin/bash
# =============================================================================
# install.sh — one-time setup for Megatron DiLoCo on Leonardo @ CINECA
#
# Run once from the login node:
#   bash examples/diloco/slurm/install.sh
# =============================================================================

set -euo pipefail

VENV_PATH="/leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/megatron"
MEGATRON_ROOT="/leonardo/home/userexternal/mciccone/exp/Megatron-LM"
TORCHFT_ROOT="/leonardo/home/userexternal/mciccone/exp/torchft"

# ---- 1. Modules -------------------------------------------------------------
module load cuda/12.6 python/3.11.7

# ---- 2. Create venv if needed -----------------------------------------------
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    python3 -m venv "${VENV_PATH}"
    echo "Created venv at ${VENV_PATH}"
fi
source "${VENV_PATH}/bin/activate"

# ---- 3. Upgrade pip ---------------------------------------------------------
pip install --upgrade pip --quiet

# ---- 4. PyTorch (CUDA 12.6) -------------------------------------------------
echo "Installing PyTorch..."
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126 \
    --quiet

# ---- 5. Megatron-LM + training deps ----------------------------------------
echo "Installing Megatron-LM..."
pip install -e "${MEGATRON_ROOT}[training]" --quiet

# ---- 6. Rust (required for torchft) ----------------------------------------
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y --quiet
fi
source "${HOME}/.cargo/env"

# ---- 7. protoc to ~/.local/bin ----------------------------------------------
if ! command -v protoc &>/dev/null; then
    echo "Installing protoc..."
    PB_VERSION="29.3"
    curl -Lo /tmp/protoc.zip \
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PB_VERSION}/protoc-${PB_VERSION}-linux-x86_64.zip"
    unzip -o /tmp/protoc.zip -d "${HOME}/.local" bin/protoc 'include/*' > /dev/null
    rm /tmp/protoc.zip
fi
export PATH="${HOME}/.local/bin:${PATH}"

# ---- 8. torchft from source -------------------------------------------------
if [[ ! -d "${TORCHFT_ROOT}" ]]; then
    echo "Cloning torchft..."
    git clone https://github.com/pytorch/torchft.git "${TORCHFT_ROOT}"
fi
echo "Installing torchft..."
pip install -e "${TORCHFT_ROOT}" --quiet

# ---- 9. Build Megatron C++ helpers (optional) --------------------------------
echo "Building Megatron C++ dataset helpers..."
cd "${MEGATRON_ROOT}"
pip install pybind11 --quiet
pip install -e . --no-build-isolation --quiet || echo "C++ helpers build failed (non-fatal)"

# ---- Done -------------------------------------------------------------------
echo ""
echo "Installation complete. Verify with:"
echo "  source ${VENV_PATH}/bin/activate"
echo "  python -c 'import torch, megatron, torchft; print(torch.__version__)'"
