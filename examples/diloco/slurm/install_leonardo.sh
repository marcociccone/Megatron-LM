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
WHEELS_DIR="${MEGATRON_ROOT}/examples/diloco/wheels"

# ---- 1. Modules -------------------------------------------------------------
# Note: no cuda module — PyTorch bundles its own CUDA runtime.
# Loading cuda/12.6 causes NVML driver/library mismatch on nodes with older drivers.
module load python/3.11.7

# ---- 2. Create venv if needed -----------------------------------------------
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    python3 -m venv "${VENV_PATH}"
    echo "Created venv at ${VENV_PATH}"
fi
source "${VENV_PATH}/bin/activate"

# ---- 3. Upgrade pip + build tools ------------------------------------------
pip install --upgrade pip wheel setuptools --quiet

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

# ---- 9. Transformer Engine (needs GCC 12 + cuDNN + NCCL headers) ------------
module load gcc/12.2.0 cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2 nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
echo "Installing Transformer Engine (takes ~15-20 min, compiles CUDA kernels)..."
pip install --no-build-isolation transformer_engine[pytorch]
module purge

# ---- 10. NVIDIA Apex (needs cuda/12.6 to match PyTorch cu126) ---------------
# Uses a pre-built wheel if available (saves ~15 min recompile).
# To rebuild: delete the wheel from WHEELS_DIR and re-run this script.
mkdir -p "${WHEELS_DIR}"
APEX_WHEEL=$(ls "${WHEELS_DIR}"/apex-*.whl 2>/dev/null | head -1)
if [[ -n "${APEX_WHEEL}" ]]; then
    echo "Installing NVIDIA Apex from cached wheel: ${APEX_WHEEL}"
    pip install --no-build-isolation "${APEX_WHEEL}"
else
    module load cuda/12.6 gcc/12.2.0
    echo "Building NVIDIA Apex from source (takes ~15 min)..."
    pip install ninja
    APEX_DIR="/tmp/apex_build"
    if [[ ! -d "${APEX_DIR}" ]]; then
        git clone https://github.com/NVIDIA/apex.git "${APEX_DIR}"
    fi
    cd "${APEX_DIR}" && git pull
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install --no-build-isolation .
    # Save wheel for future installs
    echo "Saving Apex wheel to ${WHEELS_DIR}..."
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip wheel --no-build-isolation . -w "${WHEELS_DIR}"
    module purge
fi

# ---- 11. Build Megatron C++ helpers (optional) --------------------------------
echo "Building Megatron C++ dataset helpers..."
cd "${MEGATRON_ROOT}"
pip install pybind11 --quiet
pip install -e . --no-build-isolation --quiet || echo "C++ helpers build failed (non-fatal)"

# ---- Done -------------------------------------------------------------------
echo ""
echo "Installation complete. Verify with:"
echo "  source ${VENV_PATH}/bin/activate"
echo "  python -c 'import torch, megatron, torchft, transformer_engine, apex; print(torch.__version__)'"
