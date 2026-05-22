#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "=== DualKV Setup ==="

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. CUDA 12.8 required."
    exit 1
fi

# Create venv
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/.venv"
fi
source "${SCRIPT_DIR}/.venv/bin/activate"

# PyTorch
echo "Installing PyTorch 2.9.0+cu128..."
pip install --quiet torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Flash Attention (with DualKV kernels)
echo "Installing Flash Attention (with DualKV)... this may take 10-20 minutes"
pip install --quiet ninja packaging wheel setuptools psutil

# Fetch CUTLASS at pinned commit (required by flash-attention build)
if [ ! -f "${SCRIPT_DIR}/flash-attention/csrc/cutlass/include/cutlass/cutlass.h" ]; then
    echo "  Fetching CUTLASS (pinned at 7127592)..."
    git clone https://github.com/NVIDIA/cutlass.git "${SCRIPT_DIR}/flash-attention/csrc/cutlass"
    git -C "${SCRIPT_DIR}/flash-attention/csrc/cutlass" checkout 7127592069c2fe01b041e174ba4345ef9b279671
fi

pip install --quiet --no-build-isolation -e "${SCRIPT_DIR}/flash-attention"

# veRL (with DualKV integration)
echo "Installing veRL (with DualKV)..."
pip install --quiet -e "${SCRIPT_DIR}/verl"

# Remaining dependencies
echo "Installing remaining dependencies..."
pip install --quiet vllm==0.12.0 ray==2.55.0 wandb pandas pyarrow

# Verify
echo ""
echo "=== Verifying installation ==="
python -c "from flash_attn import flash_attn_dualkv_varlen_func; print('  flash_attn DualKV: OK')"
python -c "import verl; print(f'  veRL: OK (v{verl.__version__})')"
python -c "import vllm; print(f'  vLLM: OK (v{vllm.__version__})')"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ${SCRIPT_DIR}/.venv/bin/activate"
echo "Then set WORKDIR and run experiments from experiments/"
