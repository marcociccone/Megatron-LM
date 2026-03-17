#!/bin/bash
#SBATCH --job-name=preprocess-fineweb-edu
#SBATCH -A IscrB_Decentro
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Preprocess FineWeb-edu for Megatron-LM training.
#
# Converts pre-downloaded parquet files → jsonl → Megatron .bin/.idx format
# using the Llama 3 tokenizer.
#
# Raw data already at:
#   /leonardo_work/IscrB_Decentro/mciccone/sl3/datasets/fw-edu/sample/10BT
#   /leonardo_work/IscrB_Decentro/mciccone/sl3/datasets/fw-edu/sample/100BT
#
# Submit:
#   sbatch examples/diloco/slurm/preprocess_fineweb_edu.sh
#
# Output:
#   ${DATA_PATH}/fineweb_edu_10bt_text_document.bin
#   ${DATA_PATH}/fineweb_edu_10bt_text_document.idx
# =============================================================================

set -euo pipefail

SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

# Change to 100BT to process the larger split
SPLIT="10BT"

RAW_DATA_PATH="/leonardo_work/IscrB_Decentro/mciccone/sl3/datasets/fw-edu/sample/${SPLIT}"
DATA_PATH="/leonardo_scratch/fast/IscrB_Decentro/mciccone/data"
TOKENIZER_MODEL="meta-llama/Llama-3.2-1B"  # used only for tokenizer, not weights

SPLIT_LOWER=$(echo "${SPLIT}" | tr '[:upper:]' '[:lower:]')
JSONL_FILE="${DATA_PATH}/fineweb_edu_${SPLIT_LOWER}.jsonl"
OUTPUT_PREFIX="${DATA_PATH}/fineweb_edu_${SPLIT_LOWER}"

mkdir -p "${DATA_PATH}"

# ---- 1. Parquet → JSONL -----------------------------------------------------
if [[ -f "${JSONL_FILE}" ]]; then
    echo "=== Skipping parquet→jsonl (${JSONL_FILE} already exists) ==="
else
    echo "=== Converting parquet to jsonl (${SPLIT}) ==="
    python3 - <<EOF
import pandas as pd, glob, json, os

raw = "${RAW_DATA_PATH}"
out = "${JSONL_FILE}"

files = sorted(glob.glob(f"{raw}/**/*.parquet", recursive=True))
print(f"Found {len(files)} parquet files")

with open(out, "w") as fout:
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["text"])
        for text in df["text"]:
            fout.write(json.dumps({"text": text}) + "\n")
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(files)} files")

print(f"Done. Written to {out}")
EOF
fi

# ---- 2. Tokenize with Megatron ----------------------------------------------
if [[ -f "${OUTPUT_PREFIX}_text_document.idx" ]]; then
    echo "=== Skipping tokenization (${OUTPUT_PREFIX}_text_document.idx already exists) ==="
else
    echo "=== Tokenizing with Megatron (Llama 3.2 1B tokenizer) ==="
    python "${MEGATRON_ROOT}/tools/preprocess_data.py" \
    --input         "${JSONL_FILE}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --append-eod \
    --workers        32 \
    --chunk-size     1000
fi

echo ""
echo "=== Done ==="
echo "Output files:"
ls -lh "${DATA_PATH}"/fineweb_edu_${SPLIT_LOWER}*
echo ""
echo "Use in training with:"
echo "  --data-path ${OUTPUT_PREFIX}_text_document"
