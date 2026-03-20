#!/bin/bash
#SBATCH --job-name=preprocess-fineweb-edu
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Preprocess FineWeb-edu for Megatron-LM training using datatrove.
#
# Tokenizes pre-downloaded parquet files directly to Megatron .bin format
# (no intermediate jsonl). Each task writes its own .bin file in parallel.
#
# Raw data:
#   /leonardo_work/IscrB_Decentro/mciccone/sl3/datasets/fw-edu/sample/10BT
#   /leonardo_work/IscrB_Decentro/mciccone/sl3/datasets/fw-edu/sample/100BT
#
# Submit:
#   sbatch examples/diloco/slurm/preprocess_fineweb_edu.sh
#
# Output:
#   ${OUTPUT_PATH}/*.bin  (one per task, read by Megatron as a blend)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="/leonardo/home/userexternal/mciccone/exp/Megatron-LM/examples/diloco/slurm"
source "${SCRIPT_DIR}/setup_env.sh"

# Change to 100BT to process the larger split
SPLIT="10BT"

RAW_DATA_PATH="/leonardo_work/IscrB_Decentro/mciccone/sl3/datasets/fw-edu/sample/${SPLIT}"
SPLIT_LOWER=$(echo "${SPLIT}" | tr '[:upper:]' '[:lower:]')

# Set NUM_TASKS to the number of parquet files in the raw data folder
NUM_TASKS=$(find "${RAW_DATA_PATH}" -name "*.parquet" | wc -l)
echo "Found ${NUM_TASKS} parquet files in ${RAW_DATA_PATH}"
OUTPUT_PATH="/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/fineweb_edu_${SPLIT_LOWER}"
TOKENIZER_MODEL="meta-llama/Llama-3.2-1B"
LOG_PATH="/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/datatrove_fineweb_edu_${SPLIT_LOWER}"

mkdir -p "${OUTPUT_PATH}" "${LOG_PATH}"

# ---- Install datatrove if needed --------------------------------------------
python -c "import datatrove" 2>/dev/null || pip install datatrove --quiet

# ---- Tokenize directly from parquet → Megatron .bin -------------------------
echo "=== Tokenizing FineWeb-edu ${SPLIT} with datatrove (${NUM_TASKS} tasks) ==="

# Write to a real file — datatrove uses multiprocess which needs to re-import main
TOKENIZE_SCRIPT="/tmp/tokenize_fineweb_${SPLIT_LOWER}.py"
cat > "${TOKENIZE_SCRIPT}" <<EOF
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import MegatronDocumentTokenizer

if __name__ == "__main__":
    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                "${RAW_DATA_PATH}",
                text_key="text",
                glob_pattern="**/*.parquet",
            ),
            MegatronDocumentTokenizer(
                output_folder="${OUTPUT_PATH}",
                tokenizer_name_or_path="${TOKENIZER_MODEL}",
                eos_token=None,
                save_filename="fineweb_edu_${SPLIT_LOWER}",
            ),
        ],
        tasks=${NUM_TASKS},
        logging_dir="${LOG_PATH}",
    )
    executor.run()
EOF

python3 "${TOKENIZE_SCRIPT}"

echo ""
echo "=== Done ==="
echo "Output files:"
ls -lh "${OUTPUT_PATH}/"
echo ""
echo "Use in training with (blend of all task files):"
echo "  Set DATA_DIR=${OUTPUT_PATH} in setup_env.sh (already the default)"
