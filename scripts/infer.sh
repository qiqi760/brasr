#!/bin/bash
# ============================================================================
# GLCLAP Single-File Inference Entry Script
# ============================================================================
# Usage:
#   ./scripts/infer.sh \
#       --checkpoint outputs/glclap/best_model.pt \
#       --audio /path/to/audio.wav \
#       --bias_list data/bias_lists/phonecall.txt \
#       [--model_config configs/model_config.yaml] \
#       [--threshold 0.5] \
#       [--top_k 10] \
#       [--device cuda]
# ============================================================================

set -e

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
MODEL_CONFIG="${PROJECT_ROOT}/configs/model_config.yaml"
THRESHOLD="0.5"
TOP_K="10"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --audio)
            AUDIO="$2"
            shift 2
            ;;
        --bias_list)
            BIAS_LIST="$2"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --checkpoint <path> --audio <path> --bias_list <path> [options]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH    Path to model checkpoint"
            echo "  --audio PATH         Path to audio file"
            echo "  --bias_list PATH     Path to bias-word list (one per line)"
            echo ""
            echo "Options:"
            echo "  --model_config PATH  Model config YAML (default: configs/model_config.yaml)"
            echo "  --threshold FLOAT    Similarity threshold (default: 0.5)"
            echo "  --top_k INT          Max bias words to return (default: 10)"
            echo "  --device DEVICE      torch device (default: cuda if available)"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

# Validate required args
if [[ -z "${CHECKPOINT}" || -z "${AUDIO}" || -z "${BIAS_LIST}" ]]; then
    echo "Error: --checkpoint, --audio, and --bias_list are required."
    echo "Run '$0 --help' for usage."
    exit 1
fi

# Build Python command
CMD="python ${PROJECT_ROOT}/python_scripts/infer.py"
CMD="${CMD} --checkpoint ${CHECKPOINT}"
CMD="${CMD} --audio ${AUDIO}"
CMD="${CMD} --bias_list ${BIAS_LIST}"
CMD="${CMD} --model_config ${MODEL_CONFIG}"
CMD="${CMD} --threshold ${THRESHOLD}"
CMD="${CMD} --top_k ${TOP_K}"

if [[ -n "${DEVICE}" ]]; then
    CMD="${CMD} --device ${DEVICE}"
fi

echo "============================================================================"
echo "  Audio    : ${AUDIO}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "============================================================================"
echo "Running: ${CMD}"
echo ""

eval "${CMD}"
