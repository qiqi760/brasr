#!/bin/bash
# ============================================================================
# GLCLAP Training Entry Script
# ============================================================================
# Usage:
#   ./scripts/train.sh \
#       --task contrastive-learning \
#       --dataset libri-960 \
#       [--model_config configs/model_config.yaml] \
#       [--train_config configs/train_config.yaml] \
#       [--audio_root data/contrastive-learning/audio/] \
#       [--resume outputs/glclap/checkpoint_epoch010.pt] \
#       [--local_only] \
#       [--nproc_per_node 4]
# ============================================================================

set -e

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
MODEL_CONFIG="${PROJECT_ROOT}/configs/model_config.yaml"
TRAIN_CONFIG="${PROJECT_ROOT}/configs/train_config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --train_config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --audio_root)
            AUDIO_ROOT="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --local_only)
            LOCAL_ONLY=true
            shift
            ;;
        --nproc_per_node)
            NPROC="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --task <task_name> --dataset <dataset_name> [options]"
            echo ""
            echo "Required:"
            echo "  --task TASK          Task name, e.g. contrastive-learning"
            echo "  --dataset DATASET    Dataset name, e.g. libri-960"
            echo ""
            echo "Options:"
            echo "  --model_config PATH  Model config YAML (default: configs/model_config.yaml)"
            echo "  --train_config PATH  Training config YAML (default: configs/train_config.yaml)"
            echo "  --audio_root PATH    Audio root directory (default: data/\${task}/audio/)"
            echo "  --resume PATH        Checkpoint to resume from"
            echo "  --local_only         Train LCLAP (local-only ablation)"
            echo "  --nproc_per_node N   Number of GPUs for DDP (default: 1)"
  echo "  --device DEVICE      torch device (single-card only, default: cuda)"
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
if [[ -z "${TASK}" || -z "${DATASET}" ]]; then
    echo "Error: --task and --dataset are required."
    echo "Run '$0 --help' for usage."
    exit 1
fi

# Build Python command
CMD="python ${PROJECT_ROOT}/python_scripts/train.py"
CMD="${CMD} --task ${TASK}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --model_config ${MODEL_CONFIG}"
CMD="${CMD} --train_config ${TRAIN_CONFIG}"

if [[ -n "${AUDIO_ROOT}" ]]; then
    CMD="${CMD} --audio_root ${AUDIO_ROOT}"
fi

if [[ -n "${RESUME}" ]]; then
    CMD="${CMD} --resume ${RESUME}"
fi

if [[ "${LOCAL_ONLY}" == true ]]; then
    CMD="${CMD} --local_only"
fi

if [[ -n "${NPROC}" && "${NPROC}" -gt 1 ]]; then
    CMD="torchrun --nproc_per_node=${NPROC} ${PROJECT_ROOT}/python_scripts/train.py"
else
    CMD="python ${PROJECT_ROOT}/python_scripts/train.py"
fi

# Re-build the rest of the command
CMD="${CMD} --task ${TASK}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --model_config ${MODEL_CONFIG}"
CMD="${CMD} --train_config ${TRAIN_CONFIG}"

if [[ -n "${AUDIO_ROOT}" ]]; then
    CMD="${CMD} --audio_root ${AUDIO_ROOT}"
fi

if [[ -n "${RESUME}" ]]; then
    CMD="${CMD} --resume ${RESUME}"
fi

if [[ "${LOCAL_ONLY}" == true ]]; then
    CMD="${CMD} --local_only"
fi

if [[ -n "${DEVICE}" ]]; then
    CMD="${CMD} --device ${DEVICE}"
fi

echo "============================================================================"
echo "  Task           : ${TASK}"
echo "  Dataset        : ${DATASET}"
echo "  Manifest       : data/${TASK}/${DATASET}.jsonl"
if [[ -n "${NPROC}" ]]; then
    echo "  GPUs (DDP)     : ${NPROC}"
fi
echo "============================================================================"
echo "Running: ${CMD}"
echo ""

eval "${CMD}"
