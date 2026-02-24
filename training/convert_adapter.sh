#!/usr/bin/env bash
# convert_adapter.sh — Convert PEFT safetensors adapter to GGUF format.
#
# Converts a LoRA adapter (safetensors) to GGUF format for llama.cpp
# inference on the VPS. Intended to run on RunPod after training.
#
# Usage:
#   ./convert_adapter.sh <adapter_dir> <output_gguf>
#
# Example:
#   ./convert_adapter.sh ./output/adapter ./output/soul-adapter.gguf

set -euo pipefail

LLAMA_CPP_DIR="/workspace/llama.cpp"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# --- Argument validation ---

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <adapter_dir> <output_gguf>"
    echo ""
    echo "  adapter_dir   Path to PEFT adapter directory (contains adapter_model.safetensors)"
    echo "  output_gguf   Output path for GGUF file"
    exit 1
fi

ADAPTER_DIR="$1"
OUTPUT_GGUF="$2"

if [[ ! -d "${ADAPTER_DIR}" ]]; then
    log "ERROR: Adapter directory not found: ${ADAPTER_DIR}"
    exit 1
fi

if [[ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]] && [[ ! -f "${ADAPTER_DIR}/adapter_model.bin" ]]; then
    log "ERROR: No adapter_model.safetensors or adapter_model.bin found in ${ADAPTER_DIR}"
    exit 1
fi

# --- Clone llama.cpp if not present ---

if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
    log "Cloning llama.cpp to ${LLAMA_CPP_DIR}"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_DIR}"
    log "llama.cpp cloned"
else
    log "llama.cpp already present at ${LLAMA_CPP_DIR}"
fi

# --- Install Python requirements for converter ---

log "Installing Python requirements for GGUF converter"
pip install --break-system-packages -q -r "${LLAMA_CPP_DIR}/requirements.txt" 2>/dev/null || \
    pip install --break-system-packages -q numpy sentencepiece protobuf transformers gguf

# --- Ensure output directory exists ---
mkdir -p "$(dirname "${OUTPUT_GGUF}")"

# --- Convert adapter to GGUF ---

log "Converting adapter to GGUF"
log "  Adapter: ${ADAPTER_DIR}"
log "  Output:  ${OUTPUT_GGUF}"

python "${LLAMA_CPP_DIR}/convert_lora_to_gguf.py" \
    --outfile "${OUTPUT_GGUF}" \
    "${ADAPTER_DIR}"

# --- Report ---

if [[ -f "${OUTPUT_GGUF}" ]]; then
    FILE_SIZE=$(du -h "${OUTPUT_GGUF}" | cut -f1)
    log "Conversion complete"
    log "  Output: ${OUTPUT_GGUF}"
    log "  Size:   ${FILE_SIZE}"
else
    log "ERROR: Output file not created: ${OUTPUT_GGUF}"
    exit 1
fi
