#!/usr/bin/env bash
set -euo pipefail

# soul-server.sh — Start llama-server with Soul-specific settings
# Wraps llama-server for systemd (exec replaces shell process)
#
# Configure paths via environment variables or edit defaults below.

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# --- Configuration (customize these) ---
MODEL="${SOUL_MODEL:-/opt/soul/models/Qwen3-8B-Q4_K_M.gguf}"
ADAPTER="${SOUL_ADAPTER:-/opt/soul/adapters/current.gguf}"
HOST="${SOUL_HOST:-127.0.0.1}"
PORT="${SOUL_PORT:-18791}"
THREADS="${SOUL_THREADS:-6}"
CTX_SIZE="${SOUL_CTX_SIZE:-4096}"
BATCH_SIZE="${SOUL_BATCH_SIZE:-512}"
MAX_TOKENS="${SOUL_MAX_TOKENS:-512}"

# --- Validate model file ---
if [[ ! -f "${MODEL}" ]]; then
    log "ERROR: Model file not found: ${MODEL}"
    log "Download a GGUF model first. Recommended: Qwen3-8B-Q4_K_M.gguf"
    log "  huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q4_k_m.gguf --local-dir ."
    exit 1
fi

# --- Build command ---
LLAMA_ARGS=(
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --threads "${THREADS}"
    --ctx-size "${CTX_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --n-predict "${MAX_TOKENS}"
)

# --- Adapter (optional — cold start without it) ---
if [[ -f "${ADAPTER}" ]]; then
    log "Loading LoRA adapter: ${ADAPTER} -> $(readlink -f "${ADAPTER}" 2>/dev/null || echo "${ADAPTER}")"
    LLAMA_ARGS+=(--lora "${ADAPTER}")
else
    log "WARNING: No adapter found at ${ADAPTER} — running base model only (cold start)"
fi

log "Starting llama-server on ${HOST}:${PORT}"
log "Model: ${MODEL}"
log "Threads: ${THREADS}, Context: ${CTX_SIZE}, Batch: ${BATCH_SIZE}, MaxTokens: ${MAX_TOKENS}"

# exec replaces shell with llama-server process (proper for systemd)
exec llama-server "${LLAMA_ARGS[@]}"
