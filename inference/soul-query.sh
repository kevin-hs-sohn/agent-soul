#!/usr/bin/env bash
set -euo pipefail

# soul-query.sh — Query the Soul for context-aware predictions
#
# The Soul is a small LoRA-tuned LLM that provides context from past
# conversations. Use this to get soul_output before your main LLM responds.
#
# Usage:
#   ./soul-query.sh "user message"
#   ./soul-query.sh --context "recent context" "user message"
#
# Configure via environment variables:
#   SOUL_API       - Soul inference endpoint (default: http://127.0.0.1:18791/v1/chat/completions)
#   SOUL_TIMEOUT   - Request timeout in seconds (default: 60)
#   SOUL_MAX_TOKENS - Max response tokens (default: 128)
#   SOUL_IDENTITY  - One-line identity for system prompt (default: "You are a helpful assistant.")
#   SOUL_SCRATCHPAD - Path to scratchpad file for recent context (optional)

FALLBACK=""
SOUL_API="${SOUL_API:-http://127.0.0.1:18791/v1/chat/completions}"
TIMEOUT="${SOUL_TIMEOUT:-60}"
MAX_TOKENS="${SOUL_MAX_TOKENS:-128}"
TEMPERATURE="${SOUL_TEMPERATURE:-0.7}"
IDENTITY="${SOUL_IDENTITY:-You are a helpful assistant. Continue naturally based on patterns you have learned.}"
SCRATCHPAD="${SOUL_SCRATCHPAD:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# --- Check dependencies ---
for dep in jq curl; do
    if ! command -v "${dep}" &>/dev/null; then
        log "ERROR: ${dep} is required but not installed."
        echo "${FALLBACK}"
        exit 0
    fi
done

# --- Build system prompt (lightweight — minimal tokens for fast inference) ---
build_system_prompt() {
    local prompt="/no_think
${IDENTITY}"

    # Inject last 5 lines of scratchpad for recent context (if configured)
    if [[ -n "${SCRATCHPAD}" && -f "${SCRATCHPAD}" ]]; then
        local recent=""
        recent="$(tail -5 "${SCRATCHPAD}" 2>/dev/null)"
        if [[ -n "${recent}" ]]; then
            prompt+=$'\n\nRecent context:\n'"${recent}"
        fi
    fi

    echo "${prompt}"
}

SYSTEM_PROMPT="$(build_system_prompt)"

# --- Parse arguments ---
CONTEXT=""
USER_MESSAGE=""

if [[ "${1:-}" == "--context" ]]; then
    CONTEXT="${2:-}"
    USER_MESSAGE="${3:-}"
else
    USER_MESSAGE="${1:-}"
fi

if [[ -z "${USER_MESSAGE}" ]]; then
    log "Usage: $0 [--context \"recent context\"] \"user message\""
    exit 1
fi

# --- Build user content ---
if [[ -n "${CONTEXT}" ]]; then
    USER_CONTENT="${CONTEXT}
User: ${USER_MESSAGE}"
else
    USER_CONTENT="User: ${USER_MESSAGE}"
fi

# --- Build JSON payload ---
PAYLOAD="$(jq -n \
    --arg system "${SYSTEM_PROMPT}" \
    --arg user "${USER_CONTENT}" \
    --argjson max_tokens "${MAX_TOKENS}" \
    --argjson temperature "${TEMPERATURE}" \
    '{
        model: "soul",
        messages: [
            { role: "system", content: $system },
            { role: "user", content: $user }
        ],
        max_tokens: $max_tokens,
        temperature: $temperature
    }'
)"

# --- Call Soul inference server ---
log "Querying Soul (timeout: ${TIMEOUT}s)"

RESPONSE=""
if RESPONSE="$(curl -s -f \
    --max-time "${TIMEOUT}" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD}" \
    "${SOUL_API}" 2>/dev/null)"; then

    SOUL_OUTPUT="$(echo "${RESPONSE}" | jq -r '.choices[0].message.content // empty' 2>/dev/null)"

    if [[ -n "${SOUL_OUTPUT}" ]]; then
        log "Soul output received (${#SOUL_OUTPUT} chars)"
        echo "${SOUL_OUTPUT}"
    else
        log "WARNING: Empty soul output — returning empty"
        echo "${FALLBACK}"
    fi
else
    log "ERROR: Soul server unreachable — returning empty"
    echo "${FALLBACK}"
fi
