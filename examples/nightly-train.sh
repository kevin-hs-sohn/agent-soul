#!/usr/bin/env bash
# nightly-train.sh — Example nightly Soul training pipeline.
#
# Adapt this to your infrastructure. The general pattern:
#   1. Collect session logs → 2. Preprocess → 3. Train on GPU →
#   4. Convert to GGUF → 5. Deploy → 6. Health check
#
# Cron example (run at 3 AM daily):
#   0 3 * * * /path/to/nightly-train.sh >> /var/log/soul-train.log 2>&1

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — customize these for your setup
# ---------------------------------------------------------------------------
SOUL_DIR="/opt/soul"                    # Base directory
SESSION_DIR="/path/to/agent/sessions"   # Your agent's session log directory
TRAINING_DATA="${SOUL_DIR}/training_data"
ADAPTER_DIR="${SOUL_DIR}/adapters"
INFERENCE_SERVICE="soul-inference"       # systemd service name
HEALTH_URL="http://127.0.0.1:18791/health"

# GPU server (if using remote GPU like RunPod/Lambda)
GPU_HOST="your-gpu-server"
GPU_SSH_KEY="${HOME}/.ssh/gpu_key"
GPU_WORKSPACE="/workspace"

TODAY="$(date +%Y%m%d)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Step 1: Preprocess session logs
# ---------------------------------------------------------------------------
log "Preprocessing session logs..."

mkdir -p "${TRAINING_DATA}"
TRAIN_FILE="${TRAINING_DATA}/train_${TODAY}.jsonl"

python3 "${SOUL_DIR}/preprocess.py" \
  --input "${SESSION_DIR}" \
  --output "${TRAIN_FILE}"

DOC_COUNT="$(wc -l < "${TRAIN_FILE}")"
if [[ "${DOC_COUNT}" -eq 0 ]]; then
  log "No new training data. Skipping."
  exit 0
fi
log "Preprocessed ${DOC_COUNT} documents."

# ---------------------------------------------------------------------------
# Step 2: Train on GPU
# ---------------------------------------------------------------------------
log "Uploading training data to GPU server..."
scp -i "${GPU_SSH_KEY}" "${TRAIN_FILE}" "${GPU_HOST}:${GPU_WORKSPACE}/train.jsonl"

# Find previous adapter for incremental training
PREV_ADAPTER=""
if ssh -i "${GPU_SSH_KEY}" "${GPU_HOST}" "test -d ${GPU_WORKSPACE}/adapter"; then
  PREV_ADAPTER="--base-adapter ${GPU_WORKSPACE}/adapter"
fi

log "Starting training..."
ssh -i "${GPU_SSH_KEY}" "${GPU_HOST}" "cd ${GPU_WORKSPACE} && \
  python3 train.py \
    --data train.jsonl \
    ${PREV_ADAPTER} \
    --output ./output \
    --epochs 1"

# ---------------------------------------------------------------------------
# Step 3: Convert to GGUF
# ---------------------------------------------------------------------------
log "Converting adapter to GGUF..."
ssh -i "${GPU_SSH_KEY}" "${GPU_HOST}" "cd ${GPU_WORKSPACE} && \
  bash convert_adapter.sh ./output/adapter ./output/soul-adapter.gguf"

# ---------------------------------------------------------------------------
# Step 4: Deploy adapter
# ---------------------------------------------------------------------------
log "Downloading adapter..."
ADAPTER_FILE="${ADAPTER_DIR}/soul_${TODAY}.gguf"
mkdir -p "${ADAPTER_DIR}"
scp -i "${GPU_SSH_KEY}" "${GPU_HOST}:${GPU_WORKSPACE}/output/soul-adapter.gguf" "${ADAPTER_FILE}"

# Keep previous adapter for rollback
PREV_LINK="$(readlink -f "${ADAPTER_DIR}/current.gguf" 2>/dev/null || true)"

# Update symlink
ln -sf "${ADAPTER_FILE}" "${ADAPTER_DIR}/current.gguf"
log "Adapter deployed: ${ADAPTER_FILE}"

# ---------------------------------------------------------------------------
# Step 5: Restart inference + health check
# ---------------------------------------------------------------------------
log "Restarting inference service..."
systemctl restart "${INFERENCE_SERVICE}"

sleep 5

HEALTHY=false
for i in $(seq 1 10); do
  if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
    HEALTHY=true
    break
  fi
  log "Health check attempt ${i}/10..."
  sleep 3
done

if [[ "${HEALTHY}" == "true" ]]; then
  log "Inference healthy with new adapter."
else
  log "ERROR: Health check failed. Rolling back..."
  if [[ -n "${PREV_LINK}" && -f "${PREV_LINK}" ]]; then
    ln -sf "${PREV_LINK}" "${ADAPTER_DIR}/current.gguf"
    systemctl restart "${INFERENCE_SERVICE}"
    log "Rolled back to: ${PREV_LINK}"
  fi
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 6: Cleanup old adapters (keep last 5)
# ---------------------------------------------------------------------------
ls -1t "${ADAPTER_DIR}"/soul_*.gguf 2>/dev/null | tail -n +6 | xargs -r rm -f
log "Done. Training pipeline complete for ${TODAY}."
