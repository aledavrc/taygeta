#!/usr/bin/env bash
# ============================================================================
# Entrypoint of the local OCR service (llama.cpp `llama-server`).
#
# Exposes an OpenAI-compatible API on OCR_PORT (default 8080) and loads the
# multimodal model (text + visual projector `mmproj`) from the persistent
# ocr-models volume.
#
# If the weights do not exist in the volume, they are downloaded automatically
# from Hugging Face using the OCR_MODEL_URL / OCR_MMPROJ_URL URLs (configurable
# via the environment). It does not print credentials.
# ============================================================================
set -euo pipefail

# Shared library paths: system, local install and llama.cpp build tree
# (relevant for single-stage variants, e.g. ROCm, where the .so files remain
# in /src/build/bin). Avoids the "libllama-server-impl.so: cannot open shared
# object file" error.
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:/app/lib:/src/build/bin:${LD_LIBRARY_PATH:-}"

MODEL_PATH="${OCR_MODEL_PATH:-/models/model.gguf}"
MMPROJ_PATH="${OCR_MMPROJ_PATH:-}"
MODEL_URL="${OCR_MODEL_URL:-}"
MMPROJ_URL="${OCR_MMPROJ_URL:-}"
PORT="${OCR_PORT:-8080}"
CONTEXT="${OCR_CONTEXT_SIZE:-8192}"
PARALLEL="${OCR_CONCURRENCY:-1}"
THREADS="${OCR_THREADS:-$(nproc)}"
GPU_LAYERS="${OCR_GPU_LAYERS:-99}"
REQUIRE_GPU="${OCR_REQUIRE_GPU:-0}"
API_KEY="${OCR_API_KEY:-}"

# ROCm: an HSA_OVERRIDE_GFX_VERSION defined but EMPTY makes HIP detect no GPU
# and llama.cpp silently fall back to CPU. Treat empty as
# "no override" (native detection) to avoid that silent fallback.
if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
  unset HSA_OVERRIDE_GFX_VERSION
fi

# verify_gpu: with OCR_REQUIRE_GPU=1 it aborts if llama.cpp does not see a real
# GPU. It prevents the container from being marked healthy while running on CPU
# (or entering a restart loop without a clear message).
verify_gpu() {
  if [ "${REQUIRE_GPU}" != "1" ]; then
    return 0
  fi
  local devices
  devices="$(llama-server --list-devices 2>&1 || true)"
  # Anchors to the real device lines ("  ROCm0: ...", "  CUDA0: ...") to
  # avoid a false positive with error messages that mention "ROCm".
  if ! printf '%s\n' "${devices}" | grep -Eq '^[[:space:]]*(ROCm|CUDA)[0-9]+:'; then
    echo "ERROR: OCR_REQUIRE_GPU=1 but no GPU was detected." >&2
    printf '%s\n' "${devices}" >&2
    echo "Check the driver/ROCm, access to /dev/kfd and /dev/dri, and that" >&2
    echo "HSA_OVERRIDE_GFX_VERSION is not defined empty." >&2
    return 1
  fi
  echo "GPU detected:"
  printf '%s\n' "${devices}" | grep -E '^[[:space:]]*(ROCm|CUDA)[0-9]+:' | sed 's/^/  /'
}

download_file() {
  # download_file <url> <destination>
  local url="$1"
  local dest="$2"
  local partial="${dest}.part"
  mkdir -p "$(dirname "${dest}")"
  echo "Downloading: ${url}"
  echo "         ->  ${dest}"
  if command -v curl >/dev/null 2>&1; then
    local curl_args=(-fL --retry 3 --retry-delay 5 --connect-timeout 30 --progress-bar)
    # Resume a previous partial download if it exists.
    if [ -s "${partial}" ]; then
      curl_args+=(-C -)
    fi
    curl "${curl_args[@]}" -o "${partial}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    if [ -s "${partial}" ]; then
      wget -c -O "${partial}" "${url}"
    else
      wget -O "${partial}" "${url}"
    fi
  else
    echo "ERROR: 'curl' or 'wget' is required to download the weights." >&2
    return 1
  fi
  mv -f "${partial}" "${dest}"
}

ensure_file() {
  # ensure_file <path> <url> <description>
  local path="$1"
  local url="$2"
  local label="$3"
  if [ -f "${path}" ]; then
    echo "${label} already present at '${path}'; not downloading."
    return 0
  fi
  if [ -z "${url}" ]; then
    echo "ERROR: ${label} is missing at '${path}' and its URL was not defined." >&2
    echo "Define OCR_MODEL_URL / OCR_MMPROJ_URL or mount the weights in the 'ocr-models' volume." >&2
    return 1
  fi
  download_file "${url}" "${path}"
  if [ ! -f "${path}" ]; then
    echo "ERROR: the download of ${label} failed ('${url}')." >&2
    return 1
  fi
}

ensure_file "${MODEL_PATH}" "${MODEL_URL}" "the vision model" || exit 1
if [ -n "${MMPROJ_PATH}" ]; then
  ensure_file "${MMPROJ_PATH}" "${MMPROJ_URL}" "the visual projector (mmproj)" || exit 1
fi

# Verifies (if required) that a GPU is present before starting; it fails with a
# clear message instead of degrading to CPU.
verify_gpu || exit 1

args=(
  --host 0.0.0.0
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --ctx-size "${CONTEXT}"
  --n-gpu-layers "${GPU_LAYERS}"
  --image-min-tokens 1024
  --parallel "${PARALLEL}"
  --threads "${THREADS}"
)

if [ -n "${API_KEY}" ]; then
  args+=(--api-key "${API_KEY}")
fi

if [ -n "${MMPROJ_PATH}" ]; then
  args+=(--mmproj "${MMPROJ_PATH}")
fi

echo "Starting OCR: model=${MODEL_PATH} mmproj=${MMPROJ_PATH:-<none>} ctx=${CONTEXT} parallel=${PARALLEL} ngl=${GPU_LAYERS} port=${PORT} api_key=$([ -n "${API_KEY}" ] && echo 'yes' || echo 'no')"
exec llama-server "${args[@]}"
