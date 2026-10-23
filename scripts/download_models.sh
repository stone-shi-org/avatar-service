#!/usr/bin/env bash
# Download the Ditto model files the service needs into a host folder that is then mounted at /models.
#
#   scripts/download_models.sh /path/to/avatar-models
#
# Fetches only what the TensorRT online pipeline uses (~4.5 GB): ditto_cfg/, the shipped Ampere+ engines,
# and the two ONNX graphs (+ grid-sample plugin) that are rebuilt natively for your GPU on first start.
# Uses a local `hf` CLI if present, otherwise a throwaway python:3.11-slim container.
set -euo pipefail
DEST="${1:?usage: $0 <model-dir>}"
REPO=digital-avatar/ditto-talkinghead
REV="${DITTO_MODELS_REV:-e4a2f60328ee7c32af585ac4b3cce299e4c8e254}"
mkdir -p "$DEST/ditto" "$DEST/avatars"
INCLUDE=(LICENSE README.md "ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" "ditto_cfg/v0.4_hubert_cfg_trt.pkl"
         "ditto_trt_Ampere_Plus/*.engine" "ditto_onnx/decoder.onnx" "ditto_onnx/warp_network.onnx"
         "ditto_onnx/libgrid_sample_3d_plugin.so")
INC=(); for p in "${INCLUDE[@]}"; do INC+=(--include "$p"); done   # current hf CLI: one --include per pattern
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --revision "$REV" --local-dir "$DEST/ditto" "${INC[@]}"
else
  DEST_ABS="$(cd "$DEST" && pwd)"
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$DEST_ABS/ditto:/out" python:3.11-slim sh -c \
    'repo="$0"; rev="$1"; shift; pip -q install --user "huggingface_hub[hf_xet]>=0.34" >/dev/null 2>&1 && \
     /tmp/.local/bin/hf download "$repo" --revision "$rev" --local-dir /out "$@"' \
    "$REPO" "$REV" "${INC[@]}"
fi
rm -rf "$DEST/ditto/.cache"
echo "Ditto models in $DEST/ditto ($(du -sh "$DEST/ditto" | cut -f1)). Now add portraits to $DEST/avatars/ (see README)."
