#!/bin/sh
# Start the service after checking the mounted model volume.
set -eu
M="${AVATAR_MODELS_DIR:-/models}"
if [ "${AVATAR_FAKE_RENDERER:-}" != "1" ]; then
  for p in ditto/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl ditto/ditto_trt_Ampere_Plus ditto/ditto_onnx/decoder.onnx \
           ditto/ditto_onnx/warp_network.onnx ditto/ditto_onnx/libgrid_sample_3d_plugin.so; do
    if [ ! -e "$M/$p" ]; then
      echo "ERROR: $M/$p is missing. Fill the model volume first: scripts/download_models.sh <host-model-dir>" >&2
      exit 2
    fi
  done
  if ! find "$M/avatars" -maxdepth 1 \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | grep -q .; then
    echo "ERROR: no portraits in $M/avatars/ (see README 'Avatars')" >&2
    exit 2
  fi
fi
mkdir -p "${AVATAR_DATA_DIR:-/data}"
echo "avatar-service $(cat /app/version.txt 2>/dev/null || echo dev)"
exec python -m avatar_service "$@"
