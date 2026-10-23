"""GPU-specific TensorRT engines for Ditto.

Ditto ships "hardware-compatible Ampere+" engines. They run on any Ampere-or-newer GPU but are about 20 %
slower; on an RTX 3090 that is the difference between ~24 and ~30 fps. On first start we copy the shipped
engines into the writable data volume and rebuild the two per-frame hot spots (`decoder`, `warp_network`)
natively for the GPU we're on. The result is cached per (compute capability, TensorRT version).

The two build functions follow Ditto's own `scripts/cvt_onnx_to_trt.py` (Apache-2.0, antgroup/ditto-talkinghead)
with the hardware-compatibility level removed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time

log = logging.getLogger("avatar.engines")

SHIPPED = "ditto_trt_Ampere_Plus"
NATIVE = {"decoder": "decoder_fp16.engine", "warp_network": "warp_network_fp16.engine"}
MARKER = "engines.json"


def _gpu_key() -> tuple[str, str]:
    import tensorrt as trt
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm{major}{minor}-trt{trt.__version__}", torch.cuda.get_device_name(0)


def _build(onnx_path: str, engine_path: str, plugin: str | None) -> None:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    if plugin:
        builder.get_plugin_registry().load_library(plugin)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")
    config = builder.create_builder_config()
    config.builder_optimization_level = 5
    config.set_flag(trt.BuilderFlag.FP16)
    if plugin:
        # grid-sample plugin layers must stay fp32 (same as Ditto's converter)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)
        skip = {"SHAPE", "ASSERTION", "SHUFFLE", "IDENTITY", "CONSTANT", "CONCATENATION", "GATHER", "SLICE",
                "CONDITION", "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT", "FILL", "NON_ZERO", "ONE_HOT"}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if str(layer.type)[10:] not in skip and "GridSample" in layer.name:
                layer.precision = trt.float32
        config.plugins_to_serialize = [plugin]
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    tmp = engine_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, engine_path)


def _build_isolated(onnx_path: str, engine_path: str, plugin: str | None, attempts: int = 3) -> None:
    """Build in a fresh process, with retries.

    TensorRT's autotuner occasionally hits a CUDA "illegal memory access" in one tactic (seen on an RTX 3090
    with the grid-sample plugin). That poisons the CUDA context, so a retry has to be a new process.
    """
    for i in range(1, attempts + 1):
        cmd = [sys.executable, "-m", "avatar_service.engines", onnx_path, engine_path, plugin or ""]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and os.path.isfile(engine_path):
            return
        log.warning("engine build attempt %d/%d failed (rc=%s) for %s", i, attempts, rc, os.path.basename(onnx_path))
    raise RuntimeError(f"TensorRT build failed {attempts} times for {onnx_path}")


def ensure_engines(ditto_models: str, data_dir: str, native: bool = True) -> str:
    """Return a Ditto `data_root` with every engine; build the native ones if they're missing."""
    shipped = os.path.join(ditto_models, SHIPPED)
    if not os.path.isdir(shipped):
        raise FileNotFoundError(f"{shipped} missing - run scripts/download_models.sh into the model volume")
    if not native:
        return shipped
    key, gpu = _gpu_key()
    out = os.path.join(data_dir, "engines", key)
    marker = os.path.join(out, MARKER)
    if os.path.isfile(marker):
        log.info("using cached native engines %s (%s)", out, gpu)
        return out
    os.makedirs(out, exist_ok=True)
    onnx = os.path.join(ditto_models, "ditto_onnx")
    plugin = os.path.join(onnx, "libgrid_sample_3d_plugin.so")
    built = {}
    for name, engine in NATIVE.items():
        target, stamp = os.path.join(out, engine), os.path.join(out, engine + ".native")
        if os.path.isfile(stamp) and os.path.isfile(target):
            continue   # finished by an earlier (interrupted) start
        t = time.time()
        log.info("building native %s engine for %s (takes a few minutes, once per GPU type)", name, gpu)
        _build_isolated(os.path.join(onnx, f"{name}.onnx"), target, plugin if name == "warp_network" else None)
        open(stamp, "w").close()
        built[name] = round(time.time() - t, 1)
        log.info("built %s in %.0f s", engine, built[name])
    for fn in os.listdir(shipped):   # everything else: the shipped engines as they are
        if fn.endswith(".engine") and not os.path.exists(os.path.join(out, fn)):
            shutil.copy2(os.path.join(shipped, fn), os.path.join(out, fn))
    with open(marker, "w") as f:
        json.dump({"gpu": gpu, "key": key, "built_sec": built, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    return out


if __name__ == "__main__":  # python -m avatar_service.engines ONNX ENGINE [PLUGIN]  (used by _build_isolated)
    logging.basicConfig(level=logging.INFO)
    _build(sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "") or None)
