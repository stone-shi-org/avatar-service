# avatar-service: Ditto (TensorRT) talking head over WebRTC.
# Models are NOT in the image: mount them read-only at /models (see scripts/download_models.sh),
# and a writable cache at /data (GPU-specific TensorRT engines + idle loops).
#
# torch 2.4.1: Ditto pins numpy==2.0.1, which breaks torch < 2.3 ("Numpy is not available").
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG DITTO_REPO=https://github.com/antgroup/ditto-talkinghead.git
ARG DITTO_REF=c3e47eee2e626500017a0556b470d6d4182f85e8
ARG GIT_SHA=unknown

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# build-essential: Ditto compiles a small Cython blend module (done below at build time).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Ditto's runtime dependencies (TensorRT 8.6.1 = the version its engines were exported for).
RUN pip install \
      tensorrt==8.6.1 "cuda-python<12.7" polygraphy colored \
      numpy==2.0.1 librosa==0.11.0 tqdm filetype imageio imageio-ffmpeg \
      opencv-python-headless==4.10.0.84 scikit-image cython
# TensorRT 8.6 is linked against cuDNN 8; torch 2.4 ships cuDNN 9. Different sonames, so both coexist.
RUN pip install --no-deps --target /opt/cudnn8 nvidia-cudnn-cu12==8.9.7.29
ENV LD_LIBRARY_PATH=/opt/cudnn8/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

RUN git clone --filter=blob:none "$DITTO_REPO" /opt/ditto \
 && git -C /opt/ditto checkout -q "$DITTO_REF" \
 && rm -rf /opt/ditto/.git /opt/ditto/example \
 && cd /opt/ditto && python -c "import core.utils.blend"   # pre-build the pyximport extension (~/.pyxbld)

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY avatar_service ./avatar_service
COPY scripts ./scripts
COPY tests ./tests
RUN chmod +x scripts/*.sh && echo "$GIT_SHA" > /app/version.txt

LABEL org.opencontainers.image.source="https://github.com/stone-shi-org/avatar-service" \
      org.opencontainers.image.revision="$GIT_SHA" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV AVATAR_MODELS_DIR=/models AVATAR_DATA_DIR=/data DITTO_DIR=/opt/ditto AVATAR_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('AVATAR_PORT','8000'), timeout=4)"
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
