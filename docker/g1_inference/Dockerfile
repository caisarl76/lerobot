# G1 SONIC inference pipeline images (wired together in compose.yaml next to this file):
#   server: examples/g1_dex3_training/sonic_policy_server.py with the VLA policies (GPU, CUDA 12.8 torch wheels)
#   client: sonic_policy_streamer.py alone (numpy, pyzmq, msgpack, OpenCV; no lerobot/torch), so it also builds for
#           the G1 PC2 (linux/arm64)
# Build from the repo root, e.g. docker build -f docker/g1_inference/Dockerfile --target client -t g1-inference-client .
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS client
RUN pip install --no-cache-dir numpy==2.2.6 pyzmq==27.2.0 msgpack opencv-python-headless==4.13.0.92
WORKDIR /app
# the files g1_groot_real_run.sh `sync` copies to the PC2
COPY examples/g1_dex3_training/sonic_policy_streamer.py examples/g1_dex3_training/stream_backends.py \
     examples/g1_dex3_training/eval_session.py examples/g1_dex3_training/sonic_targets.py \
     examples/g1_dex3_training/wbc_common.py examples/g1_dex3_training/sonic_token_stream.py ./
ENTRYPOINT ["python", "-u", "sonic_policy_streamer.py"]

FROM python:${PYTHON_VERSION}-slim AS server
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# HOME/HF_HOME: compose runs the server as the host user, who has no home in the image
ENV PATH=/lerobot/.venv/bin:$PATH HOME=/tmp HF_HOME=/hf UV_LINK_MODE=copy
# policies the G1 work serves (GR00T, Pi0.5, MolmoAct2, VLA-JEPA, FastWAM, Diffusion, X-VLA, LaWAM; ACT/Psi0/XR-1
# need nothing extra)
ARG EXTRAS="groot pi molmoact2 vla_jepa fastwam diffusion xvla lawam peft pyzmq-dep"
WORKDIR /lerobot
COPY setup.py pyproject.toml uv.lock README.md MANIFEST.in ./
RUN uv sync --locked --no-cache --no-install-project $(printf -- '--extra %s ' $EXTRAS)
COPY . .
# msgpack: the server's wire format, not a lerobot dependency
RUN uv sync --locked --no-cache $(printf -- '--extra %s ' $EXTRAS) && uv pip install --no-cache msgpack
WORKDIR /lerobot/examples/g1_dex3_training
ENTRYPOINT ["python", "-u", "sonic_policy_server.py"]

# server with one checkpoint baked in: no mounts, runs offline. Build contexts: `model` = the checkpoint's
# pretrained_model dir, `hf` = an HF cache holding only the Hub files loading it downloads (compose server-wbt-groot)
FROM server AS server-baked
# --chmod: checkpoints are often saved mode 600; the server runs as the host user, not root
COPY --from=model --chmod=755 . /opt/model
COPY --from=hf --chmod=755 hub /opt/hf/hub
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1
