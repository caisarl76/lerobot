# G1 inference server (Docker)

The VLA policy server for the Unitree G1 + SONIC pipeline, packaged so it runs on any NVIDIA GPU machine,
including one with no internet access. The robot side (the streamer on the G1's PC2) connects to it over
TCP and sends the camera image + robot state; the server answers with an action chunk.

```
G1 PC2: deploy + head camera + streamer  --tcp:5560-->  this server (GPU)
```

Source: `docker/g1_inference/` in https://github.com/caisarl76/lerobot (`main`). This README ships next to the
image files (handout folder on the workstation: `/mnt/data/jihun/g1_inference_image/`).

## What's in this folder

| File                                      | What it is                                                                                                                        |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `g1-inference-server-wbt-groot-he.tar.gz` | **Server with the model inside.** GR00T, handover-bottle task (HE init, 41 episodes; 5/5 on the robot). Runs offline. Start here. |
| `g1-inference-server.tar.gz`              | Server without a model: you mount any checkpoint (GR00T, Pi0.5, ACT, ...).                                                        |
| `g1-inference-client.tar.gz`              | Optional robot-side streamer. The PC2 normally runs the streamer from its own venv instead.                                       |
| `compose.yaml`                            | How to run them. Keep it next to where you run `docker compose`.                                                                  |
| `SHA256SUMS`                              | Checksums of the `.tar.gz` files.                                                                                                 |
| `hf_wbt_groot/`                           | Build input for the GR00T image. Not needed to run anything.                                                                      |

## Requirements

- Linux, NVIDIA GPU with ≥ 12 GB (GR00T uses ~10 GB), driver ≥ 570 (CUDA 12.8)
- Docker with the Compose plugin, and the NVIDIA Container Toolkit (`docker run --rm --gpus all ubuntu nvidia-smi` must work)
- ~40 GB free disk for the GR00T image

## 1. Load the image

```bash
sha256sum -c SHA256SUMS --ignore-missing         # copy arrived intact
docker load < g1-inference-server-wbt-groot-he.tar.gz
```

## 2. Start the GR00T server

```bash
GPU=0 PORT=5560 docker compose up -d --wait server-wbt-groot
docker compose ps                                  # Up (healthy)
docker compose logs server-wbt-groot | grep "ready on port"
```

- `GPU`: which GPU (`nvidia-smi` index). `PORT`: host port the robot connects to (default 5560).
- `--wait` returns once the model is loaded and warmed up (a few minutes: the checkpoint is 12 GB).
- No internet needed: the model and every file it loads are inside the image (`HF_HUB_OFFLINE=1`).
- The server log says `ready on port 5560` even if you chose another `PORT`: that is the port inside the container.

## 3. Check it answers

```bash
docker load < g1-inference-client.tar.gz          # once
docker compose run --rm -T --entrypoint python client -c "
import time, numpy as np
from sonic_policy_streamer import RemotePolicy
p = RemotePolicy('tcp://localhost:5560', 5.0)      # your PORT
k = p.image_keys[0]
for i in range(5):
    t = time.monotonic()
    c = p.chunk([np.zeros(28, np.float32)] * p.n_obs, [{k: np.zeros((480, 640, 3), np.uint8)}] * p.n_obs,
                'pick drink bottle from the table and handover')
    print('chunk', c.shape, 'finite', bool(np.isfinite(c).all()), f'{time.monotonic() - t:.2f} s')
"
```

Expect `(40, 78)` chunks, all finite. After the first call each chunk should take **well under 0.4 s**
(0.2 s on an RTX 3060). The robot asks for a new chunk every 0.4 s, so slower than that is too slow. Use a GPU
that is not also driving a desktop.

## 4. Robot side

The streamer on the PC2 connects to `tcp://<this server's IP>:<PORT>`. With the usual launcher
(`examples/g1_dex3_training/g1_groot_real_run.sh`), skip its `server` step and set the server's address:
`WS` in the script is the server IP, `PORT` the port. Settings for this model (session s03, 5/5):

```bash
PORT=5560 POLICY_FPS=50 MAX_S=30 TRIALS=5 ./g1_groot_real_run.sh session <session name> \
  "pick drink bottle from the table and handover" --hold-rest-s 1.5 --max-token-step 0.05
```

`POLICY_FPS=50` is required: this model was trained at 50 Hz.

## 5. Stop

```bash
docker compose down
```

## Other models (server without a model)

```bash
docker load < g1-inference-server.tar.gz
MODEL_DIR=/path/to/run_dir GPU=0 docker compose up -d --wait server
```

- `MODEL_DIR` is the run folder that contains `pretrained_model/`. Pi0.5: add `POLICY_SUBDIR=pretrained_model_ema`.
- The default server flags are GR00T's (`--backbone-dtype bfloat16 --noise-scale 0`). For any other policy add
  `SERVER_ARGS=` (empty).
- Base models / tokenizers come from `HF_CACHE` (default `~/.cache/huggingface`). Offline: fill that cache on a
  machine with internet first (start the server once there), copy it over, and add `HF_HUB_OFFLINE=1`.
- Weights live in the private Hugging Face repos `mncai/g1-...` (ask for access) or ask Jihun for a copy.

## Troubleshooting

| Symptom                                                           | Cause / fix                                                                |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `could not select device driver "nvidia"`                         | NVIDIA Container Toolkit missing                                           |
| `unhealthy`, log ends in `CUDA out of memory`                     | GPU busy: pick another `GPU`, or stop what uses it                         |
| `port is already allocated`                                       | another server on that port: other `PORT`, or `docker compose down`        |
| `OfflineModeIsEnabled` / `LocalEntryNotFoundError` (mount server) | `HF_CACHE` lacks a file the checkpoint needs: fill it online once          |
| robot: `no valid chunk for 1 s`                                   | chunks too slow (see step 3) or the robot can't reach `IP:PORT` (firewall) |
| `Permission denied` on mounted folders                            | the containers run as uid 1000; set `HOST_UID=$(id -u) HOST_GID=$(id -g)`  |
