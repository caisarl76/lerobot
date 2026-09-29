"""Watch the robot's head camera stream (g1_head_camera_server.py / LeRobot ImageServer) on a workstation.

Writes the frames as an MJPEG stream to stdout for ffplay, so it needs no OpenCV GUI build:
  python g1_head_camera_viewer.py --host 192.168.0.222 | ffplay -loglevel error -fflags nobuffer -f mjpeg -i -
The publisher JPEG-encodes the RGB array as if it were BGR (ImageServer's convention, which the streamer relies
on), so the viewer re-encodes each frame with the channels swapped back for correct colours. --save DIR also keeps
every received frame as PNG (what the policy sees, at the published size).
"""

import argparse
import base64
import contextlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.0.222", help="robot PC running the camera publisher")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--name", help="camera stream to show (default: the first one in each message)")
    p.add_argument("--save", type=Path, help="also save every frame as PNG in this directory")
    a = p.parse_args()

    sock = zmq.Context().socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)  # always the latest frame
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://{a.host}:{a.port}")
    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
    print(f"[viewer] subscribed to tcp://{a.host}:{a.port}", file=sys.stderr, flush=True)
    out, n, t0 = sys.stdout.buffer, 0, time.monotonic()
    while True:
        msg = json.loads(sock.recv_string())
        images = msg.get("images", {})
        name = a.name or next(iter(images), None)
        if name not in images:
            continue
        rgb = cv2.imdecode(np.frombuffer(base64.b64decode(images[name]), np.uint8), cv2.IMREAD_COLOR)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if a.save:
            cv2.imwrite(str(a.save / f"{name}_{msg['timestamps'][name]:.3f}.png"), bgr)
        out.write(cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes())
        out.flush()
        n += 1
        if n % 100 == 0:
            age = time.time() - msg["timestamps"][name]
            print(
                f"[viewer] {name} {bgr.shape[1]}x{bgr.shape[0]}, {n / (time.monotonic() - t0):.1f} Hz, "
                f"frame age {age * 1000:.0f} ms",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError, KeyboardInterrupt):  # the player window was closed, or Ctrl-C
        main()
