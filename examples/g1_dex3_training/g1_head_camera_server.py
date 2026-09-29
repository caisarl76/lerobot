"""Camera-only head-camera publisher for sonic_policy_streamer.py --images zmq, for the robot PC.

Sends the same messages as LeRobot's ImageServer (cameras/zmq/image_server.py): ZMQ PUB, JSON
{"timestamps": {name: t}, "images": {name: base64 JPEG}}, with the JPEG holding the RGB array (the streamer
decodes without colour conversion and gets RGB back). Unlike robots/unitree_g1/run_g1_server.py --camera it
starts no DDS bridge: that script releases the robot's motion mode and forwards rt/lowcmd, which must not run next
to NVIDIA's deploy. Needs only numpy, opencv-python-headless and pyzmq (no lerobot or torch).

Name the stream after the policy's camera key so the streamer uses it as is, e.g. a mono head camera for a
one-camera model trained on cam_left_high:
  python g1_head_camera_server.py --device /dev/video4 --name cam_left_high
--snapshot saves one frame as PNG and exits (check colour, exposure and framing against the training images).
"""

import argparse
import base64
import contextlib
import json
import threading
import time

import cv2
import zmq


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="/dev/video4", help="V4L2 device of the colour stream")
    p.add_argument("--name", default="cam_left_high", help="stream name = the policy's camera key suffix")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30, help="camera capture rate")
    p.add_argument(
        "--publish-fps",
        type=float,
        default=10,
        help="encode + send rate; the streamer takes one frame per replan (0.4 s), so 10 keeps frames <= 0.1 s old",
    )
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--quality", type=int, default=80, help="JPEG quality (ImageServer uses 80)")
    p.add_argument("--snapshot", help="save one frame (after --out-size) to this PNG path and exit")
    p.add_argument(
        "--out-size",
        help="WxH sent to the streamer, e.g. 640x480 (the model size) from a wider 1280x720 capture",
    )
    p.add_argument(
        "--fit",
        choices=["stretch", "pad"],
        default="stretch",
        help="--out-size with another aspect ratio: stretch to fill, or keep the aspect and pad with black",
    )
    a = p.parse_args()
    out = tuple(int(v) for v in a.out_size.lower().split("x")) if a.out_size else None

    def fit(bgr):
        if out is None or (bgr.shape[1], bgr.shape[0]) == out:
            return bgr
        if a.fit == "stretch":
            return cv2.resize(bgr, out, interpolation=cv2.INTER_AREA)
        scale = min(out[0] / bgr.shape[1], out[1] / bgr.shape[0])
        w, h = round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        top, left = (out[1] - h) // 2, (out[0] - w) // 2
        return cv2.copyMakeBorder(
            small, top, out[1] - h - top, left, out[0] - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit(f"cannot read frames from {a.device}")
    if frame.shape[:2] != (a.height, a.width):
        raise SystemExit(f"{a.device} gives {frame.shape[1]}x{frame.shape[0]}, not {a.width}x{a.height}")
    if a.snapshot:
        for _ in range(30):  # let auto exposure settle
            ok, frame = cap.read()
        frame = fit(frame)
        cv2.imwrite(a.snapshot, frame)  # imwrite takes BGR, as read
        print(f"saved {a.snapshot} ({frame.shape[1]}x{frame.shape[0]})")
        return

    latest, lock = [None, 0.0], threading.Lock()

    def capture():
        # Drain the camera at its own rate with grab() (no colour conversion) and retrieve() (the costly
        # YUYV -> BGR conversion) only the frames that get published.
        next_retrieve = 0.0
        while True:
            if not cap.grab():
                time.sleep(0.01)
                continue
            now = time.monotonic()
            if now < next_retrieve:
                continue
            ok, bgr = cap.retrieve()
            if ok:
                next_retrieve = now + 1 / a.publish_fps
                with lock:
                    latest[:] = [bgr, time.time()]

    threading.Thread(target=capture, daemon=True).start()
    sock = zmq.Context().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{a.port}")
    print(
        f"[camera] {a.device} {a.width}x{a.height}@{a.fps} as '{a.name}' on port {a.port}, "
        f"publishing at {a.publish_fps:g} Hz" + (f" as {out[0]}x{out[1]} ({a.fit})" if out else ""),
        flush=True,
    )
    sent, last = 0, 0.0
    while True:
        started = time.monotonic()
        with lock:
            bgr, stamp = latest
        if bgr is not None and stamp > last:
            rgb = cv2.cvtColor(fit(bgr), cv2.COLOR_BGR2RGB)
            _, jpg = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), a.quality])
            encoded = base64.b64encode(jpg).decode("ascii")
            with contextlib.suppress(zmq.Again):  # no subscriber / buffer full: drop
                sock.send_string(
                    json.dumps({"timestamps": {a.name: stamp}, "images": {a.name: encoded}}), zmq.NOBLOCK
                )
            last, sent = stamp, sent + 1
            if sent % int(a.publish_fps * 60) == 0:
                print(f"[camera] {sent} frames sent", flush=True)
        time.sleep(max(0.0, 1 / a.publish_fps - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
