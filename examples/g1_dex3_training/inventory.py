"""Bounded, metadata-only inventory for the Unitree G1 Dex3 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

SUFFIXES = (
    "ObjectPlacement",
    "GraspSquare",
    "CameraPackaging",
    "ToastedBread",
    "Pouring",
    "BlockStacking",
    "PickApple",
    "PickBottle",
    "PickCharger",
    "PickDoll",
    "PickGum",
    "PickSnack",
    "PickTissue",
)
REPO_IDS = tuple(f"unitreerobotics/G1_Dex3_{name}_Dataset" for name in SUFFIXES)
EXPECTED_JOINT_ORDER = (
    tuple(
        f"k{side}{joint}"
        for side in ("Left", "Right")
        for joint in (
            "ShoulderPitch",
            "ShoulderRoll",
            "ShoulderYaw",
            "Elbow",
            "WristRoll",
            "WristPitch",
            "WristYaw",
        )
    )
    + tuple(
        f"kLeftHand{joint}"
        for joint in ("Thumb0", "Thumb1", "Thumb2", "Middle0", "Middle1", "Index0", "Index1")
    )
    + tuple(
        f"kRightHand{joint}"
        for joint in ("Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Middle0", "Middle1")
    )
)


class InventoryError(ValueError):
    pass


def _flatten(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [str(value)]


def _features(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = info.get("features", info.get("features_info", {}))
    if not isinstance(features, dict):
        raise InventoryError("info.json features must be an object")
    return features


def _shape(feature: dict[str, Any]) -> list[int]:
    shape = feature.get("shape", feature.get("dtype_shape"))
    if not isinstance(shape, list):
        raise InventoryError("feature is missing shape")
    return [int(x) for x in shape]


def _metadata_hash(meta: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(p for p in meta.rglob("*") if p.is_file())
    for path in files:
        digest.update(path.relative_to(meta).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _one(
    root: Path, repo_id: str, canonical_fps: int | float | None, camera_keys: tuple[str, ...] | None = None
) -> tuple[dict[str, Any], int | float]:
    meta = root / "meta"
    info_path = meta / "info.json"
    if not root.is_dir() or not info_path.is_file():
        raise InventoryError(f"missing dataset or meta/info.json: {root}")
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"invalid {info_path}: {exc}") from exc
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise InventoryError(f"missing fps: {info_path}")
    if canonical_fps is not None and fps != canonical_fps:
        raise InventoryError(f"incompatible fps in {root}: {fps} != {canonical_fps}")
    features = deepcopy(_features(info))
    dropped_cameras = []
    if camera_keys is not None:
        for key in camera_keys:
            if key not in features or features[key].get("dtype") != "video":
                raise InventoryError(f"missing selected video camera {key}: {root}")
        dropped_cameras = [
            k for k, v in features.items() if v.get("dtype") == "video" and k not in camera_keys
        ]
        for key in dropped_cameras:
            del features[key]
        for key in camera_keys:
            value = features[key]
            if len(value["shape"]) == 3 and value["shape"][-1] == 3 and value["shape"][0] != 3:
                h, w, c = value["shape"]
                value["shape"] = [c, h, w]
            if len(value["shape"]) != 3 or value["shape"][0] != 3:
                raise InventoryError(f"unsupported RGB camera shape: {key} in {root}")
            value["names"] = ["channels", "height", "width"]
    action = features.get("action")
    state = features.get("observation.state")
    if not isinstance(action, dict) or not isinstance(state, dict):
        raise InventoryError(f"missing action or observation.state feature: {root}")
    if _shape(action) != [28] or _shape(state) != [28]:
        raise InventoryError(f"action and observation.state must have shape [28]: {root}")
    action_names = _flatten(action.get("names", action.get("name", [])))
    state_names = _flatten(state.get("names", state.get("name", [])))
    if len(action_names) != 28 or len(state_names) != 28:
        raise InventoryError(f"action and observation.state names must contain 28 joints: {root}")
    if action_names != state_names:
        raise InventoryError(f"action/state joint order mismatch: {root}")
    if tuple(action_names) != EXPECTED_JOINT_ORDER:
        raise InventoryError(f"joint order does not match the Unitree G1 Dex3 contract: {root}")
    if action.get("dtype") != "float32" or state.get("dtype") != "float32":
        raise InventoryError(f"action and state must use float32: {root}")
    episodes = info.get("total_episodes", info.get("episodes"))
    frames = info.get("total_frames", info.get("frames"))
    if type(episodes) is not int or type(frames) is not int or episodes <= 0 or frames < episodes:
        raise InventoryError(f"missing episodes/frames in {info_path}")
    if not (
        (meta / "tasks.parquet").exists() or (meta / "tasks.jsonl").exists() or (meta / "tasks").exists()
    ):
        raise InventoryError(f"missing meta/tasks: {root}")
    episode_meta = (
        list((meta / "episodes").rglob("*.parquet"))
        if (meta / "episodes").is_dir()
        else list(meta.glob("*episode*"))
    )
    if not episode_meta:
        raise InventoryError(f"missing episode metadata: {root}")
    files = [p for p in root.rglob("*") if p.is_file()]
    video = [p for p in files if p.suffix.lower() in {".mp4", ".webm", ".avi", ".mkv"}]
    data = (
        [p for p in (root / "data").rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".pq"}]
        if (root / "data").is_dir()
        else []
    )
    cameras = {
        k: v
        for k, v in features.items()
        if k.startswith("observation.images") or k.startswith("observation.camera")
    }
    if not cameras or not data or not video:
        raise InventoryError(f"missing camera schema, data parquet, or video files: {root}")
    schema = {
        k: {
            "dtype": v.get("dtype"),
            "shape": v.get("shape"),
            "names": _flatten(v["names"]) if v.get("names") is not None else None,
        }
        for k, v in features.items()
    }
    unique_files = {(p.stat().st_dev, p.stat().st_ino): p.stat().st_size for p in files}
    return {
        "repo_id": repo_id,
        "path": str(root),
        "codebase_version": info.get("codebase_version"),
        "source_robot_type": info.get("robot_type"),
        "fps": fps,
        "episodes": int(episodes),
        "frames": int(frames),
        "action_shape": _shape(action),
        "observation_state_shape": _shape(state),
        "action_names": action_names,
        "observation_state_names": state_names,
        "camera_features": cameras,
        "dropped_cameras": dropped_cameras,
        "data_file_count": len(data),
        "video_file_count": len(video),
        "schema": schema,
        "physical_bytes": sum(unique_files.values()),
        "metadata_sha256": _metadata_hash(meta),
        "validation_level": "metadata-only",
    }, fps


def build_inventory(source_root: Path, camera_keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    rows = []
    fps = None
    canonical_names = None
    canonical_cameras = None
    canonical_schema = None
    for repo_id, suffix in zip(REPO_IDS, SUFFIXES, strict=True):
        row, fps = _one(source_root / f"G1_Dex3_{suffix}_Dataset", repo_id, fps, camera_keys)
        names = (tuple(row["action_names"]), tuple(row["observation_state_names"]))
        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise InventoryError(f"incompatible joint order in {repo_id}")
        cameras = {
            key: (value.get("shape"), value.get("dtype")) for key, value in row["camera_features"].items()
        }
        if canonical_cameras is None:
            canonical_cameras = cameras
        elif cameras != canonical_cameras:
            raise InventoryError(f"incompatible camera schema in {repo_id}")
        if canonical_schema is None:
            canonical_schema = row["schema"]
        elif row["schema"] != canonical_schema:
            raise InventoryError(f"incompatible feature schema in {repo_id}")
        rows.append(row)
    return {
        "repo_ids": list(REPO_IDS),
        "total_episodes": sum(r["episodes"] for r in rows),
        "total_frames": sum(r["frames"] for r in rows),
        "validation_level": "metadata-only",
        "note": "Parquet/video contents still require runtime validation.",
        "datasets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-keys", nargs="+")
    args = parser.parse_args()
    try:
        result = build_inventory(args.source_root, tuple(args.camera_keys) if args.camera_keys else None)
    except InventoryError as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
