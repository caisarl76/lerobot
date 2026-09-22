from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lerobot.datasets.aggregate import aggregate_datasets, aggregate_videos


def _metadata(root, *, files=(0,)):
    return SimpleNamespace(
        root=root,
        episodes={
            "videos/cam/chunk_index": [0 for _ in files],
            "videos/cam/file_index": list(files),
        },
    )


def _video_destination(root, file_index):
    return root / "videos" / "cam" / "chunk-000" / f"file-{file_index:03d}.mp4"


def test_symlink_videos_requires_non_concatenated_videos_before_side_effects(tmp_path):
    with (
        patch("lerobot.datasets.aggregate.LeRobotDatasetMetadata") as metadata,
        pytest.raises(ValueError, match="symlink_videos=True requires concatenate_videos=False"),
    ):
        aggregate_datasets([], "output", aggr_root=tmp_path, symlink_videos=True)
    metadata.assert_not_called()
    assert not (tmp_path / "meta").exists()

    with pytest.raises(ValueError, match="symlink_videos=True requires concatenate_videos=False"):
        aggregate_videos(
            _metadata(tmp_path),
            _metadata(tmp_path / "out"),
            {"cam": {"chunk": 0, "file": 0}},
            1,
            1,
            symlink_videos=True,
        )


def test_symlinked_video_files_resolve_to_source_and_preserve_bytes(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_path = src_root / "videos" / "cam" / "chunk-000" / "file-000.mp4"
    src_path.parent.mkdir(parents=True)
    src_path.write_bytes(b"tiny video bytes")

    with patch("lerobot.datasets.aggregate.get_video_duration_in_s", return_value=1.0):
        aggregate_videos(
            _metadata(src_root),
            _metadata(dst_root),
            {"cam": {"chunk": 0, "file": 0}},
            100,
            1,
            concatenate_videos=False,
            symlink_videos=True,
        )

    dst_path = _video_destination(dst_root, 0)
    assert dst_path.is_symlink()
    assert dst_path.resolve() == src_path.resolve()
    assert dst_path.read_bytes() == src_path.read_bytes()


def test_symlink_rotation_points_each_file_to_source(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    for file_index, contents in enumerate((b"first", b"second")):
        path = src_root / "videos" / "cam" / "chunk-000" / f"file-{file_index:03d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    with patch("lerobot.datasets.aggregate.get_video_duration_in_s", return_value=1.0):
        aggregate_videos(
            _metadata(src_root, files=(0, 1)),
            _metadata(dst_root),
            {"cam": {"chunk": 0, "file": 0}},
            0,
            1,
            concatenate_videos=False,
            symlink_videos=True,
        )

    for file_index in (0, 1):
        src_path = src_root / "videos" / "cam" / "chunk-000" / f"file-{file_index:03d}.mp4"
        # With one file per chunk, rotation advances the chunk, not file number.
        dst_path = dst_root / "videos" / "cam" / f"chunk-{file_index:03d}" / "file-000.mp4"
        assert dst_path.is_symlink()
        assert dst_path.resolve() == src_path.resolve()
        assert dst_path.read_bytes() == src_path.read_bytes()


def test_default_video_aggregation_copies_files(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_path = src_root / "videos" / "cam" / "chunk-000" / "file-000.mp4"
    src_path.parent.mkdir(parents=True)
    src_path.write_bytes(b"copied video bytes")

    with patch("lerobot.datasets.aggregate.get_video_duration_in_s", return_value=1.0):
        aggregate_videos(
            _metadata(src_root),
            _metadata(dst_root),
            {"cam": {"chunk": 0, "file": 0}},
            100,
            1,
            concatenate_videos=False,
        )

    dst_path = _video_destination(dst_root, 0)
    assert not dst_path.is_symlink()
    assert dst_path.read_bytes() == src_path.read_bytes()
