import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from prepare_state46 import video_link


def _linked(src: Path, out: Path) -> Path:
    out.mkdir(parents=True)
    (out / "videos").symlink_to(video_link(src, out), target_is_directory=True)
    return out / "videos"


def test_video_link_resolves_from_any_output_dir():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = root / "input" / "src"
        (src / "videos" / "cam").mkdir(parents=True)
        for out in (root / "input" / "sibling", root / "output" / "deeper" / "out"):
            assert (_linked(src, out) / "cam").is_dir()


def test_video_link_rebases_relative_and_keeps_absolute_source_links():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "shared" / "videos_g2" / "cam").mkdir(parents=True)
        rel = root / "input" / "rel"
        rel.mkdir(parents=True)
        (rel / "videos").symlink_to("../../shared/videos_g2", target_is_directory=True)
        assert (_linked(rel, root / "output" / "out") / "cam").is_dir()

        absolute = root / "input" / "abs"
        absolute.mkdir()
        (absolute / "videos").symlink_to("/run-output/videos_g2", target_is_directory=True)
        assert video_link(absolute, root / "output" / "other") == "/run-output/videos_g2"
