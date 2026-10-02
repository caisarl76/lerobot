"""Add exact global action/state quantiles to a prepared G1 dataset without decoding video."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def augment(root: Path) -> dict:
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    total = info["total_frames"]
    summary = {}
    for key in ("observation.state", "action"):
        width = info["features"][key]["shape"][0]
        pieces = []
        for path in sorted((root / "data").rglob("*.parquet")):
            column = pq.read_table(path, columns=[key])[key].combine_chunks()
            values = pc.list_flatten(column).to_numpy(zero_copy_only=False).reshape(len(column), width)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {key}: {path}")
            pieces.append(values)
        values = np.concatenate(pieces)
        if values.shape != (total, width):
            raise ValueError(f"{key} row/width mismatch: {values.shape} != {(total, width)}")
        quantiles = np.quantile(values, QUANTILES, axis=0)
        for i, q in enumerate(QUANTILES):
            stats[key][f"q{round(q * 100):02d}"] = quantiles[i].tolist()
        summary[key] = {"rows": total, "width": width, "q01_min": float(quantiles[0].min()), "q99_max": float(quantiles[-1].max())}
        print(f"computed exact {key} quantiles over {total} rows", flush=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=stats_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(stats, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, stats_path)
    audit = {"method": "exact numpy quantile over every parquet row; no video decoding", "quantiles": list(QUANTILES), "features": summary}
    (root / "meta/quantile-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(augment(parser.parse_args().root), indent=2), flush=True)


if __name__ == "__main__":
    main()
