"""Dex3 right-hand + bottle detector for the G1 ego view (plan: docs/research/2026-10-15-g1-handover-261014-out-of-view.md, section 6).

sample OUT   pick ~500 ego-view frames from the handover recordings, write OUT/images/*.jpg and OUT/frames.json
             (source, episode, frame, split, reviewer label). Every episode contributes frames spread over its
             length; excluded intervals and out-of-frame Failed episodes are oversampled so out-of-view cases are
             well covered. Test split = whole episodes (10 new + the held-out old 27, 35).
labels OUT   export the boxes from the local Label Studio (OUT/start_label_studio.sh, project 1) to OUT/labels.json
train OUT [EPOCHS]   fine-tune torchvision Faster R-CNN MobileNetV3 (COCO) on the train split -> OUT/model.pt
eval OUT     per-class presence accuracy and box IoU on the test split (whole held-out episodes)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import av
import numpy as np
import torch

CLASSES = ("right_hand", "bottle")
LS_URL = "http://127.0.0.1:8090"
EDGE_PX = 3  # a box this close to the image border counts as cut off

D = Path("/mnt/data/jihun/datasets/G1_WBT_GR00T")
NEW, OLD = D / "handover_261014", D / "handover_bottle_260930_subset_261002"
RUN = D / "official_annotations/workspace/runs/36a1f30ab44f451cab67150e5319bba7/run.json"
VIDEO = "videos/chunk-000/observation.images.ego_view/episode_{:06d}.mp4"
MIN_GAP = 25  # frames (0.5 s) between samples of one episode


def lengths(root: Path) -> dict[int, int]:
    return {e["episode_index"]: e["length"] for e in map(json.loads, (root / "meta/episodes.jsonl").open())}


def spread(rng, n: int, start: int, end: int) -> list[int]:
    """n frames in [start, end), one per equal bin."""
    return [int(start + (k + rng.random()) / n * (end - start)) for k in range(n)]


def thin(frames: list[int]) -> list[int]:
    out: list[int] = []
    for f in sorted(set(frames)):
        if not out or f - out[-1] >= MIN_GAP:
            out.append(f)
    return out


def label(review: dict | None, frame: int) -> str:
    if review is None:
        return "kept"
    if review["decision"] == "delete":
        return "failed:" + review["decision_reason"]
    inside = any(iv["start_frame"] <= frame < iv["end_frame"] for iv in review.get("excluded_intervals", []))
    return "excluded" if inside else "kept"


def plan(rng) -> list[dict]:
    run = json.loads(RUN.read_text())["episodes"]
    new_test = set(rng.choice(sorted(lengths(NEW)), 10, replace=False).tolist())
    rows = []
    for ep, n in lengths(NEW).items():
        review = run[str(ep)]
        frames = spread(rng, 6, 0, n)
        for iv in review.get("excluded_intervals", [])[:3]:
            frames += spread(rng, 1, iv["start_frame"], iv["end_frame"])
        if review["decision"] == "delete" and "frame" in review["decision_reason"]:
            frames += spread(rng, 3, 0, n)
        rows += [
            {
                "source": "new",
                "episode": ep,
                "frame": f,
                "split": "test" if ep in new_test else "train",
                "review": label(review, f),
            }
            for f in thin(frames)
        ]
    for ep, n in lengths(OLD).items():
        rows += [
            {
                "source": "old",
                "episode": ep,
                "frame": f,
                "split": "test" if ep in (27, 35) else "train",
                "review": "kept",
            }
            for f in thin(spread(rng, 3, 0, n))
        ]
    return rows


def sample(out: Path) -> None:
    rows = plan(np.random.default_rng(0))
    (out / "images").mkdir(parents=True, exist_ok=True)
    for source, ep in sorted({(r["source"], r["episode"]) for r in rows}):
        want = {r["frame"]: r for r in rows if (r["source"], r["episode"]) == (source, ep)}
        with av.open(str((NEW if source == "new" else OLD) / VIDEO.format(ep))) as c:
            for i, f in enumerate(c.decode(video=0)):
                if i in want:
                    name = f"{source}_ep{ep:03d}_f{i:05d}.jpg"
                    f.to_image().save(out / "images" / name, quality=95)
                    want[i]["image"] = name
                if i >= max(want):
                    break
    missing = [r for r in rows if "image" not in r]
    assert not missing, f"frames past the video end: {missing[:3]}"
    (out / "frames.json").write_text(json.dumps(rows, indent=1))
    split_n, review_n = (Counter(r[key] for r in rows) for key in ("split", "review"))
    print(f"{len(rows)} frames -> {out}; split {dict(split_n)}; review {dict(review_n)}")


def labels(out: Path) -> None:
    import requests

    login = dict(line.strip().split("=", 1) for line in (out / "label_studio/.login").open() if "=" in line)
    s = requests.Session()
    s.get(f"{LS_URL}/user/login/")
    s.post(
        f"{LS_URL}/user/login/",
        data={
            "email": login["LS_USER"],
            "password": login["LS_PASS"],
            "csrfmiddlewaretoken": s.cookies["csrftoken"],
        },
        headers={"Referer": f"{LS_URL}/user/login/"},
    )
    tasks = s.get(f"{LS_URL}/api/projects/1/export?exportType=JSON", headers={"Referer": LS_URL})
    tasks.raise_for_status()
    result = {}
    for task in tasks.json():
        done = [a for a in task["annotations"] if not a.get("was_cancelled")]
        if not done:
            continue
        boxes = []
        for r in done[-1]["result"]:
            if r["type"] == "rectanglelabels":
                v, w, h = r["value"], r["original_width"] / 100, r["original_height"] / 100
                boxes.append(
                    [
                        v["rectanglelabels"][0],
                        v["x"] * w,
                        v["y"] * h,
                        (v["x"] + v["width"]) * w,
                        (v["y"] + v["height"]) * h,
                    ]
                )
        result[task["data"]["name"]] = boxes
    (out / "labels.json").write_text(json.dumps(result, indent=1))
    n = {c: sum(any(b[0] == c for b in bs) for bs in result.values()) for c in CLASSES}
    print(f"{len(result)} labelled frames (frames with a box: {n}) -> {out / 'labels.json'}")


def split(out: Path, which: str) -> tuple[list[str], dict]:
    lab = json.loads((out / "labels.json").read_text())
    names = [
        r["image"]
        for r in json.loads((out / "frames.json").read_text())
        if r["split"] == which and r["image"] in lab
    ]
    return names, lab


def build_model(pretrained: bool):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    m = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT" if pretrained else None, weights_backbone=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1
    )
    return m


def load(out: Path, name: str, lab: dict, jitter=None):
    from torchvision.io import read_image

    img = read_image(str(out / "images" / name)).float() / 255
    if jitter is not None:
        img = jitter(img)
    boxes = [b for b in lab[name] if b[0] in CLASSES]
    return img, {
        "boxes": torch.tensor([b[1:] for b in boxes], dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor([CLASSES.index(b[0]) + 1 for b in boxes], dtype=torch.int64),
    }


def train(out: Path, epochs: int = 30, batch: int = 8, dev: str = "cuda:0") -> None:
    from torchvision.transforms import ColorJitter  # no flips: they would turn the right hand into a left one

    names, lab = split(out, "train")
    jitter = ColorJitter(0.4, 0.4, 0.3, 0.03)  # recordings differ mostly in lighting
    model = build_model(pretrained=True).to(dev).train()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    steps = epochs * -(-len(names) // batch)
    warm = max(steps // 10, 1)  # linear warmup, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps))
    )
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        order, total = rng.permutation(len(names)), 0.0
        for k in range(0, len(names), batch):
            imgs, targets = zip(
                *(load(out, names[i], lab, jitter) for i in order[k : k + batch]), strict=True
            )
            loss = sum(
                model(
                    [i.to(dev) for i in imgs], [{k2: v.to(dev) for k2, v in t.items()} for t in targets]
                ).values()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(imgs)
        print(f"epoch {epoch + 1}/{epochs} loss {total / len(names):.4f}", flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    print(f"{len(names)} train frames -> {out / 'model.pt'}")


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


@torch.no_grad()
def predict(model, imgs, dev: str) -> list[dict]:
    """class -> (best score, its box) per image."""
    res = []
    for o in model([i.to(dev) for i in imgs]):
        best = {}
        for s, cls, b in zip(o["scores"].tolist(), o["labels"].tolist(), o["boxes"].tolist(), strict=True):
            c = CLASSES[cls - 1]
            if s > best.get(c, (0, None))[0]:
                best[c] = (s, b)
        res.append(best)
    return res


def evaluate(out: Path, dev: str = "cuda:0") -> None:
    names, lab = split(out, "test")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    model.to(dev).eval()
    preds = []
    for k in range(0, len(names), 8):
        imgs = [load(out, n, lab)[0] for n in names[k : k + 8]]
        preds += predict(model, imgs, dev)
    width, height = 640, 480
    print(f"{len(names)} test frames")
    for c in CLASSES:
        gt = [[b[1:] for b in lab[n] if b[0] == c] for n in names]
        cut = [any(min(b[0], b[1], width - b[2], height - b[3]) < EDGE_PX for b in g) for g in gt]
        for threshold in (0.3, 0.5, 0.7):
            pred = [p.get(c, (0, None))[0] >= threshold for p in preds]
            has = [bool(g) for g in gt]
            acc = np.mean([p == h for p, h in zip(pred, has, strict=True)])
            tp = sum(p and h for p, h in zip(pred, has, strict=True))
            ious = [
                max(iou(p[c][1], g) for g in gt_i)
                for p, gt_i, ok in zip(preds, gt, pred, strict=True)
                if ok and gt_i
            ]
            cut_acc = (
                np.mean([p for p, h, ct in zip(pred, has, cut, strict=True) if ct])
                if any(cut)
                else float("nan")
            )
            print(
                f"{c:10s} threshold {threshold}: presence acc {acc:.3f} (recall {tp / max(sum(has), 1):.3f}, precision "
                f"{tp / max(sum(pred), 1):.3f}; {sum(has)} with / {len(has) - sum(has)} without), cut-off boxes found "
                f"{cut_acc:.3f} ({sum(cut)}), box IoU mean {np.mean(ious) if ious else float('nan'):.3f}"
                f" (>=0.5: {np.mean([i >= 0.5 for i in ious]) if ious else float('nan'):.3f})"
            )


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 2 else ("", [])
    if cmd == "sample":
        sample(Path(args[0]))
    elif cmd == "labels":
        labels(Path(args[0]))
    elif cmd == "train":
        train(Path(args[0]), *map(int, args[1:2]))
    elif cmd == "eval":
        evaluate(Path(args[0]))
    else:
        sys.exit(__doc__)
