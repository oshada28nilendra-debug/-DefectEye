"""Explicit manifests for synthetic smoke tests and real MVTec-style datasets."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images(directory: str | Path):
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in EXTENSIONS)


def write_manifest(root: Path, payload: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def synthetic(root: Path, seed: int = 17):
    if (root / "manifest.json").exists():
        raise ValueError("destination already has a manifest; choose a new output directory")
    rng = np.random.default_rng(seed)
    size = 224
    y, x = np.mgrid[:size, :size]
    payload = {"kind": "synthetic_smoke_test", "seed": seed,
               "description": "Procedural textured panels and inserted scratches/blobs; not industrial benchmark data",
               "train": [], "calibration": [], "test": []}
    for split, count in [("train", 4), ("calibration", 12), ("test", 32)]:
        (root / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            phase = rng.uniform(-0.15, 0.15)
            gray = (0.50 + 0.09 * np.sin(x * 0.21 + phase)
                    + 0.04 * np.sin(y * 0.10 + phase)
                    + rng.normal(0, 0.008, (size, size)))
            rgb = np.stack([gray * 1.08, gray, gray * 0.90], axis=-1)
            im = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255))
            mask = Image.new("L", im.size)
            is_bad = split == "test" and i >= count // 2
            if is_bad:
                d, m = ImageDraw.Draw(im), ImageDraw.Draw(mask)
                cx, cy = rng.integers(45, size - 65, size=2).tolist()
                if i % 2:
                    shape = (cx, cy, cx + 18, cy + 28)
                    d.ellipse(shape, fill=(15, 15, 15)); m.ellipse(shape, fill=255)
                else:
                    points = [(cx, cy), (cx + 35, cy + 45)]
                    d.line(points, fill=(235, 235, 235), width=5)
                    m.line(points, fill=255, width=5)
            name = f"{split}/{i:03}.png"
            im.save(root / name)
            if split == "test":
                mask_name = f"test/{i:03}_mask.png"
                mask.save(root / mask_name)
                payload[split].append({"image": name, "label": int(is_bad),
                                       "mask": mask_name, "group": f"synthetic-{i}"})
            else:
                payload[split].append(name)
    write_manifest(root, payload)
    return root / "manifest.json"


def mvtec_manifest(category: Path, output: Path, shots=4, calibration=20, seed=17):
    """Split ONLY train/good for references and calibration; leave official test intact."""
    candidates = images(category / "train" / "good")
    if shots < 1 or calibration < 1 or len(candidates) < shots + calibration:
        raise ValueError("not enough normal training images for references plus calibration")
    order = np.random.default_rng(seed).permutation(len(candidates))
    payload = {"kind": "real_mvtec_layout", "source": str(category.resolve()), "seed": seed,
               "train": [str(candidates[i].resolve()) for i in order[:shots]],
               "calibration": [str(candidates[i].resolve()) for i in order[shots:shots + calibration]],
               "test": []}
    for p in images(category / "test"):
        normal = p.parent.name == "good"
        mask = category / "ground_truth" / p.parent.name / (p.stem + "_mask.png")
        if not normal and not mask.exists():
            raise ValueError(f"missing anomaly mask: {mask}")
        payload["test"].append({"image": str(p.resolve()), "label": int(not normal),
                                "mask": None if normal else str(mask.resolve()), "group": p.stem})
    if not payload["test"]:
        raise ValueError("test folder is empty")
    write_manifest(output.parent, payload)
    if output.name != "manifest.json":
        (output.parent / "manifest.json").rename(output)


def read_manifest(path: Path):
    payload = json.loads(path.read_text())
    for key in ["kind", "train", "calibration", "test"]:
        if key not in payload:
            raise ValueError(f"manifest missing {key}")
    resolve = lambda p: Path(p) if Path(p).is_absolute() else path.parent / p
    refs = [resolve(p) for p in payload["train"]]
    calibration = [resolve(p) for p in payload["calibration"]]
    records = []
    for row in payload["test"]:
        if row["label"] not in [0, 1]:
            raise ValueError("labels must be 0 or 1")
        if row["label"] and not row.get("mask"):
            raise ValueError("anomalous records require a mask")
        records.append({**row, "image": resolve(row["image"]),
                        "mask": resolve(row["mask"]) if row.get("mask") else None})
    # Check actual decoded image content, not only filenames, to catch copied data.
    from .core import image_hash
    groups = [refs, calibration, [r["image"] for r in records]]
    seen = set()
    for group in groups:
        hashes = [image_hash(p) for p in group]
        if len(hashes) != len(set(hashes)) or seen.intersection(hashes):
            raise ValueError("duplicate image content or split leakage in manifest")
        seen.update(hashes)
    if not refs or not calibration or not records:
        raise ValueError("all three splits must be nonempty")
    return payload, refs, calibration, records
