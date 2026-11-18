"""Prepare a custom, part-disjoint few-shot split from the official KSDD archive.

Download from https://www2.vicos.si/resources/kolektorsdd/ after reviewing its
CC BY-NC-SA 4.0 terms. The source archive and derived images are not redistributed
in the project delivery. This is NOT the official benchmark training protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image


def prepare(archive: Path, destination: Path, shots=4, calibration=20, seed=17):
    if (destination / "manifest.json").exists():
        raise ValueError("destination manifest already exists")
    extracted = destination / "source"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = extracted / item.filename
            if not target.resolve().is_relative_to(extracted.resolve()):
                raise ValueError("unsafe archive member")
        z.extractall(extracted)
    groups = sorted({p.parent for p in extracted.rglob("*.jpg")})
    if len(groups) < 10:
        raise ValueError("expected at least ten KSDD physical-part groups")
    rng = np.random.default_rng(seed)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    n_test, n_cal = max(1, int(len(groups) * 0.3)), max(1, int(len(groups) * 0.25))
    partitions = {"test": groups[:n_test], "calibration": groups[n_test:n_test + n_cal],
                  "train": groups[n_test + n_cal:]}
    payload = {"kind": "real_kolektor_custom_group_split", "seed": seed,
               "description": "Real KolektorSDD images and human defect masks; custom physical-part-disjoint split, not official benchmark protocol",
               "source": "https://www2.vicos.si/resources/kolektorsdd/",
               "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "license": "CC BY-NC-SA 4.0; dataset not redistributed in delivery",
               "groups": {k: [p.name for p in v] for k, v in partitions.items()},
               "train": [], "calibration": [], "test": []}
    for split, group_list in partitions.items():
        candidates = []
        for group in group_list:
            for image in sorted(group.glob("*.jpg")):
                mask = image.with_name(image.stem + "_label.bmp")
                if not mask.exists():
                    raise ValueError(f"missing mask: {mask}")
                with Image.open(mask) as im:
                    is_bad = np.asarray(im.convert("L")).max() > 0
                candidates.append((image, mask, bool(is_bad)))
        if split != "test":
            candidates = [r for r in candidates if not r[2]]
            limit = shots if split == "train" else calibration
            if len(candidates) < limit:
                raise ValueError(f"not enough normal images for {split}")
            candidates = [candidates[i] for i in rng.permutation(len(candidates))[:limit]]
        (destination / split).mkdir(parents=True, exist_ok=True)
        for image, mask, is_bad in candidates:
            name = f"{image.parent.name}_{image.stem}"
            relative = f"{split}/{name}.jpg"
            shutil.copy2(image, destination / relative)
            if split == "test":
                relative_mask = f"test/{name}_mask.png"
                with Image.open(mask) as im:
                    im.convert("L").save(destination / relative_mask)
                payload[split].append({"image": relative, "mask": relative_mask,
                                       "label": int(is_bad), "group": image.parent.name})
            else:
                payload[split].append(relative)
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    # Source copies are redundant; retain only the selected subset locally.
    shutil.rmtree(extracted)
    return {k: len(payload[k]) for k in ["train", "calibration", "test"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--calibration", type=int, default=20)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    print(json.dumps(prepare(a.archive, a.output, a.shots, a.calibration, a.seed)))
