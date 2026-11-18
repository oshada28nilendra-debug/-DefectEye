"""Frozen ResNet patch features, normal memory bank, and normal-only calibration.

This is an independent baseline, not an implementation of AnomalyDINO/SubspaceAD.
No neural-network fine-tuning occurs. Fitting stores normal patch embeddings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from sklearn.metrics import pairwise_distances_argmin_min


def image_hash(path: str | Path) -> str:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32) / 255


def normalize_lighting(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return rgb.copy()
    if mode != "percentile":
        raise ValueError("normalization must be none or percentile")
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    low, high = np.percentile(luminance, [2, 98])
    # Common RGB affine transform preserves relative channel differences; clipping
    # and saturation are irreversible, so this is not general illumination correction.
    return np.clip((rgb - low) / max(float(high - low), 0.05), 0, 1).astype(np.float32)


@dataclass
class Config:
    size: int = 224
    normalization: str = "none"
    max_patches: int = 4096
    top_fraction: float = 0.01
    seed: int = 17
    threads: int = 2

    def validate(self):
        if self.size < 64 or self.size > 1024 or self.size % 32:
            raise ValueError("size must be a multiple of 32 between 64 and 1024")
        if self.normalization not in {"none", "percentile"}:
            raise ValueError("unknown normalization")
        if self.max_patches < 2 or not 0 < self.top_fraction <= 1 or self.threads < 1:
            raise ValueError("invalid memory, aggregation or thread configuration")


class FeatureExtractor:
    def __init__(self, config: Config):
        import torch
        import torchvision
        from torchvision.models import ResNet18_Weights, resnet18
        config.validate()
        torch.set_num_threads(config.threads)
        self.torch = torch
        self.config = config
        self.versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2 = model.layer1, model.layer2
        for module in [self.stem, self.layer1, self.layer2]:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        t = self.torch
        rgb = normalize_lighting(rgb, self.config.normalization)
        resized = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).resize(
            (self.config.size, self.config.size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255
        x = t.from_numpy(arr.transpose(2, 0, 1).copy())[None]
        with t.inference_mode():
            a = self.layer1(self.stem((x - self.mean) / self.std))
            b = self.layer2(a)
            a = t.nn.functional.adaptive_avg_pool2d(a, b.shape[-2:])
            features = t.cat([a, b], dim=1)
            features = t.nn.functional.avg_pool2d(features, 3, stride=1, padding=1)
            features = t.nn.functional.normalize(features, dim=1)
        grid = tuple(features.shape[-2:])
        return features[0].permute(1, 2, 0).reshape(-1, 192).numpy().copy(), grid


class Inspector:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.validate()
        self.extractor = FeatureExtractor(self.config)
        self.memory = None
        self.threshold = None
        self.heat_scale = None
        self.metadata = {"schema": 1, "backbone": "resnet18_IMAGENET1K_V1_layer1_layer2",
                         "versions": self.extractor.versions}

    def fit(self, paths: list[Path]):
        if not paths:
            raise ValueError("at least one normal reference image is required")
        hashes = [image_hash(p) for p in paths]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate reference images")
        features = np.concatenate([self.extractor(load_rgb(p))[0] for p in paths])
        if len(features) > self.config.max_patches:
            rng = np.random.default_rng(self.config.seed)
            features = features[rng.choice(len(features), self.config.max_patches, replace=False)]
        self.memory = features.astype(np.float32)
        self.threshold = self.heat_scale = None
        self.metadata.update({"reference_hashes": hashes, "references": [str(p) for p in paths],
                              "reference_count": len(paths), "memory_patches": len(features)})
        return self

    def predict_array(self, rgb: np.ndarray) -> dict:
        if self.memory is None:
            raise ValueError("fit or load a model first")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
            raise ValueError("expected finite HxWx3 RGB array")
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB values must be in [0, 1]")
        features, grid = self.extractor(rgb)
        # Exact nearest neighbors; no approximate search or test-time adaptation.
        _, distances = pairwise_distances_argmin_min(features, self.memory, metric="euclidean")
        k = max(1, math.ceil(len(distances) * self.config.top_fraction))
        score = float(np.mean(np.partition(distances, -k)[-k:]))
        coarse = gaussian_filter(distances.reshape(grid), sigma=0.7)
        heatmap = np.asarray(Image.fromarray(coarse.astype(np.float32)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)).copy()
        return {"score": score, "anomalous": None if self.threshold is None else score > self.threshold,
                "threshold": self.threshold, "heatmap": heatmap}

    def calibrate(self, paths: list[Path], alpha: float = 0.1):
        if self.memory is None:
            raise ValueError("fit first")
        if not 0 < alpha < 1 or not paths:
            raise ValueError("normal calibration images and alpha in (0,1) are required")
        hashes = [image_hash(p) for p in paths]
        if len(set(hashes)) != len(hashes) or set(hashes) & set(self.metadata["reference_hashes"]):
            raise ValueError("calibration must be unique and disjoint from references")
        rank = math.ceil((len(paths) + 1) * (1 - alpha))
        if rank > len(paths):
            raise ValueError("not enough calibration images for this alpha; increase alpha or add images")
        predictions = [self.predict_array(load_rgb(p)) for p in paths]
        self.threshold = float(np.sort([p["score"] for p in predictions])[rank - 1])
        self.heat_scale = max(float(np.quantile(np.concatenate(
            [p["heatmap"].ravel() for p in predictions]), 0.995)), 1e-6)
        self.metadata.update({"calibration_hashes": hashes, "calibration_count": len(paths),
                              "alpha": alpha, "threshold_rank": rank,
                              "threshold_note": "finite-sample normal-score order statistic; exchangeability required"})
        return self

    def save(self, path: str | Path):
        if self.memory is None:
            raise ValueError("cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.metadata, "config": asdict(self.config), "threshold": self.threshold,
                "heat_scale": self.heat_scale}
        with path.open("wb") as handle:
            np.savez_compressed(handle, memory=self.memory, metadata=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as bundle:
            meta = json.loads(str(bundle["metadata"]))
            if meta.get("schema") != 1:
                raise ValueError("unsupported model schema")
            memory = np.array(bundle["memory"], dtype=np.float32)
        if memory.ndim != 2 or memory.shape[1] != 192 or not len(memory) or not np.isfinite(memory).all():
            raise ValueError("invalid patch memory")
        model = cls(Config(**meta["config"]))
        model.memory = memory
        model.threshold, model.heat_scale = meta["threshold"], meta["heat_scale"]
        model.metadata = meta
        return model


def save_overlay(rgb: np.ndarray, heatmap: np.ndarray, scale: float, path: Path):
    """Use a fixed model-calibration scale, never per-image min/max normalization."""
    z = np.clip(heatmap / max(scale, 1e-6), 0, 1)
    # Green through yellow to red, with low scores nearly transparent.
    color = np.stack([z, 1 - z, np.zeros_like(z)], axis=-1)
    opacity = (0.55 * z)[..., None]
    overlay = np.clip(rgb * (1 - opacity) + color * opacity, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(overlay * 255)).save(path)
