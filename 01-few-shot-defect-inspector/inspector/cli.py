from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import Config, Inspector, load_rgb, save_overlay
from .data import images, mvtec_manifest, synthetic
from .evaluate import evaluate


def parser():
    p = argparse.ArgumentParser(description="Few-shot normal patch-memory inspection with a pretrained ResNet18")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Generate explicitly synthetic data, fit and compare lighting shifts")
    d.add_argument("--output", type=Path, default=Path("outputs/synthetic-demo"))
    d.add_argument("--seed", type=int, default=17)
    f = sub.add_parser("fit", help="Fit references and calibrate on separate normal images")
    f.add_argument("--normal-dir", type=Path, required=True)
    f.add_argument("--calibration-dir", type=Path, required=True)
    f.add_argument("--model", type=Path, required=True)
    f.add_argument("--normalization", choices=["none", "percentile"], default="none")
    f.add_argument("--shots", type=int, default=4)
    f.add_argument("--size", type=int, default=224)
    f.add_argument("--alpha", type=float, default=0.1)
    f.add_argument("--seed", type=int, default=17)
    i = sub.add_parser("predict", help="Inspect one image; write raw scores, heatmap and overlay")
    i.add_argument("--model", type=Path, required=True)
    i.add_argument("--image", type=Path, required=True)
    i.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate", help="Compare raw/normalized inputs under fixed lighting factors")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--size", type=int, default=224)
    e.add_argument("--alpha", type=float, default=0.1)
    e.add_argument("--seed", type=int, default=17)
    e.add_argument("--factors", type=float, nargs="+", default=[1.0, 0.65, 1.35])
    m = sub.add_parser("prepare-mvtec", help="Create manifest from an existing licensed MVTec-style category")
    m.add_argument("--category", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--shots", type=int, default=4)
    m.add_argument("--calibration", type=int, default=20)
    m.add_argument("--seed", type=int, default=17)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    try:
        if args.command == "demo":
            manifest = synthetic(args.output / "data", args.seed)
            evaluate(manifest, args.output / "evaluation", seed=args.seed)
        elif args.command == "fit":
            paths = images(args.normal_dir)
            if args.shots < 1 or len(paths) < args.shots:
                raise ValueError("not enough normal images for requested shots")
            model = Inspector(Config(size=args.size, normalization=args.normalization, seed=args.seed))
            model.fit(paths[:args.shots]).calibrate(images(args.calibration_dir), args.alpha)
            model.save(args.model)
            print(json.dumps({"model": str(args.model), "threshold": model.threshold,
                              "reference_count": args.shots, "calibration_count": model.metadata["calibration_count"]}))
        elif args.command == "predict":
            model = Inspector.load(args.model)
            rgb = load_rgb(args.image)
            result = model.predict_array(rgb)
            args.output.mkdir(parents=True, exist_ok=True)
            heat = result.pop("heatmap")
            np.save(args.output / "heatmap.npy", heat)
            save_overlay(rgb, heat, model.heat_scale or 1.0, args.output / "overlay.png")
            (args.output / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
        elif args.command == "evaluate":
            evaluate(args.manifest, args.output, args.size, args.factors, args.alpha, seed=args.seed)
        elif args.command == "prepare-mvtec":
            mvtec_manifest(args.category, args.output, args.shots, args.calibration, args.seed)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
