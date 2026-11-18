"""Fixed-threshold lighting-shift evaluation; test labels never select a model."""
from pathlib import Path
import csv
import html
import json
import platform
import time
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from .core import Config, Inspector, load_rgb, save_overlay
from .data import read_manifest


def binary_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores > threshold
    normal, abnormal = labels == 0, labels == 1
    both = len(np.unique(labels)) == 2
    return {"image_auroc": float(roc_auc_score(labels, scores)) if both else None,
            "image_average_precision": float(average_precision_score(labels, scores)) if both else None,
            "normal_false_alarm_rate": float(predicted[normal].mean()) if normal.any() else None,
            "defect_recall": float(predicted[abnormal].mean()) if abnormal.any() else None}


def evaluate(manifest: Path, output: Path, size=224, factors=(1.0, 0.65, 1.35), alpha=0.1,
             max_patches=4096, seed=17):
    if any(f <= 0 for f in factors):
        raise ValueError("lighting factors must be positive")
    payload, refs, calibration, records = read_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    report = {"dataset_kind": payload["kind"], "provenance": payload.get("description", payload.get("source")),
              "reference_count": len(refs), "calibration_count": len(calibration),
              "test_count": len(records), "test_normal": sum(r["label"] == 0 for r in records),
              "test_defect": sum(r["label"] == 1 for r in records),
              "seed": seed, "size": size, "pixel_metric_resolution": [size, size],
              "notes": ["Independent ResNet18 baseline, not a reproduction of cited papers",
                        "Global brightness multiplication with clipping; not physical relighting",
                        "Reference/calibration/test image content checked for overlap",
                        "Clean calibration threshold is reused unchanged for each lighting factor",
                        "Pixel scores are bilinearly resized; masks use nearest-neighbor",
                        "No test-time recalibration or label-driven model selection"],
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "results": []}
    all_rows = []
    for normalization in ["none", "percentile"]:
        t0 = time.perf_counter()
        model = Inspector(Config(size=size, normalization=normalization, max_patches=max_patches, seed=seed))
        model.fit(refs).calibrate(calibration, alpha)
        model.save(output / f"model_{normalization}.npz")
        fit_seconds = time.perf_counter() - t0
        report["environment"].update(model.extractor.versions)
        for factor in factors:
            labels, scores, pixel_labels, pixel_scores, latencies = [], [], [], [], []
            saved = 0
            for i, record in enumerate(records):
                rgb = np.clip(load_rgb(record["image"]) * factor, 0, 1)
                t0 = time.perf_counter()
                result = model.predict_array(rgb)
                latencies.append(time.perf_counter() - t0)
                labels.append(record["label"]); scores.append(result["score"])
                if record["mask"]:
                    with Image.open(record["mask"]) as mask_image:
                        if mask_image.size != (rgb.shape[1], rgb.shape[0]):
                            raise ValueError(f"mask dimensions do not match {record['image']}")
                        mask = np.asarray(mask_image.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
                else:
                    mask = np.zeros((size, size), dtype=bool)
                if bool(mask.any()) != bool(record["label"]):
                    raise ValueError(f"mask and image label disagree for {record['image']}")
                heat = np.asarray(Image.fromarray(result["heatmap"]).resize((size, size), Image.Resampling.BILINEAR))
                pixel_labels.append(mask.ravel()); pixel_scores.append(heat.ravel())
                all_rows.append({"normalization": normalization, "brightness": factor,
                                 "image": str(record["image"]), "label": record["label"],
                                 "score": result["score"], "threshold": model.threshold,
                                 "anomalous": result["anomalous"]})
                # First normal and first two anomalous examples, chosen by input order,
                # not by whether the detector succeeded.
                if i == 0 or (record["label"] and saved < 2):
                    name = f"{normalization}_{factor:g}_{i:03}"
                    Image.fromarray(np.uint8(rgb * 255)).save(output / f"{name}_input.png")
                    save_overlay(rgb, result["heatmap"], model.heat_scale, output / f"{name}_overlay.png")
                    if record["label"]:
                        saved += 1
            py, ps = np.concatenate(pixel_labels), np.concatenate(pixel_scores)
            entry = {"normalization": normalization, "brightness": factor,
                     "threshold": model.threshold, "fit_and_calibration_seconds": fit_seconds,
                     "median_inference_seconds": float(np.median(latencies)),
                     "memory_patches": len(model.memory),
                     "pixel_auroc": float(roc_auc_score(py, ps)) if len(np.unique(py)) == 2 else None,
                     "pixel_average_precision": float(average_precision_score(py, ps)) if py.any() else None,
                     **binary_metrics(labels, scores, model.threshold)}
            report["results"].append(entry)
            print(json.dumps(entry), flush=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    render_report(report, output)
    return report


def render_report(report: dict, output: Path):
    columns = ["normalization", "brightness", "image_auroc", "normal_false_alarm_rate",
               "defect_recall", "pixel_auroc", "pixel_average_precision", "median_inference_seconds"]
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(round(r[c], 4) if isinstance(r[c], float) else r[c]))}</td>"
                                         for c in columns) + "</tr>" for r in report["results"])
    if report["dataset_kind"].startswith("synthetic"):
        gallery = "".join(f'<figure><img src="{p.name}" alt="Anomaly heatmap overlay"><figcaption>{p.stem}</figcaption></figure>'
                          for p in sorted(output.glob("*_overlay.png")))
    else:
        gallery = "<p>Source industrial photographs and derived overlays are omitted from the distributable report. Download the licensed dataset and rerun evaluation to inspect them locally.</p>"
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Few-Shot Defect Inspector — Validation</title><style>
body{font:16px system-ui;background:#0b1220;color:#e8edf6;margin:0;padding:32px;max-width:1300px;margin:auto}
h1{font-size:36px}p{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:10px;border:1px solid #34445d;text-align:left}.scroll{overflow:auto}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
figure{margin:0;background:#172236;padding:12px;border-radius:12px}img{width:100%;max-height:360px;object-fit:contain}
figcaption{font-size:12px;margin-top:8px}.notice{background:#24354b;padding:16px;border-left:4px solid #8cc7fb}
</style><h1>Few-Shot Defect Inspector</h1>'''
    page += f'<p class="notice">Dataset: <strong>{html.escape(report["dataset_kind"])}</strong>. '
    page += f'{report["reference_count"]} normal references · {report["calibration_count"]} calibration images · {report["test_count"]} test images.</p>'
    page += "<p>" + html.escape(str(report["provenance"])) + "</p>"
    page += "<p>" + "<br>".join(html.escape(n) for n in report["notes"]) + "</p>"
    page += '<div class="scroll"><table><tr>' + "".join(f"<th>{c}</th>" for c in columns) + "</tr>" + rows + "</table></div>"
    page += '<h2>Examples selected by input order</h2><p>Overlays share a fixed calibration scale within each model. Red means a larger distance from known normal patches; it is not a calibrated defect probability.</p><div class="gallery">' + gallery + "</div></html>"
    (output / "report.html").write_text(page)
