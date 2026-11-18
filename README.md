# DefectEye

DefectEye learns what a good part looks like from a few reference photos. Given a new photo of the same part type, it assigns an anomaly score and highlights the regions that differ from the reference images.

**Status:** CPU research prototype, validated on 17 November 2026.

[Detailed setup and reproduction commands](01-few-shot-defect-inspector/README.md) · [Validation report](01-few-shot-defect-inspector/VALIDATION.md) · [License](01-few-shot-defect-inspector/LICENSE)

## What it does

- Fits a reference model from defect-free images and calibrates the alert threshold on a separate set of good images.
- Scores new images and exports a heatmap and overlay so you can inspect the flagged region.
- Compares raw input with percentile normalization under simulated brightness changes.

## Quick start

Use **Python 3.10+**. From the root of a local clone:

```bash
cd 01-few-shot-defect-inspector
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -e .
python -m inspector.cli demo --output outputs/my-demo
```

Open `outputs/my-demo/evaluation/report.html` after the command finishes. The demo generates synthetic images and defects. Choose a fresh output directory for each run. The first run downloads the pretrained ResNet18 weights, approximately 45 MB.

## Use your own data

Put defect-free images in separate reference and calibration folders. Keep the product type and camera view consistent. The example uses four reference images and requires at least nine calibration images at the default threshold setting.

```bash
python -m inspector.cli fit --normal-dir data/reference --calibration-dir data/calibration --shots 4 --model outputs/model.npz
python -m inspector.cli predict --model outputs/model.npz --image data/query.png --output outputs/query
```

Prediction writes `prediction.json`, `heatmap.npy`, and `overlay.png`.

## How it works

A frozen ImageNet-pretrained ResNet18 extracts features from image patches. The detector builds a memory bank of normal patches and scores each new patch by its distance to the closest reference. The highest patch distances contribute to the image score, and their locations form the heatmap.

## Measured results

The [KolektorSDD](https://www2.vicos.si/resources/kolektorsdd/) experiment used four normal reference images, 20 separate calibration images, and 119 test images. Physical parts were kept separate across the custom splits.

| Raw input condition | Image AUROC | False alarms among good images | Defects detected |
| --- | ---: | ---: | ---: |
| Original brightness | 0.7870 | 9/103 | 5/16 |
| 0.65× brightness | 0.7215 | 39/103 | 11/16 |
| 1.35× brightness | 0.6299 | 102/103 | 16/16 |

The original threshold missed 11 of 16 defects. Brightness changes increased false alarms substantially, and the tested normalization traded fewer false alarms in darker images for weaker detection. The evaluation covers one custom split. [Full metrics and limitations](01-few-shot-defect-inspector/VALIDATION.md).

## Repository guide

| Path | Contents |
| --- | --- |
| [`inspector/`](01-few-shot-defect-inspector/inspector) | Model, CLI, and evaluation code |
| [`scripts/`](01-few-shot-defect-inspector/scripts) | Dataset preparation |
| [`tests/`](01-few-shot-defect-inspector/tests) | Behavioral tests |
| [`reports/`](01-few-shot-defect-inspector/reports) | Measured results, per-sample scores, and dataset split metadata |

From the project folder, run `python -m unittest discover -s tests -v`. All **10 tests** passed in the recorded validation.
