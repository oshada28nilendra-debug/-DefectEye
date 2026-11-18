# 01 — Few-Shot Defect Inspector

A runnable research prototype for learning normal product appearance from a few
reference photographs, assigning anomaly scores, and producing localized heatmaps.
Includes a controlled comparison under brightness changes and **actual evaluation
on real industrial images**, separately from synthetic smoke tests.

**Status:** Implemented and locally validated on 17 November 2026. This is a baseline and experiment, not a production quality-control system.

## Quick start

Python 3.10+; validation used Python 3.12 on Linux CPU. From this project folder:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m inspector.cli --help
python -m inspector.cli demo --output outputs/my-demo
```

Open `outputs/my-demo/evaluation/report.html` in a browser. This demo generates
**procedural synthetic textures and synthetic defects**; it is not an industrial
benchmark. Use a new output folder for subsequent demo runs to avoid overwriting data.

The first use automatically downloads official ImageNet-pretrained ResNet18 weights
(about 45 MB) via torchvision. After caching, inference is local and requires no API
or cloud model. An uncached offline system needs the weights transferred into the
normal PyTorch checkpoint cache first. Failure to fetch weights is an error; the
program never silently substitutes random weights.

For the exact directly used validation versions (Linux CPU):

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
```

The tested versions file is a record of direct dependencies, not a complete
cross-platform lockfile. Let pip resolve compatible torch/torchvision versions on
other platforms. CPU is sufficient; no GPU was used or benchmarked.

## What the algorithm does

1. Resize RGB input to 224×224 and apply ImageNet normalization.
2. Extract frozen ResNet18 layer1 and layer2 features, align their spatial grids,
   concatenate, locally average, and L2-normalize each patch vector.
3. Fit a memory bank from normal references only (4 by default). The bank is a
   deterministic random subset if it exceeds 4,096 vectors. It is **not** an
   optimized PatchCore coreset.
4. Compute exact Euclidean distance from each test patch to its closest normal
   memory vector. Average the largest 1% of distances for the image anomaly score.
5. Smooth and resize the spatial distance map to produce a coarse heatmap. Heatmap
   color uses a fixed normal-calibration scale rather than per-image normalization.
6. Calibrate an image threshold using **separate, known-normal calibration images**.
   At alpha=.1 and n=20, choose order statistic ceil((n+1)×.9)=19.

Fitting estimates normal appearance; **no neural-network fine-tuning** was performed.
The backbone is supervised ImageNet-pretrained ResNet18, not DINOv2. This independent
baseline is inspired by patch-feature anomaly inspection research and does **not**
reproduce AnomalyDINO, PatchCore, or SubspaceAD or claim their published performance.

The calibration rule has a familiar finite-sample normal-score interpretation only
under suitable exchangeability assumptions. Images from the same part are correlated,
and changed illumination violates distribution assumptions. Alpha is **not** a
guarantee that deployed false alarms stay below 10%. Scores are not defect probabilities.

## Fit and inspect your own product

Use one camera view/product category at a time. Put only defect-free images in
`data/my-product/reference` and different defect-free images in
`data/my-product/calibration`. With alpha=.1 use at least 9 calibration images;
20 or more is recommended for an exploratory prototype. More independent physical
parts are better than many nearly identical views of one part.

```bash
python -m inspector.cli fit --normal-dir data/my-product/reference \
  --calibration-dir data/my-product/calibration --shots 4 \
  --normalization none --model outputs/my-product.npz
python -m inspector.cli predict --model outputs/my-product.npz \
  --image data/my-product/query.png --output outputs/query
```

Prediction writes `prediction.json`, `heatmap.npy` (raw distance values), and
`overlay.png`. The portable `.npz` file stores normal embeddings and metadata,
not executable pickle objects or the neural weights. Exact copies across reference
and calibration sets are rejected. The evaluation manifest also checks test overlap.
Content-hash checks cannot detect near duplicates; group-aware collection remains necessary.

## Lighting experiment

The evaluator fits two independent models: raw RGB (`none`) and global luminance
percentile normalization (`percentile`). It applies brightness multipliers 1.0,
0.65, and 1.35 to test images and keeps each model's **clean calibration threshold
fixed**. Test labels never select a threshold. Brightness multiplication clips
values into [0,1]; it is not a simulation of shadows, different light spectra,
exposure noise, reflections, or camera displacement. Camera-position robustness
is not evaluated in this version.

```bash
python -m inspector.cli evaluate --manifest data/my-product/manifest.json \
  --output outputs/my-product-evaluation --factors 1.0 0.65 1.35
```

The manifest schema is shown in `examples/manifest.schema-example.json` and is
generated by the supplied dataset preparation scripts. Reported metrics include
image AUROC/AP, fixed-threshold normal false alarm rate, defect recall, pixel
AUROC/AP, and measured CPU inference latency. Pixel metrics use masks resized
with nearest-neighbor sampling to the model's 224×224 resolution. They are
**not official full-resolution benchmark segmentation scores**.

## Real-data validation: KolektorSDD

Official source: https://www2.vicos.si/resources/kolektorsdd/

Review its CC BY-NC-SA 4.0 terms before use. Source images, labels, real-data models,
and derived real-data overlays are **not bundled**. The delivered project contains
code, synthetic example data/models, selected split metadata, and evaluation results.
Keep dataset restrictions separate from the license on this project's original code.

Download the fine-annotation archive using the official page, then:

```bash
python scripts/prepare_kolektor.py --archive /path/to/KolektorSDD.zip \
  --output data/kolektor --shots 4 --calibration 20 --seed 17
python -m inspector.cli evaluate --manifest data/kolektor/manifest.json \
  --output outputs/kolektor --size 224 --seed 17
```

This uses a **custom physical-part-disjoint split**, not the official supervised
training protocol. Part folders are shuffled using seed 17, 30% assigned to test,
25% to calibration, and the rest to the reference pool. Choose 4 normal reference
images and 20 normal calibration images; evaluate all 119 test images (103 normal,
16 defective). No source physical part crosses partitions. Fixed split IDs and
archive SHA-256 are in `reports/kolektor_manifest.json`.

See `VALIDATION.md` for measured results and caveats. Neither an ideal synthetic
AUROC nor a high pixel AUROC alone demonstrates deployment readiness. Small
defects and a conservative threshold lead to poor recall on this real dataset.

## MVTec AD support

Official data and license: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
(CC BY-NC-SA 4.0). Download and extract one category separately. The preparer takes
references/calibration from `train/good` only, leaving `test` untouched:

```bash
python -m inspector.cli prepare-mvtec --category /datasets/mvtec/bottle \
  --output data/bottle/manifest.json --shots 4 --calibration 20
python -m inspector.cli evaluate --manifest data/bottle/manifest.json \
  --output outputs/bottle
```

MVTec was **not downloaded or evaluated in this run**; no MVTec performance claim
is made. The supplied parser follows the common category/train/test/ground_truth
layout. User-defined grouping must handle any acquisition-level correlations.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests exercise defect-vs-normal scoring, localization signal, calibration order
statistics, split leakage rejection, insufficient calibration, invalid inputs,
normalization behavior, metric accounting, and save/load prediction equivalence.
They use synthetic images and the real pretrained backbone. Tests require cached
weights or network access on first use. They are not a substitute for industrial validation.

## Research grounding

- Damm et al., **AnomalyDINO**, WACV 2025: frozen DINOv2 patch features and nearest
  normal-patch comparison. https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html
- Lendering et al., **SubspaceAD**, 2026: frozen patch features with a PCA normal
  subspace and residual anomaly scores. The present prototype uses nearest-neighbor
  distances instead of their PCA scoring. https://arxiv.org/abs/2602.23013
- Tabernik et al., **Segmentation-Based Deep-Learning Approach for Surface-Defect
  Detection**, Journal of Intelligent Manufacturing: dataset provenance and
  original task. https://www2.vicos.si/resources/kolektorsdd/
- Official torchvision ResNet18 API and weights:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

The project-specific hypothesis is that a simple input normalization may reduce
brightness-induced false alarms at fixed thresholds. Its benefit is assessed, not
assumed; normalization can also erase useful appearance information.

## Limitations and next experiment

- 224×224 resizing distorts aspect ratio and can hide very thin defects. Heatmaps
  are coarse nearest-feature distances, not precise defect boundaries.
- Only one custom real-data split and seed were evaluated. No statistical confidence
  intervals, multiple-site validation, or official leaderboard comparisons are claimed.
- Calibration normal images require manual assurance; contaminated references can
  teach the system to accept defects. Near duplicates are not automatically detected.
- Percentile normalization may amplify noise, suppress contrast clues, and cannot
  recover clipping. It should not be enabled solely on synthetic results.
- A model learned on one product/view should not be reused across unrelated products.
- Next: retain native aspect ratio through tiled inference, compare a frozen DINOv2
  backbone on the same splits, vary reference count across seeds, and evaluate real
  lighting/camera changes with independently collected normal calibration parts.


## Project files

The CLI generates synthetic inputs, fitted demo models, and visual reports locally.
