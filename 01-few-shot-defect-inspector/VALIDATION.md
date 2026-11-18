# Validation — 17 November 2026

## Executed checks

- 10 behavioral tests passed using synthetic images and the actual ImageNet-pretrained ResNet18 backbone.
- Editable package installation and the `defect-inspector` command succeeded.
- Synthetic demo fit, calibration, lighting evaluation, and single-image prediction completed.
- Real-data evaluation completed on KolektorSDD with a custom physical-part-disjoint split.
- `python -m pip check`: No broken requirements found.
- Python compilation of package, scripts, and tests passed.

## Real industrial evaluation

4 normal references; 20 separate normal calibration images; 119 test images (103 normal, 16 defective).
One seed (17), 224×224 model input, CPU only. These are custom-subset results, not official benchmark scores.

| Input | Brightness | Image AUROC | False alarms / normal | Detected / defective | Pixel AUROC | Pixel AP | Median inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1.00 | 0.7870 | 9/103 (8.7%) | 5/16 (31.2%) | 0.8141 | 0.0280 | 49.8 ms |
| none | 0.65 | 0.7215 | 39/103 (37.9%) | 11/16 (68.8%) | 0.8084 | 0.0099 | 52.3 ms |
| none | 1.35 | 0.6299 | 102/103 (99.0%) | 16/16 (100.0%) | 0.6206 | 0.0012 | 47.0 ms |
| percentile | 1.00 | 0.5273 | 9/103 (8.7%) | 2/16 (12.5%) | 0.7594 | 0.0028 | 55.7 ms |
| percentile | 0.65 | 0.5273 | 9/103 (8.7%) | 2/16 (12.5%) | 0.7593 | 0.0028 | 54.6 ms |
| percentile | 1.35 | 0.5850 | 99/103 (96.1%) | 16/16 (100.0%) | 0.5987 | 0.0009 | 57.8 ms |

### Interpretation

The raw baseline ranks clean normal/defective images moderately (AUROC 0.7870), but at its normal-only calibrated threshold it detects only 5 of 16 defects and falsely flags 9 of 103 normals. It is not ready for production inspection.

Brightness changes produce severe threshold drift. At ×1.35 brightness, the raw model flags 102 of 103 normal images. Its perfect defect recall in that condition is not useful detection performance.

Percentile normalization stabilizes the darker condition but damages clean discrimination (AUROC 0.5273, recall 2/16) and does not solve clipping in the brighter condition (99/103 false alarms). Do not enable it by default based on the easy synthetic demonstration.

Pixel AUROC alone is misleading for tiny defects: clean raw pixel AP is only 0.0280. All pixel metrics use 224×224 masks, not native-resolution official evaluation.

## Synthetic smoke test

4 references, 12 normal calibration images, 32 test images (16 normal, 16 procedural defects).
Both variants rank these easy synthetic images perfectly, while raw brightness-shift thresholds produce false alarms. This confirms pipeline behavior only; it is not evidence of real-world accuracy.

## Reproducibility and scope

- `reports/kolektor_manifest.json` records exact source part partitions, filenames, seed, and official archive SHA-256.
- `reports/*/results.json` records raw aggregate metrics, environment versions, and measured timing.
- `reports/*/predictions.csv` records per-image labels, scores, thresholds, and decisions.
- Reference/calibration/test decoded-image hashes are checked for exact overlap; source physical-part IDs are separated during KSDD preparation.
- No test label is used to fit either model or calibrate thresholds; both specified input variants are reported.
- No neural-network training, DINOv2 benchmark reproduction, MVTec evaluation, camera-position experiment, multi-seed study, or confidence interval was performed.
- Global brightness scaling is a deterministic digital stress test, not a real acquisition experiment.
- Runtime results depend on hardware and thread settings; timings exclude image file decoding and include feature extraction and anomaly scoring.
- Downloaded weights are about 45 MB. Each bundled synthetic patch-memory model is about 2 MB, excluding PyTorch and its runtime.

## Actual test log

```text
test_calibration_uses_expected_order_statistic (test_inspector.InspectorTests.test_calibration_uses_expected_order_statistic) ... ok
test_defect_scores_above_normal (test_inspector.InspectorTests.test_defect_scores_above_normal) ... ok
test_infeasible_calibration_alpha_rejected (test_inspector.InspectorTests.test_infeasible_calibration_alpha_rejected) ... ok
test_invalid_rgb_and_configuration_rejected (test_inspector.InspectorTests.test_invalid_rgb_and_configuration_rejected) ... ok
test_localization_has_signal_in_annotated_region (test_inspector.InspectorTests.test_localization_has_signal_in_annotated_region) ... ok
test_manifest_duplicate_across_splits_rejected (test_inspector.InspectorTests.test_manifest_duplicate_across_splits_rejected) ... ok
test_metrics_separate_false_alarms_and_recall (test_inspector.InspectorTests.test_metrics_separate_false_alarms_and_recall) ... ok
test_percentile_normalization_reduces_affine_brightness_difference (test_inspector.InspectorTests.test_percentile_normalization_reduces_affine_brightness_difference) ... ok
test_serialization_preserves_predictions (test_inspector.InspectorTests.test_serialization_preserves_predictions) ... ok
test_train_calibration_leakage_rejected (test_inspector.InspectorTests.test_train_calibration_leakage_rejected) ... ok

----------------------------------------------------------------------
Ran 10 tests in 7.470s

OK
```

