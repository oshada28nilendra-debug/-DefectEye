"""Behavioral tests using a real pretrained backbone and explicitly synthetic images."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from inspector.core import Config, Inspector, load_rgb, normalize_lighting
from inspector.data import synthetic, read_manifest
from inspector.evaluate import binary_metrics


class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.manifest = synthetic(cls.root / "data")
        _, cls.refs, cls.cal, cls.records = read_manifest(cls.manifest)
        cls.model = Inspector(Config(size=128, max_patches=1024)).fit(cls.refs).calibrate(cls.cal)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_defect_scores_above_normal(self):
        normal = [self.model.predict_array(load_rgb(r["image"]))["score"] for r in self.records[:3]]
        bad = [self.model.predict_array(load_rgb(r["image"]))["score"] for r in self.records[16:19]]
        self.assertGreater(float(np.median(bad)), float(np.median(normal)))

    def test_localization_has_signal_in_annotated_region(self):
        record = self.records[17]
        result = self.model.predict_array(load_rgb(record["image"]))
        mask = np.asarray(Image.open(record["mask"])) > 0
        self.assertGreater(float(result["heatmap"][mask].mean()), float(np.median(result["heatmap"][~mask])))

    def test_calibration_uses_expected_order_statistic(self):
        scores = sorted(self.model.predict_array(load_rgb(p))["score"] for p in self.cal)
        # ceil((12 + 1) * .9) = 12, a conservative threshold.
        self.assertAlmostEqual(self.model.threshold, scores[-1], places=6)

    def test_train_calibration_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.model.calibrate(self.refs)

    def test_manifest_duplicate_across_splits_rejected(self):
        payload = json.loads(self.manifest.read_text())
        payload["calibration"][0] = payload["train"][0]
        path = self.manifest.with_name("leaky.json")
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_manifest(path)

    def test_serialization_preserves_predictions(self):
        path = self.root / "model.npz"
        self.model.save(path)
        loaded = Inspector.load(path)
        rgb = load_rgb(self.records[16]["image"])
        a, b = self.model.predict_array(rgb), loaded.predict_array(rgb)
        np.testing.assert_allclose(a["heatmap"], b["heatmap"], atol=1e-6)
        self.assertAlmostEqual(a["score"], b["score"], places=6)
        self.assertEqual(a["anomalous"], b["anomalous"])

    def test_percentile_normalization_reduces_affine_brightness_difference(self):
        rgb = load_rgb(self.refs[0])
        a = normalize_lighting(rgb, "percentile")
        b = normalize_lighting(rgb * .65, "percentile")
        np.testing.assert_allclose(a, b, atol=2e-6)

    def test_infeasible_calibration_alpha_rejected(self):
        with self.assertRaisesRegex(ValueError, "not enough"):
            self.model.calibrate(self.cal[:2], alpha=0.01)

    def test_invalid_rgb_and_configuration_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict_array(np.full((64, 64, 3), np.nan))
        with self.assertRaises(ValueError):
            Config(size=127).validate()

    def test_metrics_separate_false_alarms_and_recall(self):
        result = binary_metrics([0, 0, 1, 1], [.1, .8, .7, .9], .75)
        self.assertEqual(result["normal_false_alarm_rate"], .5)
        self.assertEqual(result["defect_recall"], .5)


if __name__ == "__main__":
    unittest.main()
