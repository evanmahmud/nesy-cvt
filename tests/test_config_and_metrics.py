import json
import math
import unittest

from nesy_cvt import NesyCVTConfig
from nesy_cvt.metrics import accuracy_at_k, expected_calibration_error, macro_f1


class ConfigAndMetricsTests(unittest.TestCase):
    def test_config_round_trips_through_json(self):
        config = NesyCVTConfig(
            image_size=32,
            num_classes=100,
            embed_dim=64,
            depths=(1, 2),
            num_heads=4,
            concept_dim=16,
        )

        restored = NesyCVTConfig.from_json(config.to_json())

        self.assertEqual(restored.image_size, 32)
        self.assertEqual(restored.num_classes, 100)
        self.assertEqual(restored.embed_dim, 64)
        self.assertEqual(restored.depths, (1, 2))
        self.assertEqual(restored.num_heads, 4)
        self.assertEqual(json.loads(restored.to_json())["model_type"], "nesy-cvt")

    def test_accuracy_top_k_counts_correct_predictions(self):
        logits = [
            [0.1, 0.8, 0.2],
            [0.7, 0.1, 0.2],
            [0.2, 0.3, 0.5],
        ]
        labels = [1, 2, 0]

        scores = accuracy_at_k(logits, labels, topk=(1, 2))

        self.assertAlmostEqual(scores["top1"], 1 / 3)
        self.assertAlmostEqual(scores["top2"], 2 / 3)

    def test_macro_f1_averages_per_class_f1(self):
        predictions = [0, 1, 1, 2]
        labels = [0, 1, 2, 2]

        score = macro_f1(predictions, labels, num_classes=3)

        self.assertAlmostEqual(score, (1.0 + (2 / 3) + (2 / 3)) / 3)

    def test_expected_calibration_error_uses_confidence_bins(self):
        confidences = [0.95, 0.8, 0.4, 0.2]
        correct = [True, True, False, True]

        score = expected_calibration_error(confidences, correct, num_bins=2)

        self.assertTrue(math.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
