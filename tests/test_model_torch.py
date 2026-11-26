import importlib.util
import unittest

torch_available = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(torch_available, "PyTorch is required for model tensor tests")
class TorchModelTests(unittest.TestCase):
    def test_spectral_mixer_identity_mode_is_identity_at_zero_gates(self):
        import torch

        from nesy_cvt.model import SpectralMixer

        pixels = torch.randn(2, 4, 8, 8)
        mixer = SpectralMixer(4, mode="residual_identity")

        output, spectral_loss = mixer(pixels)

        self.assertTrue(torch.allclose(output, pixels, atol=1e-5))
        self.assertAlmostEqual(float(spectral_loss.detach()), 0.0, places=6)

    def test_legacy_config_keeps_old_spectral_mixer_behavior(self):
        from nesy_cvt.config import NesyCVTConfig

        restored = NesyCVTConfig.from_dict(
            {"image_size": 32, "num_classes": 10, "embed_dim": 32, "depths": [1], "num_heads": 4}
        )

        self.assertEqual(restored.spectral_mixer_mode, "legacy_double")

    def test_model_forward_returns_logits_concepts_features_and_loss(self):
        import torch

        from nesy_cvt import NesyCVTConfig, NesyCVTForImageClassification

        config = NesyCVTConfig(
            image_size=32,
            num_classes=10,
            embed_dim=32,
            depths=(1,),
            num_heads=4,
            concept_dim=8,
        )
        model = NesyCVTForImageClassification(config)
        pixels = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([1, 3])

        output = model(pixels, labels=labels)

        self.assertEqual(tuple(output.logits.shape), (2, 10))
        self.assertEqual(tuple(output.concept_logits.shape), (2, 8))
        self.assertEqual(tuple(output.features.shape), (2, 32))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

    def test_model_ablation_flags_keep_finite_forward_and_loss(self):
        import torch

        from nesy_cvt import NesyCVTConfig, NesyCVTForImageClassification

        config = NesyCVTConfig(
            image_size=8,
            num_classes=10,
            embed_dim=16,
            depths=(1,),
            num_heads=4,
            concept_dim=8,
            use_spectral_mixer=False,
            use_transformer=False,
            use_concept_auxiliary=False,
            classifier_type="linear",
        )
        model = NesyCVTForImageClassification(config)
        output = model(torch.randn(2, 3, 8, 8), labels=torch.tensor([1, 3]))

        self.assertEqual(tuple(output.logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(output.loss))

    def test_named_size_constructors_return_100_class_logits(self):
        import torch

        from nesy_cvt import (
            nesy_cvt_base,
            nesy_cvt_huge,
            nesy_cvt_large,
            nesy_cvt_nano,
            nesy_cvt_small,
            nesy_cvt_tiny,
        )

        for size, constructor, expected_dim in (
            ("nano", nesy_cvt_nano, 64),
            ("tiny", nesy_cvt_tiny, 96),
            ("small", nesy_cvt_small, 128),
            ("base", nesy_cvt_base, 192),
            ("large", nesy_cvt_large, 256),
            ("huge", nesy_cvt_huge, 320),
        ):
            model = constructor(image_size=8).eval()
            pixels = torch.rand(1, 3, 8, 8)
            labels = torch.tensor([7])
            output = model(pixels, labels=labels)

            self.assertEqual(model.config.model_size, size)
            self.assertEqual(model.config.embed_dim, expected_dim)
            self.assertEqual(model.config.reasoning_mode, "cifar100_taxonomy")
            self.assertEqual(tuple(output.logits.shape), (1, 100))
            self.assertEqual(tuple(output.concept_logits.shape), (1, 20))
            self.assertIsNotNone(output.reasoning_trace)
            self.assertIn("rule", output.loss_components)
            self.assertIn("concept", output.loss_components)
            self.assertTrue(
                torch.allclose(
                    output.logits,
                    output.reasoning_trace["neural_logits"]
                    + model.config.taxonomy_rule_weight
                    * output.reasoning_trace["rule_log_scores"],
                )
            )
            output.loss.backward()
            self.assertIsNotNone(model.concept_head.weight.grad)

    def test_from_pretrained_loads_exported_style_folder(self):
        import tempfile
        from pathlib import Path

        import torch

        from nesy_cvt import NesyCVTForImageClassification, nesy_cvt_tiny

        model = nesy_cvt_tiny(
            num_classes=10,
            image_size=8,
            reasoning_mode="learned_projection",
        ).eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "tiny"
            model.config.save_pretrained(model_dir)
            torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
            restored = NesyCVTForImageClassification.from_pretrained(
                temp_dir, subfolder="tiny"
            ).eval()

        with torch.inference_mode():
            sample = torch.rand(1, 3, 8, 8)
            actual = restored(sample).logits
            expected = model(sample).logits
        self.assertEqual(tuple(actual.shape), (1, 10))
        self.assertEqual(restored.config.model_size, "tiny")
        self.assertTrue(torch.allclose(actual, expected))

    def test_taxonomy_benchmark_reports_reasoning_metrics(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from nesy_cvt.comparison import _evaluate
        from nesy_cvt.model_factory import nesy_cvt_nano

        model = nesy_cvt_nano(image_size=8).eval()
        labels = torch.tensor([0, 1, 2, 3])
        coarse = model.fine_to_coarse[labels]
        loader = DataLoader(
            TensorDataset(torch.rand(4, 3, 8, 8), labels, coarse),
            batch_size=2,
        )

        scores = _evaluate(model, loader, torch.device("cpu"), warmup_batches=0)

        self.assertIn("coarse_top1", scores)
        self.assertIn("taxonomy_agreement", scores)
        self.assertIn("hierarchy_consistency_mae", scores)


if __name__ == "__main__":
    unittest.main()
