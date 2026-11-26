import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nesy_cvt.benchmark import BenchmarkResult, build_benchmark_tables, save_metrics_json
from nesy_cvt.config import NesyCVTConfig
from nesy_cvt.hf_export import push_to_hub, write_model_card


class BenchmarkAndDocsTests(unittest.TestCase):
    def test_benchmark_tables_keep_controlled_and_literature_results_separate(self):
        controlled = [
            BenchmarkResult(
                model="NeSy-CVT-Tiny",
                dataset="CIFAR-100",
                top1=0.42,
                top5=0.71,
                params_m=5.2,
                flops_g=1.1,
                latency_ms=6.4,
                source="rerun",
            )
        ]
        literature = [
            BenchmarkResult(
                model="EVA-02-L",
                dataset="ImageNet-1K",
                top1=0.90056,
                top5=None,
                params_m=304.0,
                flops_g=None,
                latency_ms=None,
                source="reported",
                reference_url="https://www.codesota.com/benchmark/imagenet-1k",
            )
        ]

        tables = build_benchmark_tables(controlled, literature)

        self.assertEqual(tables["controlled"][0]["source"], "rerun")
        self.assertEqual(tables["literature_sota"][0]["source"], "reported")
        self.assertIn("not_rerun_warning", tables["literature_sota"][0])

    def test_metrics_json_is_written_with_reproducibility_metadata(self):
        payload = {"top1": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"

            save_metrics_json(path, payload, seed=7, hardware="local-cpu")

            data = json.loads(path.read_text())
            self.assertEqual(data["metrics"]["top1"], 0.5)
            self.assertEqual(data["reproducibility"]["seed"], 7)
            self.assertEqual(data["reproducibility"]["hardware"], "local-cpu")

    def test_required_documentation_files_exist(self):
        required = [
            "README.md",
            "docs/architecture.md",
            "docs/mathematics.md",
            "docs/benchmarking.md",
            "docs/reproducibility.md",
            "docs/model_card.md",
            "docs/paper_outline.md",
            "docs/paper_draft.md",
        ]

        missing = [path for path in required if not Path(path).exists()]

        self.assertEqual(missing, [])

    def test_model_card_reads_comparison_run_metadata(self):
        training = {
            "metrics": {
                "model": "nesy_cvt",
                "seed": 42,
                "epochs_requested": 50,
                "best_epoch_by_validation_top1": 47,
                "training_seconds_including_validation": 6400.0,
                "hardware": "cuda (NVIDIA GeForce RTX 5070)",
                "pytorch": "2.13.0+cu132",
                "config": {
                    "dataset": "CIFAR-100 fine labels",
                    "batch_size": 16,
                    "optimizer": "AdamW",
                    "learning_rate": 0.0003,
                    "input_size": 32,
                    "train_transform": "AutoAugment",
                    "seed": 42,
                },
                "test": {
                    "top1": 0.6,
                    "top5": 0.85,
                    "macro_f1": 0.59,
                    "images": 10000,
                },
            },
            "extra": {"epochs": [{"epoch": epoch} for epoch in range(1, 51)]},
        }

        with tempfile.TemporaryDirectory() as tmp:
            card_path = write_model_card(
                tmp,
                NesyCVTConfig(),
                training=training,
                benchmark_summary={
                    "model": "nesy_cvt",
                    "seeds": [42, 43, 44],
                    "aggregate": {
                        "top1": {"mean": 0.6, "sample_std": 0.01, "n": 3},
                        "top5": {"mean": 0.85, "sample_std": 0.005, "n": 3},
                        "macro_f1": {"mean": 0.59, "sample_std": 0.02, "n": 3},
                        "ece_15": {"mean": 0.03, "sample_std": 0.004, "n": 3},
                    },
                },
            )
            card = card_path.read_text(encoding="utf-8")

        self.assertIn("Top-1 accuracy: 60.00%", card)
        self.assertIn("Epochs: 50", card)
        self.assertIn("Batch size: 16", card)
        self.assertIn("selected by validation top-1 at epoch 47", card)
        self.assertIn("60.00% ± 1.00%", card)
        self.assertIn("Aggregate official CIFAR-100 test results", card)
        self.assertIn("neuro-symbolic-ai", card)
        self.assertIn("Nano, Tiny, Small, Base, Large, Huge", card)
        self.assertIn("learned concept-to-class projection", card)

    def test_model_card_describes_taxonomy_reasoning_checkpoint(self):
        config = NesyCVTConfig(
            num_classes=100,
            concept_dim=20,
            concept_schema="cifar100_coarse20",
            reasoning_mode="cifar100_taxonomy",
        )

        with tempfile.TemporaryDirectory() as tmp:
            card = write_model_card(tmp, config).read_text(encoding="utf-8")

        self.assertIn("differentiable CIFAR-100 taxonomy reasoner", card)
        self.assertIn("subfolder=\"base\"", card)
        self.assertIn("reasoning_mode=cifar100_taxonomy", card)

    def test_hf_push_creates_repo_with_visibility_then_uploads(self):
        hub_module = types.ModuleType("huggingface_hub")
        hub_module.create_repo = Mock()
        hub_module.upload_folder = Mock()

        with patch.dict(sys.modules, {"huggingface_hub": hub_module}):
            push_to_hub("example/nesy-cvt", "staging", private=True)

        hub_module.create_repo.assert_called_once_with(
            repo_id="example/nesy-cvt",
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        hub_module.upload_folder.assert_called_once_with(
            repo_id="example/nesy-cvt",
            folder_path="staging",
            repo_type="model",
        )


if __name__ == "__main__":
    unittest.main()
