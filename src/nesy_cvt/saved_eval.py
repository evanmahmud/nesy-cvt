from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from .comparison import _CoarseTargetDataset, _evaluate
from .config import NesyCVTConfig
from .model_factory import build_classifier
from .profiling import estimate_flops_per_image
from .runtime import resolve_device


RUN_PROTOCOL_KEYS = (
    "dataset",
    "input_size",
    "train_transform",
    "eval_transform",
    "batch_size",
    "optimizer",
    "optimizer_momentum",
    "learning_rate",
    "weight_decay",
    "lr_schedule",
    "grad_clip_norm",
    "data_recipe",
    "epochs",
    "validation_split_seed",
    "validation_images_per_class",
    "precision",
)
TEST_METRICS = (
    "top1",
    "top5",
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "auc_ovr_macro",
    "ece_15",
    "nll",
    "inference_latency_ms_per_image_batch",
    "inference_fps",
    "single_image_latency_mean_ms",
    "single_image_latency_p50_ms",
    "single_image_latency_p95_ms",
    "pipeline_images_per_second",
    "peak_gpu_memory_mb",
    "taxonomy_agreement",
    "hierarchy_consistency_mae",
    "coarse_top1",
)
RUN_METRICS = (
    "num_parameters",
    "trainable_parameters",
    "model_size_mb",
    "flops_per_image",
    "flops_g_per_image",
    "train_peak_gpu_memory_mb",
    "training_seconds_including_validation",
)
CSV_FIELDS = (
    "model",
    "seed",
    "best_epoch_by_validation_top1",
    "top1",
    "top5",
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "auc_ovr_macro",
    "num_parameters",
    "trainable_parameters",
    "flops_per_image",
    "flops_g_per_image",
    "checkpoint_bytes",
    "model_size_mb",
    "single_image_latency_mean_ms",
    "single_image_latency_p50_ms",
    "single_image_latency_p95_ms",
    "inference_latency_ms_per_image_batch",
    "inference_fps",
    "pipeline_images_per_second",
    "train_peak_gpu_memory_mb",
    "peak_gpu_memory_mb",
    "training_seconds_including_validation",
    "taxonomy_agreement",
    "hierarchy_consistency_mae",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        config = metrics["config"]
        seed = int(metrics["seed"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid training metadata in {run_dir}") from exc

    # Defaults document the historical runner's actual defaults and preserve
    # compatibility with pre-v2 manifests that did not serialize them.
    config.setdefault("weight_decay", 0.01)
    config.setdefault("optimizer_momentum", None)
    config.setdefault("lr_schedule", "none")
    config.setdefault("grad_clip_norm", 1.0)
    config.setdefault("data_recipe", "legacy_v1")

    checkpoint = run_dir / "pytorch_model.bin"
    config_file = run_dir / "config.json"
    if not checkpoint.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"Expected config.json and pytorch_model.bin in {run_dir}")
    missing = [key for key in RUN_PROTOCOL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Run {run_dir} is missing protocol metadata: {', '.join(missing)}")
    return {
        "run_dir": run_dir,
        "model": str(metrics.get("model", run_dir.parent.name)),
        "seed": seed,
        "metrics": metrics,
        "config": config,
        "checkpoint": checkpoint,
    }


def _test_set_hash(dataset) -> str:
    digest = hashlib.sha256()
    digest.update(dataset.data.tobytes())
    digest.update(json.dumps([int(target) for target in dataset.targets]).encode("ascii"))
    return digest.hexdigest()


def _aggregate_runs(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name in sorted({record["model"] for record in records}):
        model_records = [record for record in records if record["model"] == model_name]
        aggregates[model_name] = {}
        for metric in (*TEST_METRICS, *RUN_METRICS):
            values = [
                float(record["test"][metric])
                if metric in record["test"]
                else float(record[metric])
                for record in model_records
                if metric in record["test"] or record.get(metric) is not None
            ]
            if values:
                aggregates[model_name][metric] = {
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                    "n": len(values),
                }
    return aggregates


def evaluate_saved_checkpoints(
    *,
    runs_dir: str,
    data_dir: str,
    report_path: str,
    device_name: str = "auto",
    batch_size: int | None = None,
    download: bool = False,
    warmup_batches: int = 10,
) -> dict[str, Any]:
    """Re-score completed checkpoints on one common CIFAR-100 test dataset, without training."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from .comparison import _CoarseTargetDataset

    root = Path(runs_dir)
    run_dirs = sorted(
        path
        for model_dir in root.iterdir()
        if model_dir.is_dir()
        for path in model_dir.glob("seed-*")
        if path.is_dir()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No model/seed checkpoint folders found under {root}")
    manifests = [_load_run_metadata(path) for path in run_dirs]

    protocol_signatures = {
        tuple(manifest["config"][key] for key in RUN_PROTOCOL_KEYS) for manifest in manifests
    }
    if len(protocol_signatures) != 1:
        raise ValueError(
            "The selected checkpoints do not share one training/validation protocol. "
            "Use a run folder containing checkpoints trained with the same data split and recipe."
        )
    protocol = dict(zip(RUN_PROTOCOL_KEYS, next(iter(protocol_signatures))))
    if protocol["dataset"] != "CIFAR-100 fine labels":
        raise ValueError(f"Expected CIFAR-100 checkpoints, found {protocol['dataset']!r}")

    seeds_by_model: dict[str, set[int]] = {}
    loss_weights_by_model: dict[str, dict[str, float]] = {}
    weight_keys = (
        "taxonomy_rule_weight",
        "taxonomy_consistency_weight",
        "concept_loss_weight",
    )
    model_weight_signatures: dict[str, set[tuple[float, ...]]] = {}
    for manifest in manifests:
        seeds_by_model.setdefault(manifest["model"], set()).add(manifest["seed"])
        model_name = manifest["model"]
        weight_signature = tuple(
            float(manifest["config"].get(key, 0.0)) for key in weight_keys
        )
        model_weight_signatures.setdefault(model_name, set()).add(weight_signature)
        loss_weights_by_model[model_name] = dict(zip(weight_keys, weight_signature))
    if len({tuple(sorted(seeds)) for seeds in seeds_by_model.values()}) != 1:
        raise ValueError("Each model must have the same seed set for a controlled comparison")
    inconsistent_weights = [
        model_name
        for model_name, signatures in model_weight_signatures.items()
        if len(signatures) != 1
    ]
    if inconsistent_weights:
        raise ValueError(
            "Loss weights must be consistent across seeds for each model: "
            + ", ".join(sorted(inconsistent_weights))
        )
    protocol["component_loss_weights_by_model"] = loss_weights_by_model

    eval_batch_size = int(batch_size or protocol["batch_size"])
    if eval_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = resolve_device(torch, device_name)
    device_label = str(device)
    if device.type == "cuda":
        device_label = f"{device} ({torch.cuda.get_device_name(device)})"

    if protocol["data_recipe"] == "cifar100_standard_v2":
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]
        )
    else:
        eval_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
    test_base = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=eval_transform,
    )
    test_hash = _test_set_hash(test_base)
    records: list[dict[str, Any]] = []

    for manifest in manifests:
        run_dir = manifest["run_dir"]
        model_name = manifest["model"]
        config = NesyCVTConfig.from_pretrained(run_dir)
        model = build_classifier(model_name, config)
        state_dict = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()

        test_dataset = test_base
        if config.concept_schema == "cifar100_coarse20":
            test_dataset = _CoarseTargetDataset(test_base, data_dir, train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(f"Evaluating saved checkpoint: {model_name}, seed {manifest['seed']}", flush=True)
        test_metrics = _evaluate(
            model,
            test_loader,
            device,
            warmup_batches=warmup_batches,
        )
        flops = estimate_flops_per_image(
            model,
            device=device,
            image_size=config.image_size,
            in_channels=config.in_channels,
        )
        checkpoint_bytes = manifest["checkpoint"].stat().st_size
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        old_metrics = manifest["metrics"]
        old_test = old_metrics.get("test") or {}
        if old_test.get("images") not in (None, 10_000):
            raise ValueError(f"Unexpected previous test image count in {run_dir}")

        record = {
            "model": model_name,
            "seed": manifest["seed"],
            "best_epoch_by_validation_top1": old_metrics.get(
                "best_epoch_by_validation_top1"
            ),
            "best_validation_top1": old_metrics.get("best_validation_top1"),
            "training_seconds_including_validation": old_metrics.get(
                "training_seconds_including_validation"
            ),
            "train_peak_gpu_memory_mb": old_metrics.get("train_peak_gpu_memory_mb"),
            "test": test_metrics,
            "num_parameters": parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "checkpoint_bytes": checkpoint_bytes,
            "model_size_mb": checkpoint_bytes / 1_000_000.0,
            "checkpoint_sha256": _file_sha256(manifest["checkpoint"]),
            "run_dir": str(run_dir),
            "flops_breakdown": flops["flops_breakdown"],
            "flops_per_image": flops["flops_per_image"],
            "flops_g_per_image": flops["flops_g_per_image"],
            "flops_method": flops["flops_method"],
            "config": manifest["config"],
            "model_config": config.to_dict(),
        }
        records.append(record)
        del model, state_dict, test_loader, test_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_names = sorted(seeds_by_model)
    seed_values = sorted(next(iter(seeds_by_model.values())))
    report: dict[str, Any] = {
        "benchmark_name": "CIFAR-100 saved-checkpoint detailed comparison",
        "evaluation_only": True,
        "protocol": {
            "dataset": "CIFAR-100 fine labels",
            "train_validation": (
                "same stratified 45,000/5,000 split from the official 50,000-image training "
                f"set; split seed {protocol['validation_split_seed']}, "
                f"{protocol['validation_images_per_class']} validation images per class"
            ),
            "test": "same official CIFAR-100 10,000-image test split for every checkpoint",
            "test_examples": len(test_base),
            "test_data_sha256": test_hash,
            "test_set_history": (
                "Test images were excluded from training and validation, but earlier pilot and "
                "pipeline runs inspected this official test split. It is not a pristine blinded "
                "confirmatory set for this project."
            ),
            "models": model_names,
            "seeds": seed_values,
            "checkpoint_count": len(records),
            "batch_size": eval_batch_size,
            "inference_precision": "float32",
            "flops_method": records[0]["flops_method"],
            "class_metrics": "unweighted macro averages across 100 one-vs-rest classes",
            "auc": "macro one-vs-rest ROC AUC computed from softmax probabilities",
            "inference_timing": (
                "single-image latency is measured on-device at batch size 1; batched inference "
                "FPS is model-only wall time at the reported batch size; pipeline throughput "
                "includes loading the test batches"
            ),
            "common_training_protocol": protocol,
            "component_loss_weights_by_model": loss_weights_by_model,
            "selection_rule": "checkpoint selected by validation top-1 before test evaluation",
            "comparison_scope": (
                "matched dataset, split, input size, and recorded recipe; this is a fixed-recipe "
                "comparison, not a claim that each architecture received its individually optimal recipe"
            ),
        },
        "hardware": device_label,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "runs": records,
        "aggregate_mean_and_sample_std": _aggregate_runs(records),
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = report_file.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in CSV_FIELDS}
            row.update(record["test"])
            writer.writerow(row)
    print(f"Detailed evaluation JSON: {report_file}", flush=True)
    print(f"Detailed evaluation CSV: {csv_path}", flush=True)
    return report
