from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
from pathlib import Path
from typing import Any

from .benchmark import save_metrics_json
from .comparison import MODEL_NAMES, _train_one
from .runtime import resolve_device

AGGREGATE_METRICS = (
    "top1",
    "top5",
    "macro_f1",
    "ece_15",
    "nll",
    "images_per_second",
    "batch_latency_mean_ms",
    "batch_latency_p50_ms",
    "batch_latency_p95_ms",
    "peak_gpu_memory_mb",
    "training_seconds_including_validation",
    "num_parameters",
    "taxonomy_agreement",
    "hierarchy_consistency_mae",
)


def _load_completed_run(
    run_dir: Path,
    *,
    model_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_per_class: int,
    validation_only: bool = False,
    taxonomy_rule_weight: float = 0.5,
    taxonomy_consistency_weight: float = 0.1,
) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        history = payload.get("extra", {}).get("epochs", [])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None

    run_config = metrics.get("config", {})
    evaluation_complete = (
        metrics.get("test") is None
        and metrics.get("config", {}).get("test_evaluated") is False
        if validation_only
        else metrics.get("test", {}).get("images") == 10_000
    )
    is_complete = (
        metrics.get("model") == model_name
        and metrics.get("seed") == seed
        and metrics.get("epochs_requested") == epochs
        and len(history) == epochs
        and metrics.get("best_epoch_by_validation_top1", 0) > 0
        and evaluation_complete
        and run_config.get("batch_size") == batch_size
        and run_config.get("learning_rate") == learning_rate
        and run_config.get("taxonomy_rule_weight", 0.5) == taxonomy_rule_weight
        and run_config.get("taxonomy_consistency_weight", 0.1) == taxonomy_consistency_weight
        and run_config.get("validation_images_per_class") == val_per_class
        and (run_dir / "config.json").is_file()
        and (run_dir / "pytorch_model.bin").is_file()
    )
    if not is_complete:
        return None
    return {**metrics, "epochs": history}


def _aggregate(
    records: list[dict[str, Any]], model_names: list[str], *, validation_only: bool = False
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    metric_names = (
        (
            "best_validation_top1",
            "best_validation_top5",
            "best_validation_coarse_top1",
            "best_validation_taxonomy_agreement",
            "best_validation_hierarchy_consistency_mae",
        )
        if validation_only
        else AGGREGATE_METRICS
    )
    for model_name in model_names:
        model_records = [record for record in records if record["model"] == model_name]
        aggregate[model_name] = {}
        for metric in metric_names:
            values = [
                float(record["test"][metric])
                if record.get("test") is not None and metric in record["test"]
                else float(record[metric])
                for record in model_records
                if (record.get("test") is not None and metric in record["test"])
                or (metric in record and record[metric] is not None)
            ]
            if not values:
                continue
            aggregate[model_name][metric] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
            }
    return aggregate


def run_resumable_comparison(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    if not args.models or not args.seeds:
        raise ValueError("At least one model and one seed are required")
    if args.epochs <= 0 or args.batch_size <= 0 or args.val_per_class <= 0:
        raise ValueError("epochs, batch-size, and val-per-class must be positive")

    device = resolve_device(torch, args.device)
    device_label = str(device)
    if device.type == "cuda":
        device_label = f"{device} ({torch.cuda.get_device_name(device)})"

    root = Path(args.runs_dir)
    records: list[dict[str, Any]] = []
    skipped_runs: list[str] = []
    restarted_runs: list[str] = []
    for model_name in args.models:
        for seed in args.seeds:
            run_dir = root / model_name / f"seed-{seed}"
            completed = _load_completed_run(
                run_dir,
                model_name=model_name,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                val_per_class=args.val_per_class,
                validation_only=args.validation_only,
                taxonomy_rule_weight=args.taxonomy_rule_weight,
                taxonomy_consistency_weight=args.taxonomy_consistency_weight,
            )
            run_key = f"{model_name}/seed-{seed}"
            if completed is not None:
                print(f"Preserving completed run {run_key}", flush=True)
                records.append(completed)
                skipped_runs.append(run_key)
                continue

            if (run_dir / "progress.json").is_file():
                print(f"Restarting incomplete run {run_key} from epoch 1", flush=True)
                restarted_runs.append(run_key)
            else:
                print(f"Starting {run_key}", flush=True)

            record = _train_one(
                model_name,
                seed,
                data_dir=args.data_dir,
                download=args.download,
                output_dir=run_dir,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                val_per_class=args.val_per_class,
                warmup_batches=args.warmup_batches,
                validation_only=args.validation_only,
                taxonomy_rule_weight=args.taxonomy_rule_weight,
                taxonomy_consistency_weight=args.taxonomy_consistency_weight,
            )
            record["hardware"] = device_label
            record["python"] = platform.python_version()
            record["pytorch"] = torch.__version__
            record["torchvision"] = __import__("torchvision").__version__
            record["cuda_runtime"] = torch.version.cuda
            record["platform"] = platform.platform()
            record["cudnn_deterministic"] = (
                torch.backends.cudnn.deterministic if hasattr(torch.backends, "cudnn") else None
            )
            record["run_dir"] = str(run_dir)
            save_metrics_json(
                run_dir / "metrics.json",
                {key: value for key, value in record.items() if key != "epochs"},
                seed=seed,
                hardware=device_label,
                extra={"epochs": record["epochs"]},
            )
            records.append(record)
            del record
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected_runs = len(args.models) * len(args.seeds)
    if len(records) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} complete runs, found {len(records)}")

    report: dict[str, Any] = {
        "benchmark_name": "CIFAR-100 controlled comparison",
        "protocol": {
            "dataset": "CIFAR-100 fine labels",
            "train_validation": "stratified 45,000/5,000 split from official 50,000 training images",
            "test": (
                "not evaluated; validation-only mode"
                if args.validation_only
                else "official 10,000-image test split; evaluated after selecting best validation checkpoint"
            ),
            "validation_only": args.validation_only,
            "models": args.models,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "taxonomy_rule_weight": args.taxonomy_rule_weight,
            "taxonomy_consistency_weight": args.taxonomy_consistency_weight,
            "optimizer": "AdamW",
            "augmentation": "Resize(32), AutoAugment(CIFAR10 policy), ToTensor",
            "validation_selection": "best validation top-1 checkpoint",
        },
        "hardware": device_label,
        "recovery": {
            "resumable_launcher": "python -m nesy_cvt.resume_comparison",
            "preserved_completed_runs": skipped_runs,
            "restarted_incomplete_runs": restarted_runs,
        },
        "runs": records,
        (
            "validation_aggregate_mean_and_sample_std"
            if args.validation_only
            else "aggregate_mean_and_sample_std"
        ): _aggregate(records, args.models, validation_only=args.validation_only),
    }
    report_file = Path(args.report)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = report_file.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "seed",
                "best_epoch_by_validation_top1",
                "best_validation_top1",
                "best_validation_top5",
                "best_validation_coarse_top1",
                "best_validation_taxonomy_agreement",
                "best_validation_hierarchy_consistency_mae",
                *AGGREGATE_METRICS,
            ],
        )
        writer.writeheader()
        for record in records:
            row = {
                "model": record["model"],
                "seed": record["seed"],
                "best_epoch_by_validation_top1": record["best_epoch_by_validation_top1"],
                "best_validation_top1": record.get("best_validation_top1"),
                "best_validation_top5": record.get("best_validation_top5"),
                "best_validation_coarse_top1": record.get("best_validation_coarse_top1"),
                "best_validation_taxonomy_agreement": record.get(
                    "best_validation_taxonomy_agreement"
                ),
                "best_validation_hierarchy_consistency_mae": record.get(
                    "best_validation_hierarchy_consistency_mae"
                ),
            }
            for metric in AGGREGATE_METRICS:
                row[metric] = (record.get("test") or {}).get(metric, record.get(metric))
            writer.writerow(row)
    print(f"Benchmark JSON: {report_file}", flush=True)
    print(f"Benchmark CSV: {csv_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume completed CIFAR-100 runs and restart incomplete model-seed runs."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=[
        "nesy_cvt", "resnet18_cifar", "mobilenet_v3_small", "vit_tiny_patch4"
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--taxonomy-rule-weight", type=float, default=0.5)
    parser.add_argument("--taxonomy-consistency-weight", type=float, default=0.1)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--runs-dir", default="runs/cifar100-comparison")
    parser.add_argument("--report", default="results/cifar100-comparison.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--validation-only", action="store_true")
    run_resumable_comparison(parser.parse_args())


if __name__ == "__main__":
    main()
