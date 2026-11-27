from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import run_cifar100_comparison
from .config import NesyCVTConfig
from .evaluation import EvalConfig, evaluate
from .hf_export import prepare_hf_export, push_to_hub, write_model_card
from .model_factory import MODEL_NAMES
from .training import TrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nesy-cvt", description="NeSy-CVT research CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train", help="Run a small training job")
    train_parser.add_argument("--dataset", choices=["synthetic", "cifar100"], default="synthetic")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--download", action="store_true")
    train_parser.add_argument("--output-dir", default="runs/nesy-cvt-tiny")
    train_parser.add_argument(
        "--model-size",
        choices=["nano", "tiny", "small", "base", "large", "huge"],
        default="tiny",
    )
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--synthetic-samples", type=int, default=256)
    train_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    eval_parser = subcommands.add_parser("eval", help="Evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint-dir", required=True)
    eval_parser.add_argument("--output-path", default="runs/eval/metrics.json")
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    eval_parser.add_argument("--dataset", choices=["synthetic", "cifar100"], default="synthetic")
    eval_parser.add_argument("--data-dir", default="data")
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--download", action="store_true")

    benchmark_parser = subcommands.add_parser(
        "benchmark", help="Train and compare NeSy-CVT against CIFAR-100 reference models"
    )
    benchmark_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["nesy_cvt", "resnet18_cifar", "mobilenet_v3_small", "vit_tiny_patch4"],
    )
    benchmark_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    benchmark_parser.add_argument("--epochs", type=int, default=50)
    benchmark_parser.add_argument("--batch-size", type=int, default=16)
    benchmark_parser.add_argument("--learning-rate", type=float, default=3e-4)
    benchmark_parser.add_argument("--taxonomy-rule-weight", type=float, default=0.5)
    benchmark_parser.add_argument("--taxonomy-consistency-weight", type=float, default=0.1)
    benchmark_parser.add_argument("--concept-loss-weight", type=float, default=0.1)
    benchmark_parser.add_argument(
        "--data-recipe",
        choices=["legacy_v1", "cifar100_standard_v2"],
        default="legacy_v1",
        help="Keep historical transforms or use CIFAR crop/flip/normalization v2.",
    )
    benchmark_parser.add_argument(
        "--optimizer", choices=["adamw", "sgd"], default="adamw"
    )
    benchmark_parser.add_argument("--weight-decay", type=float, default=0.01)
    benchmark_parser.add_argument(
        "--lr-schedule", choices=["none", "cosine"], default="none"
    )
    benchmark_parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Maximum gradient norm; use 0 to disable clipping.",
    )
    benchmark_parser.add_argument("--val-per-class", type=int, default=50)
    benchmark_parser.add_argument("--warmup-batches", type=int, default=10)
    benchmark_parser.add_argument("--data-dir", default="data")
    benchmark_parser.add_argument("--download", action="store_true")
    benchmark_parser.add_argument("--runs-dir", default="runs/cifar100-comparison")
    benchmark_parser.add_argument("--report", default="results/cifar100-comparison.json")
    benchmark_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse finished model/seed runs and preserve partial files before retrying an interrupted run.",
    )
    benchmark_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    benchmark_parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Skip the CIFAR-100 test split; report validation metrics only.",
    )

    benchmark_eval_parser = subcommands.add_parser(
        "benchmark-eval",
        help="Re-evaluate saved benchmark checkpoints on one shared CIFAR-100 test set",
    )
    benchmark_eval_parser.add_argument("--runs-dir", default="runs/cifar100-final-benchmark")
    benchmark_eval_parser.add_argument("--data-dir", default="data")
    benchmark_eval_parser.add_argument("--download", action="store_true")
    benchmark_eval_parser.add_argument("--report", default="results/cifar100-detailed.json")
    benchmark_eval_parser.add_argument("--batch-size", type=int)
    benchmark_eval_parser.add_argument("--warmup-batches", type=int, default=10)
    benchmark_eval_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    benchmark_report_parser = subcommands.add_parser(
        "benchmark-report",
        help="Create comparative CSV/Markdown tables and quality/efficiency plots",
    )
    benchmark_report_parser.add_argument("--input", default="results/cifar100-detailed.json")
    benchmark_report_parser.add_argument(
        "--output-dir", default="results/cifar100-detailed-report"
    )

    benchmark_compare_parser = subcommands.add_parser(
        "benchmark-compare",
        help="Create preserved before/after tables and Plotly figures from two test reports",
    )
    benchmark_compare_parser.add_argument("--before", required=True)
    benchmark_compare_parser.add_argument("--after", required=True)
    benchmark_compare_parser.add_argument(
        "--output-dir", default="results/cifar100-paper-before-after"
    )

    curve_parser = subcommands.add_parser(
        "benchmark-curves",
        help="Plot before/after train and validation curves from progress logs",
    )
    curve_parser.add_argument("--before-runs", required=True)
    curve_parser.add_argument("--after-runs", required=True)
    curve_parser.add_argument(
        "--output-dir", default="results/cifar100-paper-learning-curves"
    )

    export_parser = subcommands.add_parser("export", help="Write config and HF model card template")
    export_parser.add_argument("--checkpoint-dir")
    export_parser.add_argument("--output-dir", default="hf_export/nesy-cvt")
    export_parser.add_argument("--model-name", default="NeSy-CVT")
    export_parser.add_argument("--results-path")

    push_parser = subcommands.add_parser("push-hf", help="Upload an exported folder to Hugging Face")
    push_parser.add_argument("--repo-id", required=True)
    push_parser.add_argument("--local-dir", required=True)
    push_parser.add_argument("--private", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from .model_factory import config_for_model

        metrics = train(
            TrainConfig(
                output_dir=args.output_dir,
                dataset=args.dataset,
                data_dir=args.data_dir,
                download=args.download,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                synthetic_samples=args.synthetic_samples,
                device=args.device,
            ),
            model_config=config_for_model(f"nesy_cvt_{args.model_size}", NesyCVTConfig()),
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0

    if args.command == "eval":
        metrics = evaluate(
            EvalConfig(
                checkpoint_dir=args.checkpoint_dir,
                output_path=args.output_path,
                seed=args.seed,
                device=args.device,
                dataset=args.dataset,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                download=args.download,
            )
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark":
        run_cifar100_comparison(
            models=args.models,
            seeds=args.seeds,
            data_dir=args.data_dir,
            download=args.download,
            runs_dir=args.runs_dir,
            report_path=args.report,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_per_class=args.val_per_class,
            warmup_batches=args.warmup_batches,
            validation_only=args.validation_only,
            taxonomy_rule_weight=args.taxonomy_rule_weight,
            taxonomy_consistency_weight=args.taxonomy_consistency_weight,
            concept_loss_weight=args.concept_loss_weight,
            data_recipe=args.data_recipe,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            lr_schedule=args.lr_schedule,
            grad_clip=args.grad_clip,
            resume=args.resume,
        )
        return 0

    if args.command == "benchmark-eval":
        from .saved_eval import evaluate_saved_checkpoints

        report = evaluate_saved_checkpoints(
            runs_dir=args.runs_dir,
            data_dir=args.data_dir,
            report_path=args.report,
            device_name=args.device,
            batch_size=args.batch_size,
            download=args.download,
            warmup_batches=args.warmup_batches,
        )
        print(
            json.dumps(
                {
                    "models": report["protocol"]["models"],
                    "seeds": report["protocol"]["seeds"],
                    "report": args.report,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark-report":
        from .reporting import create_benchmark_report

        outputs = create_benchmark_report(args.input, args.output_dir)
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark-compare":
        from .reporting import create_before_after_report

        outputs = create_before_after_report(args.before, args.after, args.output_dir)
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark-curves":
        from .reporting import create_training_curve_report

        outputs = create_training_curve_report(
            args.before_runs, args.after_runs, args.output_dir
        )
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0

    if args.command == "export":
        if args.checkpoint_dir:
            output_dir = prepare_hf_export(
                args.checkpoint_dir,
                args.output_dir,
                model_name=args.model_name,
                results_path=args.results_path,
            )
        else:
            config = NesyCVTConfig()
            output_dir = Path(args.output_dir)
            config.save_pretrained(output_dir)
            write_model_card(output_dir, config, model_name=args.model_name)
        print(str(output_dir))
        return 0

    if args.command == "push-hf":
        push_to_hub(args.repo_id, args.local_dir, private=args.private)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
