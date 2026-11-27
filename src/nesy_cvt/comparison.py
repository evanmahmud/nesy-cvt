from __future__ import annotations

import csv
import gc
import json
import math
import pickle
import platform
import random
import shutil
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .benchmark import save_metrics_json
from .config import NesyCVTConfig
from .metrics import (
    accuracy_at_k,
    expected_calibration_error,
    macro_f1,
    macro_precision_recall,
    multiclass_ovr_macro_auc,
)
from .model_factory import MODEL_NAMES, build_classifier, config_for_model
from .profiling import estimate_flops_per_image
from .runtime import resolve_device


def _completed_run_for_resume(
    run_dir: Path,
    *,
    model_name: str,
    seed: int,
    expected_config: dict[str, Any],
    expected_hardware: str,
) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = payload.get("metrics")
    if not isinstance(record, dict):
        raise ValueError(f"Invalid completed-run metadata in {metrics_path}")
    if record.get("model") != model_name or int(record.get("seed", -1)) != seed:
        raise ValueError(f"Resume metadata does not match {model_name}, seed {seed}: {metrics_path}")
    if record.get("hardware") != expected_hardware:
        raise ValueError(f"Resume hardware does not match this run: {metrics_path}")
    actual_config = record.get("config", {})
    mismatches = [
        key
        for key, expected in expected_config.items()
        if actual_config.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            f"Cannot resume {model_name}, seed {seed}; run settings differ for: "
            + ", ".join(mismatches)
        )
    if not (run_dir / "config.json").is_file() or not (run_dir / "pytorch_model.bin").is_file():
        raise ValueError(f"Completed run is missing its model files: {run_dir}")
    record["epochs"] = payload.get("extra", {}).get("epochs", [])
    record["run_dir"] = str(run_dir)
    return record


def _preserve_interrupted_run(run_dir: Path) -> None:
    stamp = time.time_ns()
    preserved = []
    for name in ("progress.json", "pytorch_model.bin", "pytorch_model.bin.tmp"):
        source = run_dir / name
        if source.is_file():
            destination = source.with_name(f"{source.stem}.interrupted-{stamp}{source.suffix}")
            shutil.copy2(source, destination)
            preserved.append(destination.name)
    if preserved:
        print(
            f"Preserved incomplete run files for review in {run_dir}: "
            + ", ".join(preserved),
            flush=True,
        )


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _stratified_indices(targets: list[int], *, val_per_class: int, seed: int):
    import torch

    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[int(target)].append(index)
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_indices in indices_by_class.values():
        rng.shuffle(class_indices)
        val_indices.extend(class_indices[:val_per_class])
        train_indices.extend(class_indices[val_per_class:])
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    train_indices = [train_indices[index] for index in order]
    return train_indices, sorted(val_indices)


class _CoarseTargetDataset:
    """Add CIFAR-100 superclass labels to a torchvision fine-label dataset."""

    def __init__(self, dataset, data_dir: str, *, train: bool):
        self.dataset = dataset
        filename = "train" if train else "test"
        source = Path(data_dir) / dataset.base_folder / filename
        with source.open("rb") as stream:
            payload = pickle.load(stream, encoding="latin1")
        self.coarse_targets = [int(target) for target in payload["coarse_labels"]]
        self.targets = dataset.targets
        if len(self.coarse_targets) != len(self.targets):
            raise ValueError("CIFAR-100 fine and coarse target counts do not match")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        pixels, fine_target = self.dataset[index]
        return pixels, fine_target, self.coarse_targets[index]


def _build_loaders(
    data_dir: str,
    *,
    download: bool,
    batch_size: int,
    val_per_class: int,
    split_seed: int,
    shuffle_seed: int,
    supervised_concepts: bool = False,
    validation_only: bool = False,
    data_recipe: str = "legacy_v1",
):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    if data_recipe == "legacy_v1":
        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
            ]
        )
        eval_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
        train_transform_description = "Resize(32), AutoAugment(CIFAR10 policy), ToTensor"
        eval_transform_description = "Resize(32), ToTensor"
    elif data_recipe == "cifar100_standard_v2":
        normalize = transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [transforms.ToTensor(), normalize]
        )
        train_transform_description = (
            "RandomCrop(32,padding=4), RandomHorizontalFlip, "
            "AutoAugment(CIFAR10 policy), ToTensor, Normalize(CIFAR100 stats)"
        )
        eval_transform_description = "ToTensor, Normalize(CIFAR100 stats)"
    else:
        raise ValueError(f"Unknown CIFAR-100 data recipe: {data_recipe}")
    train_augmented = datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    train_clean = datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=eval_transform
    )
    if supervised_concepts:
        train_augmented = _CoarseTargetDataset(train_augmented, data_dir, train=True)
        train_clean = _CoarseTargetDataset(train_clean, data_dir, train=True)
    test_set = None
    if not validation_only:
        test_set = datasets.CIFAR100(
            root=data_dir, train=False, download=download, transform=eval_transform
        )
        if supervised_concepts:
            test_set = _CoarseTargetDataset(test_set, data_dir, train=False)
    train_indices, val_indices = _stratified_indices(
        train_clean.targets, val_per_class=val_per_class, seed=split_seed
    )
    loader_generator = torch.Generator().manual_seed(shuffle_seed)
    train_loader = DataLoader(
        Subset(train_augmented, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        Subset(train_clean, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = (
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        if test_set is not None
        else None
    )
    return train_loader, val_loader, test_loader, train_transform_description, eval_transform_description


def _measure_single_image_latency(
    model, sample_pixels, device, *, warmup: int = 20, repeats: int = 100
) -> dict[str, float | int]:
    import torch

    image = sample_pixels[:1].to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starts = []
            ends = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(image)
                end.record()
                starts.append(start)
                ends.append(end)
            torch.cuda.synchronize(device)
            timings_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            timings_ms = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(image)
                timings_ms.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings_ms)
    return {
        "single_image_latency_mean_ms": statistics.mean(timings_ms),
        "single_image_latency_p50_ms": statistics.median(timings_ms),
        "single_image_latency_p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "single_image_latency_repeats": repeats,
    }


def _evaluate(
    model,
    loader,
    device,
    *,
    warmup_batches: int = 10,
    reset_peak_memory: bool = True,
    detailed_metrics: bool = True,
) -> dict[str, float | int]:
    import torch
    import torch.nn.functional as F

    model.eval()
    first_batch = next(iter(loader))
    first_pixels = first_batch[0]
    with torch.no_grad():
        for _ in range(warmup_batches):
            model(first_pixels.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        if reset_peak_memory:
            torch.cuda.reset_peak_memory_stats(device)

    all_logits: list[list[float]] = []
    all_labels: list[int] = []
    all_coarse_labels: list[int] = []
    all_coarse_predictions: list[int] = []
    taxonomy_agreements = 0
    taxonomy_examples = 0
    hierarchy_consistency_mae: list[float] = []
    batch_latency_ms: list[float] = []
    inference_seconds_batch = 0.0
    total_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            pixels, labels = batch[:2]
            coarse_labels = batch[2] if len(batch) > 2 else None
            batch_start = time.perf_counter()
            device_pixels = pixels.to(device)
            model_start = time.perf_counter()
            output = model(device_pixels)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds_batch += time.perf_counter() - model_start
            logits = output.logits if hasattr(output, "logits") else output
            if (
                coarse_labels is not None
                and hasattr(output, "concept_logits")
                and getattr(getattr(model, "config", None), "concept_schema", None)
                == "cifar100_coarse20"
            ):
                all_coarse_predictions.extend(
                    output.concept_logits.argmax(dim=-1).detach().cpu().tolist()
                )
                all_coarse_labels.extend(coarse_labels.tolist())
            reasoning_trace = getattr(output, "reasoning_trace", None)
            if reasoning_trace is not None:
                concept_probabilities = reasoning_trace["concept_probabilities"]
                fine_to_coarse = reasoning_trace["fine_to_coarse"]
                neural_probabilities = torch.softmax(reasoning_trace["neural_logits"], dim=-1)
                coarse_from_neural = torch.zeros_like(concept_probabilities).scatter_add(
                    1,
                    fine_to_coarse.unsqueeze(0).expand(pixels.shape[0], -1),
                    neural_probabilities,
                )
                fine_predictions = logits.argmax(dim=-1)
                concept_predictions = concept_probabilities.argmax(dim=-1)
                taxonomy_agreements += int(
                    fine_to_coarse[fine_predictions].eq(concept_predictions).sum().item()
                )
                taxonomy_examples += int(pixels.shape[0])
                hierarchy_consistency_mae.extend(
                    (concept_probabilities - coarse_from_neural)
                    .abs()
                    .mean(dim=-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
            batch_latency_ms.append((time.perf_counter() - batch_start) * 1000)
            all_logits.extend(logits.detach().cpu().tolist())
            all_labels.extend(labels.tolist())
    elapsed = time.perf_counter() - total_start

    logits_tensor = torch.tensor(all_logits, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    probabilities = torch.softmax(logits_tensor, dim=-1)
    confidences, predictions_tensor = probabilities.max(dim=-1)
    predictions = predictions_tensor.tolist()
    correct = predictions_tensor.eq(labels_tensor).tolist()
    scores = accuracy_at_k(all_logits, all_labels, topk=(1, 5))
    if detailed_metrics:
        scores.update(macro_precision_recall(predictions, all_labels, logits_tensor.shape[-1]))
        scores["macro_f1"] = macro_f1(predictions, all_labels, logits_tensor.shape[-1])
        scores["auc_ovr_macro"] = multiclass_ovr_macro_auc(
            probabilities.tolist(), all_labels, logits_tensor.shape[-1]
        )
        scores["ece_15"] = expected_calibration_error(confidences.tolist(), correct, num_bins=15)
        scores["nll"] = float(F.cross_entropy(logits_tensor, labels_tensor).item())
    if all_coarse_labels:
        scores["coarse_top1"] = sum(
            prediction == target
            for prediction, target in zip(all_coarse_predictions, all_coarse_labels)
        ) / len(all_coarse_labels)
    if taxonomy_examples:
        scores["taxonomy_agreement"] = taxonomy_agreements / taxonomy_examples
        scores["hierarchy_consistency_mae"] = statistics.mean(hierarchy_consistency_mae)
    scores["images"] = len(all_labels)
    scores["evaluation_seconds"] = elapsed
    scores["images_per_second"] = len(all_labels) / max(elapsed, 1e-9)
    scores["pipeline_images_per_second"] = scores["images_per_second"]
    if detailed_metrics:
        scores["inference_seconds_batch"] = inference_seconds_batch
        scores["inference_latency_ms_per_image_batch"] = (
            inference_seconds_batch * 1000.0 / max(len(all_labels), 1)
        )
        scores["inference_fps"] = len(all_labels) / max(inference_seconds_batch, 1e-9)
        scores.update(
            _measure_single_image_latency(
                model,
                first_pixels,
                device,
                warmup=max(20, warmup_batches),
                repeats=100,
            )
        )
    scores["batch_latency_mean_ms"] = statistics.mean(batch_latency_ms)
    scores["batch_latency_p50_ms"] = statistics.median(batch_latency_ms)
    ordered_latencies = sorted(batch_latency_ms)
    scores["batch_latency_p95_ms"] = ordered_latencies[math.ceil(0.95 * len(ordered_latencies)) - 1]
    scores["peak_gpu_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    return scores


def _train_one(
    model_name: str,
    seed: int,
    *,
    data_dir: str,
    download: bool,
    output_dir: Path,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_per_class: int,
    warmup_batches: int,
    validation_only: bool = False,
    taxonomy_rule_weight: float = 0.5,
    taxonomy_consistency_weight: float = 0.1,
    concept_loss_weight: float = 0.1,
    data_recipe: str = "legacy_v1",
    optimizer_name: str = "adamw",
    weight_decay: float = 0.01,
    lr_schedule: str = "none",
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch import nn

    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config_for_model(model_name, NesyCVTConfig())
    effective_taxonomy_rule_weight = (
        0.0 if model_name == "nesy_cvt_no_rule_fusion" else taxonomy_rule_weight
    )
    effective_taxonomy_consistency_weight = (
        0.0
        if model_name == "nesy_cvt_no_hierarchy_consistency"
        else taxonomy_consistency_weight
    )
    effective_concept_loss_weight = (
        0.0
        if model_name == "nesy_cvt_no_coarse_supervision"
        else concept_loss_weight
    )
    if config.reasoning_mode == "cifar100_taxonomy":
        config = replace(
            config,
            taxonomy_rule_weight=effective_taxonomy_rule_weight,
            taxonomy_consistency_weight=effective_taxonomy_consistency_weight,
            concept_loss_weight=effective_concept_loss_weight,
        )
    train_loader, val_loader, test_loader, train_transform_description, eval_transform_description = _build_loaders(
        data_dir,
        download=download,
        batch_size=batch_size,
        val_per_class=val_per_class,
        split_seed=1729,
        shuffle_seed=seed,
        supervised_concepts=config.concept_schema == "cifar100_coarse20",
        validation_only=validation_only,
        data_recipe=data_recipe,
    )
    model = build_classifier(model_name, config).to(device)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_schedule == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown learning-rate schedule: {lr_schedule}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    best_val_top1 = -1.0
    best_val_top5 = 0.0
    best_val_coarse_top1: float | None = None
    best_val_taxonomy_agreement: float | None = None
    best_val_hierarchy_consistency_mae: float | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    train_peak_mb = 0.0
    train_start = time.perf_counter()
    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            pixels, labels = batch[:2]
            coarse_labels = batch[2] if len(batch) > 2 else None
            pixels = pixels.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            is_nesy_model = model_name.startswith("nesy_cvt")
            if is_nesy_model:
                model_kwargs: dict[str, Any] = {"labels": labels}
                if coarse_labels is not None:
                    model_kwargs["concept_targets"] = F.one_hot(
                        coarse_labels.to(device), num_classes=config.concept_dim
                    ).to(dtype=torch.float32)
                output = model(pixels, **model_kwargs)
            else:
                output = model(pixels)
            logits = output.logits if is_nesy_model else output
            loss = output.loss if is_nesy_model else F.cross_entropy(logits, labels)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss for {model_name}, seed {seed}")
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            train_peak_mb = max(
                train_peak_mb,
                torch.cuda.max_memory_allocated(device) / (1024**2),
            )

        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            warmup_batches=0,
            reset_peak_memory=False,
            detailed_metrics=False,
        )
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(steps, 1),
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        if "coarse_top1" in val_metrics:
            epoch_result["val_coarse_top1"] = val_metrics["coarse_top1"]
        for metric in ("taxonomy_agreement", "hierarchy_consistency_mae"):
            if metric in val_metrics:
                epoch_result[f"val_{metric}"] = val_metrics[metric]
        history.append(epoch_result)
        print(
            f"{model_name} seed={seed} epoch={epoch + 1}/{epochs} "
            f"train_loss={epoch_result['train_loss']:.4f} "
            f"val_top1={epoch_result['val_top1']:.4f}",
            flush=True,
        )
        if float(val_metrics["top1"]) > best_val_top1:
            best_val_top1 = float(val_metrics["top1"])
            best_val_top5 = float(val_metrics["top5"])
            best_val_coarse_top1 = (
                float(val_metrics["coarse_top1"])
                if "coarse_top1" in val_metrics
                else None
            )
            best_val_taxonomy_agreement = (
                float(val_metrics["taxonomy_agreement"])
                if "taxonomy_agreement" in val_metrics
                else None
            )
            best_val_hierarchy_consistency_mae = (
                float(val_metrics["hierarchy_consistency_mae"])
                if "hierarchy_consistency_mae" in val_metrics
                else None
            )
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            config.save_pretrained(output_dir)
            checkpoint_tmp = output_dir / "pytorch_model.bin.tmp"
            torch.save(best_state, checkpoint_tmp)
            checkpoint_tmp.replace(output_dir / "pytorch_model.bin")
        if scheduler is not None:
            scheduler.step()
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "seed": seed,
                    "completed_epochs": epoch + 1,
                    "best_epoch_by_validation_top1": best_epoch,
                    "best_validation_top1": best_val_top1,
                    "best_validation_top5": best_val_top5,
                    "best_validation_coarse_top1": best_val_coarse_top1,
                    "best_validation_taxonomy_agreement": best_val_taxonomy_agreement,
                    "best_validation_hierarchy_consistency_mae": best_val_hierarchy_consistency_mae,
                    "epochs": history,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_metrics = None
    validation_metrics = None
    if validation_only:
        validation_metrics = _evaluate(
            model, val_loader, device, warmup_batches=warmup_batches
        )
    if test_loader is not None:
        test_metrics = _evaluate(model, test_loader, device, warmup_batches=warmup_batches)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "pytorch_model.bin")
    config.save_pretrained(output_dir)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    checkpoint_bytes = (output_dir / "pytorch_model.bin").stat().st_size
    flops = estimate_flops_per_image(
        model,
        device=device,
        image_size=config.image_size,
        in_channels=config.in_channels,
    )
    run_record: dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "epochs_requested": epochs,
        "best_epoch_by_validation_top1": best_epoch,
        "best_validation_top1": best_val_top1,
        "best_validation_top5": best_val_top5,
        "best_validation_coarse_top1": best_val_coarse_top1,
        "best_validation_taxonomy_agreement": best_val_taxonomy_agreement,
        "best_validation_hierarchy_consistency_mae": best_val_hierarchy_consistency_mae,
        "training_seconds_including_validation": training_seconds,
        "train_images_per_epoch": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "test": test_metrics,
        "num_parameters": num_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint_bytes": checkpoint_bytes,
        "model_size_mb": checkpoint_bytes / 1_000_000.0,
        **flops,
        "model_config": config.to_dict(),
        "train_peak_gpu_memory_mb": train_peak_mb,
        "epochs": history,
        "validation": validation_metrics,
        "config": {
            "dataset": "CIFAR-100 fine labels",
            "input_size": config.image_size,
            "train_transform": train_transform_description,
            "eval_transform": eval_transform_description,
            "batch_size": batch_size,
            "optimizer": optimizer_name.upper(),
            "optimizer_momentum": 0.9 if optimizer_name == "sgd" else None,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "lr_schedule": lr_schedule,
            "grad_clip_norm": grad_clip,
            "data_recipe": data_recipe,
            "epochs": epochs,
            "validation_split_seed": 1729,
            "validation_images_per_class": val_per_class,
            "dataloader_workers": 0,
            "precision": "float32",
            "concept_supervision": (
                "CIFAR-100 coarse labels (20 superclass concepts)"
                if config.concept_schema == "cifar100_coarse20"
                else "none; latent auxiliary class head"
            ),
            "taxonomy_rule_weight": effective_taxonomy_rule_weight,
            "taxonomy_consistency_weight": effective_taxonomy_consistency_weight,
            "concept_loss_weight": effective_concept_loss_weight,
            "test_evaluated": test_metrics is not None,
        },
    }
    return run_record


def run_cifar100_comparison(
    *,
    models: list[str],
    seeds: list[int],
    data_dir: str,
    download: bool,
    runs_dir: str,
    report_path: str,
    device_name: str,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    val_per_class: int = 50,
    warmup_batches: int = 10,
    validation_only: bool = False,
    taxonomy_rule_weight: float = 0.5,
    taxonomy_consistency_weight: float = 0.1,
    concept_loss_weight: float = 0.1,
    data_recipe: str = "legacy_v1",
    optimizer_name: str = "adamw",
    weight_decay: float = 0.01,
    lr_schedule: str = "none",
    grad_clip: float = 1.0,
    resume: bool = False,
) -> dict[str, Any]:
    import torch

    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    if not models or not seeds:
        raise ValueError("At least one model and one seed are required")
    if epochs <= 0 or batch_size <= 0 or val_per_class <= 0:
        raise ValueError("epochs, batch-size, and val-per-class must be positive")
    if (
        taxonomy_rule_weight < 0
        or taxonomy_consistency_weight < 0
        or concept_loss_weight < 0
    ):
        raise ValueError("taxonomy weights must be non-negative")

    device = resolve_device(torch, device_name)
    device_label = str(device)
    if device.type == "cuda":
        device_label = f"{device} ({torch.cuda.get_device_name(device)})"
    records: list[dict[str, Any]] = []
    root = Path(runs_dir)
    train_transform_description = (
        "Resize(32), AutoAugment(CIFAR10 policy), ToTensor"
        if data_recipe == "legacy_v1"
        else "RandomCrop(32,padding=4), RandomHorizontalFlip, "
        "AutoAugment(CIFAR10 policy), ToTensor, Normalize(CIFAR100 stats)"
    )
    eval_transform_description = (
        "Resize(32), ToTensor"
        if data_recipe == "legacy_v1"
        else "ToTensor, Normalize(CIFAR100 stats)"
    )
    for model_name in models:
        for seed in seeds:
            run_dir = root / model_name / f"seed-{seed}"
            if resume:
                effective_rule_weight = (
                    0.0 if model_name == "nesy_cvt_no_rule_fusion" else taxonomy_rule_weight
                )
                effective_consistency_weight = (
                    0.0
                    if model_name == "nesy_cvt_no_hierarchy_consistency"
                    else taxonomy_consistency_weight
                )
                effective_concept_weight = (
                    0.0
                    if model_name == "nesy_cvt_no_coarse_supervision"
                    else concept_loss_weight
                )
                expected_config = {
                    "dataset": "CIFAR-100 fine labels",
                    "input_size": 32,
                    "train_transform": train_transform_description,
                    "eval_transform": eval_transform_description,
                    "batch_size": batch_size,
                    "optimizer": optimizer_name.upper(),
                    "optimizer_momentum": 0.9 if optimizer_name == "sgd" else None,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "lr_schedule": lr_schedule,
                    "grad_clip_norm": grad_clip,
                    "data_recipe": data_recipe,
                    "epochs": epochs,
                    "validation_split_seed": 1729,
                    "validation_images_per_class": val_per_class,
                    "precision": "float32",
                    "taxonomy_rule_weight": effective_rule_weight,
                    "taxonomy_consistency_weight": effective_consistency_weight,
                    "concept_loss_weight": effective_concept_weight,
                    "test_evaluated": not validation_only,
                }
                reused = _completed_run_for_resume(
                    run_dir,
                    model_name=model_name,
                    seed=seed,
                    expected_config=expected_config,
                    expected_hardware=device_label,
                )
                if reused is not None:
                    print(f"Reusing completed run: {model_name}, seed {seed}", flush=True)
                    records.append(reused)
                    continue
                _preserve_interrupted_run(run_dir)
            print(f"Starting {model_name}, seed {seed} on {device_label}", flush=True)
            record = _train_one(
                model_name,
                seed,
                data_dir=data_dir,
                download=download,
                output_dir=run_dir,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_per_class=val_per_class,
                warmup_batches=warmup_batches,
                validation_only=validation_only,
                taxonomy_rule_weight=taxonomy_rule_weight,
                taxonomy_consistency_weight=taxonomy_consistency_weight,
                concept_loss_weight=concept_loss_weight,
                data_recipe=data_recipe,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                lr_schedule=lr_schedule,
                grad_clip=grad_clip,
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

    aggregate_metrics = (
        "top1",
        "top5",
        "macro_f1",
        "ece_15",
        "nll",
        "images_per_second",
        "batch_latency_mean_ms",
        "batch_latency_p50_ms",
        "batch_latency_p95_ms",
        "precision_macro",
        "recall_macro",
        "auc_ovr_macro",
        "inference_latency_ms_per_image_batch",
        "inference_fps",
        "single_image_latency_mean_ms",
        "single_image_latency_p50_ms",
        "single_image_latency_p95_ms",
        "train_peak_gpu_memory_mb",
        "peak_gpu_memory_mb",
        "training_seconds_including_validation",
        "num_parameters",
        "trainable_parameters",
        "model_size_mb",
        "flops_per_image",
        "flops_g_per_image",
        "taxonomy_agreement",
        "hierarchy_consistency_mae",
    )
    aggregate: dict[str, dict[str, dict[str, float | int | None]]] = {}
    report_metrics = aggregate_metrics
    for model_name in models:
        model_records = [record for record in records if record["model"] == model_name]
        aggregate[model_name] = {}
        for metric in report_metrics:
            values = [
                float(record["validation"][metric])
                if validation_only and record.get("validation") is not None and metric in record["validation"]
                else float(record["test"][metric])
                if record.get("test") is not None and metric in record["test"]
                else float(record[metric])
                for record in model_records
                if (validation_only and record.get("validation") is not None and metric in record["validation"])
                or (record.get("test") is not None and metric in record["test"])
                or (metric in record and record[metric] is not None)
            ]
            if not values:
                continue
            aggregate[model_name][metric] = {
                "mean": statistics.mean(values) if values else float("nan"),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
            }

    report: dict[str, Any] = {
        "benchmark_name": "CIFAR-100 controlled comparison",
        "protocol": {
            "dataset": "CIFAR-100 fine labels",
            "train_validation": "stratified 45,000/5,000 split from official 50,000 training images",
            "test": (
                "not evaluated; validation-only mode"
                if validation_only
                else "official 10,000-image test split; evaluated after selecting best validation checkpoint"
            ),
            "validation_only": validation_only,
            "resume_enabled": resume,
            "models": models,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "taxonomy_rule_weight": taxonomy_rule_weight,
            "taxonomy_consistency_weight": taxonomy_consistency_weight,
            "concept_loss_weight": concept_loss_weight,
            "component_ablation_overrides": {
                "nesy_cvt_no_rule_fusion": {"taxonomy_rule_weight": 0.0},
                "nesy_cvt_no_hierarchy_consistency": {
                    "taxonomy_consistency_weight": 0.0
                },
                "nesy_cvt_no_coarse_supervision": {"concept_loss_weight": 0.0},
            },
            "optimizer": optimizer_name.upper(),
            "optimizer_momentum": 0.9 if optimizer_name == "sgd" else None,
            "lr_schedule": lr_schedule,
            "grad_clip_norm": grad_clip,
            "data_recipe": data_recipe,
            "augmentation": (
                "Resize(32), AutoAugment(CIFAR10 policy), ToTensor"
                if data_recipe == "legacy_v1"
                else "RandomCrop(32,pad=4), RandomHorizontalFlip, AutoAugment(CIFAR10 policy), "
                "ToTensor, Normalize(CIFAR100 stats)"
            ),
            "validation_selection": "best validation top-1 checkpoint",
            "class_metrics": "macro-averaged one-vs-rest precision, recall, and F1",
            "auc": "macro one-vs-rest ROC AUC from softmax probabilities",
            "model_size": "serialized pytorch_model.bin checkpoint in decimal MB",
            "inference_timing": (
                "single-image latency is measured at batch size 1; inference FPS is model-only "
                "throughput at the configured batch size; pipeline throughput includes data loading"
            ),
            "flops_method": (
                "analytical estimate for Conv2d, Linear, attention matrix operations, cosine "
                "classifier and approximate FFT; see docs/benchmarking.md for operations omitted"
            ),
            "test_set_history": (
                "The official CIFAR-100 test images are excluded from training, but prior pilot "
                "and pipeline evaluations inspected test results; this is not a blinded holdout."
            ),
        },
        "hardware": device_label,
        "runs": records,
        "aggregate_mean_and_sample_std": aggregate,
        "validation_aggregate_mean_and_sample_std": aggregate if validation_only else None,
    }
    report_file = Path(report_path)
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
                *aggregate_metrics,
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
            for metric in aggregate_metrics:
                source_metrics = record.get("validation") if validation_only else record.get("test")
                row[metric] = (source_metrics or {}).get(metric, record.get(metric))
            writer.writerow(row)
    print(f"Benchmark JSON: {report_file}", flush=True)
    print(f"Benchmark CSV: {csv_path}", flush=True)
    return report
