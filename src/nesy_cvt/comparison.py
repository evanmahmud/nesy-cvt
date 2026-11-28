from __future__ import annotations

import csv
import gc
import hashlib
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
from datetime import datetime, timezone
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


def _stratified_fold_indices(
    targets: list[int], *, folds: int, fold_index: int, seed: int
) -> tuple[list[int], list[int]]:
    """Return one deterministic stratified fold and its complementary train indices."""
    import torch

    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if not 0 <= fold_index < folds:
        raise ValueError(f"fold_index must be in [0, {folds - 1}]")
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[int(target)].append(index)
    if any(len(indices) < folds for indices in indices_by_class.values()):
        raise ValueError("Each class must have at least `folds` examples")

    rng = random.Random(seed)
    validation_folds: list[list[int]] = [[] for _ in range(folds)]
    for class_indices in indices_by_class.values():
        rng.shuffle(class_indices)
        quotient, remainder = divmod(len(class_indices), folds)
        offset = 0
        for current_fold in range(folds):
            fold_size = quotient + (1 if current_fold < remainder else 0)
            validation_folds[current_fold].extend(
                class_indices[offset : offset + fold_size]
            )
            offset += fold_size

    # Audit the partition itself: every training example must appear in exactly
    # one validation fold before any model is trained.
    flattened_validation = [index for fold in validation_folds for index in fold]
    if (
        len(flattened_validation) != len(targets)
        or len(set(flattened_validation)) != len(targets)
        or set(flattened_validation) != set(range(len(targets)))
    ):
        raise RuntimeError("Stratified folds do not form a disjoint partition of training data")

    val_indices = sorted(validation_folds[fold_index])
    val_index_set = set(val_indices)
    train_indices = [index for index in range(len(targets)) if index not in val_index_set]
    if val_index_set.intersection(train_indices):
        raise RuntimeError("Train/validation index overlap detected")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    train_indices = [train_indices[index] for index in order]
    return train_indices, sorted(val_indices)


def _indices_sha256(indices: list[int]) -> str:
    canonical = json.dumps(sorted(int(index) for index in indices), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


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
    folds: int = 1,
    fold_index: int = 0,
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
    if folds == 1:
        train_indices, val_indices = _stratified_indices(
            train_clean.targets, val_per_class=val_per_class, seed=split_seed
        )
    else:
        train_indices, val_indices = _stratified_fold_indices(
            train_clean.targets,
            folds=folds,
            fold_index=fold_index,
            seed=split_seed,
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
    train_index_set = set(train_indices)
    val_index_set = set(val_indices)
    overlap_count = len(train_index_set.intersection(val_index_set))
    covered_count = len(train_index_set.union(val_index_set))
    if overlap_count != 0 or covered_count != len(train_clean):
        raise RuntimeError(
            f"Invalid data split: overlap={overlap_count}, covered={covered_count}, "
            f"dataset={len(train_clean)}"
        )
    split_audit = {
        "train_images": len(train_indices),
        "validation_images": len(val_indices),
        "train_validation_overlap_count": overlap_count,
        "training_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(val_indices),
        "fold_partition_audit_passed": True if folds > 1 else None,
        "official_test_dataset_loaded_for_post_training_evaluation": test_set is not None,
    }
    return (
        train_loader,
        val_loader,
        test_loader,
        train_transform_description,
        eval_transform_description,
        split_audit,
    )


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
    folds: int = 1,
    fold_index: int = 0,
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

    run_wall_start = time.perf_counter()
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
    (
        train_loader,
        val_loader,
        test_loader,
        train_transform_description,
        eval_transform_description,
        split_audit,
    ) = _build_loaders(
        data_dir,
        download=download,
        batch_size=batch_size,
        val_per_class=val_per_class,
        split_seed=1729,
        shuffle_seed=seed,
        folds=folds,
        fold_index=fold_index,
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
        gpu_device_index = (
            int(device.index) if device.index is not None else int(torch.cuda.current_device())
        )
        gpu_visible_device_count = int(torch.cuda.device_count())
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_memory_mb = (
            torch.cuda.get_device_properties(device).total_memory / (1024**2)
        )
    else:
        gpu_device_index = None
        gpu_visible_device_count = 0
        gpu_name = None
        gpu_total_memory_mb = None

    best_val_top1 = -1.0
    best_val_top5 = 0.0
    best_val_coarse_top1: float | None = None
    best_val_taxonomy_agreement: float | None = None
    best_val_hierarchy_consistency_mae: float | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    train_peak_mb = 0.0
    train_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_wall_start = time.perf_counter()
        epoch_start_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_train_start = time.perf_counter()
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
        epoch_training_seconds = time.perf_counter() - epoch_train_start
        epoch_peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        epoch_peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        train_peak_mb = max(train_peak_mb, epoch_peak_allocated_mb)

        validation_start = time.perf_counter()
        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            warmup_batches=0,
            reset_peak_memory=False,
            detailed_metrics=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_validation_seconds = time.perf_counter() - validation_start
        epoch_end_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(steps, 1),
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "epoch_start_utc": epoch_start_utc,
            "epoch_end_utc": epoch_end_utc,
            "epoch_training_seconds": epoch_training_seconds,
            "epoch_validation_seconds": epoch_validation_seconds,
            "epoch_total_seconds": time.perf_counter() - epoch_wall_start,
            "gpu_device_index": gpu_device_index,
            "gpu_visible_device_count": gpu_visible_device_count,
            "gpu_name": gpu_name,
            "gpu_total_memory_mb": gpu_total_memory_mb,
            "gpu_peak_allocated_mb": epoch_peak_allocated_mb,
            "gpu_peak_reserved_mb": epoch_peak_reserved_mb,
        }
        if "coarse_top1" in val_metrics:
            epoch_result["val_coarse_top1"] = val_metrics["coarse_top1"]
        for metric in ("taxonomy_agreement", "hierarchy_consistency_mae"):
            if metric in val_metrics:
                epoch_result[f"val_{metric}"] = val_metrics[metric]
        history.append(epoch_result)
        print(
            f"{model_name} fold={fold_index + 1}/{folds} seed={seed} epoch={epoch + 1}/{epochs} "
            f"train_loss={epoch_result['train_loss']:.4f} "
            f"val_top1={epoch_result['val_top1']:.4f} "
            f"epoch_seconds={epoch_result['epoch_total_seconds']:.2f} "
            f"gpu_peak_allocated_mb={epoch_peak_allocated_mb:.1f} "
            f"gpu_peak_reserved_mb={epoch_peak_reserved_mb:.1f}",
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
                    "fold_index": fold_index + 1,
                    "folds": folds,
                    "data_split_audit": split_audit,
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
        "fold_index": fold_index + 1,
        "folds": folds,
        "epochs_requested": epochs,
        "best_epoch_by_validation_top1": best_epoch,
        "best_validation_top1": best_val_top1,
        "best_validation_top5": best_val_top5,
        "best_validation_coarse_top1": best_val_coarse_top1,
        "best_validation_taxonomy_agreement": best_val_taxonomy_agreement,
        "best_validation_hierarchy_consistency_mae": best_val_hierarchy_consistency_mae,
        "training_seconds_including_validation": training_seconds,
        "run_wall_seconds": time.perf_counter() - run_wall_start,
        "train_images_per_epoch": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "data_split_audit": split_audit,
        "gpu_device_index": gpu_device_index,
        "gpu_visible_device_count": gpu_visible_device_count,
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_memory_mb,
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
            "split_audit_version": "sha256_indices_v1",
            "validation_split_strategy": (
                "fixed_stratified_holdout" if folds == 1 else "stratified_k_fold"
            ),
            "folds": folds,
            "fold_index": fold_index + 1,
            "validation_images_per_class": val_per_class if folds == 1 else None,
            "training_indices_sha256": split_audit["training_indices_sha256"],
            "validation_indices_sha256": split_audit["validation_indices_sha256"],
            "train_validation_overlap_count": split_audit[
                "train_validation_overlap_count"
            ],
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


def _write_timing_artifacts(
    records: list[dict[str, Any]], output_dir: Path, *, benchmark_wall_seconds: float
) -> dict[str, Any]:
    """Write per-epoch and per-model timing/VRAM summaries for a benchmark run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = output_dir / "epoch_timing.csv"
    epoch_fields = [
        "model",
        "fold_index",
        "seed",
        "epoch",
        "epoch_start_utc",
        "epoch_end_utc",
        "epoch_training_seconds",
        "epoch_validation_seconds",
        "epoch_total_seconds",
        "gpu_device_index",
        "gpu_visible_device_count",
        "gpu_name",
        "gpu_total_memory_mb",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
    ]
    epoch_rows: list[dict[str, Any]] = []
    for record in records:
        for epoch_record in record.get("epochs", []):
            epoch_rows.append(
                {
                    "model": record["model"],
                    "fold_index": record.get("fold_index", 1),
                    "seed": record["seed"],
                    **{field: epoch_record.get(field) for field in epoch_fields[3:]},
                }
            )
    with epoch_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=epoch_fields)
        writer.writeheader()
        writer.writerows(epoch_rows)

    summary_fields = [
        "scope",
        "model",
        "seed",
        "fold_count",
        "fit_count",
        "epoch_count",
        "epoch_training_seconds",
        "epoch_validation_seconds",
        "sum_model_fold_seed_wall_seconds",
        "mean_epoch_total_seconds",
        "peak_gpu_allocated_mb",
        "peak_gpu_reserved_mb",
        "gpu_device_index",
        "gpu_visible_device_count",
        "gpu_name",
        "benchmark_wall_seconds_through_aggregation",
    ]
    groups: list[tuple[str, str, int | str, list[dict[str, Any]]]] = []
    for model_name in sorted({str(record["model"]) for record in records}):
        model_records = [record for record in records if record["model"] == model_name]
        for seed in sorted({int(record["seed"]) for record in model_records}):
            groups.append(
                (
                    "model_seed",
                    model_name,
                    seed,
                    [record for record in model_records if int(record["seed"]) == seed],
                )
            )
        groups.append(("model_total", model_name, "", model_records))
    groups.append(("overall", "__all__", "", records))

    summary_rows: list[dict[str, Any]] = []
    for scope, model_name, seed, group_records in groups:
        epoch_records = [
            epoch_record
            for record in group_records
            for epoch_record in record.get("epochs", [])
        ]

        def _sum_epoch_field(field: str) -> float:
            return sum(float(item.get(field, 0.0) or 0.0) for item in epoch_records)

        summary_rows.append(
            {
                "scope": scope,
                "model": model_name,
                "seed": seed,
                "fold_count": len({int(record.get("fold_index", 1)) for record in group_records}),
                "fit_count": len(group_records),
                "epoch_count": len(epoch_records),
                "epoch_training_seconds": _sum_epoch_field("epoch_training_seconds"),
                "epoch_validation_seconds": _sum_epoch_field("epoch_validation_seconds"),
                "sum_model_fold_seed_wall_seconds": sum(
                    float(record.get("run_wall_seconds", 0.0) or 0.0)
                    for record in group_records
                ),
                "mean_epoch_total_seconds": (
                    statistics.mean(
                        float(item.get("epoch_total_seconds", 0.0) or 0.0)
                        for item in epoch_records
                    )
                    if epoch_records
                    else 0.0
                ),
                "peak_gpu_allocated_mb": max(
                    (float(item.get("gpu_peak_allocated_mb", 0.0) or 0.0) for item in epoch_records),
                    default=0.0,
                ),
                "peak_gpu_reserved_mb": max(
                    (float(item.get("gpu_peak_reserved_mb", 0.0) or 0.0) for item in epoch_records),
                    default=0.0,
                ),
                "gpu_device_index": next(
                    (record.get("gpu_device_index") for record in group_records if record.get("gpu_device_index") is not None),
                    None,
                ),
                "gpu_visible_device_count": max(
                    (int(record.get("gpu_visible_device_count", 0) or 0) for record in group_records),
                    default=0,
                ),
                "gpu_name": next(
                    (record.get("gpu_name") for record in group_records if record.get("gpu_name")),
                    None,
                ),
                "benchmark_wall_seconds_through_aggregation": (
                    benchmark_wall_seconds if scope == "overall" else None
                ),
            }
        )
    summary_path = output_dir / "timing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    visible_gpu_counts = {
        int(record.get("gpu_visible_device_count", 0) or 0) for record in records
    }
    runtime_summary = {
        "gpu_devices_used_by_runner": 1 if any(visible_gpu_counts) else 0,
        "gpu_visible_device_count": max(visible_gpu_counts, default=0),
        "gpu_device_index_used": next(
            (record.get("gpu_device_index") for record in records if record.get("gpu_device_index") is not None),
            None,
        ),
        "gpu_name": next(
            (record.get("gpu_name") for record in records if record.get("gpu_name")), None
        ),
        "model_fold_seed_fit_count": len(records),
        "completed_epoch_count": len(epoch_rows),
        "sum_epoch_training_seconds": sum(
            float(item.get("epoch_training_seconds", 0.0) or 0.0)
            for item in epoch_rows
        ),
        "sum_epoch_validation_seconds": sum(
            float(item.get("epoch_validation_seconds", 0.0) or 0.0)
            for item in epoch_rows
        ),
        "sum_model_fold_seed_wall_seconds": sum(
            float(record.get("run_wall_seconds", 0.0) or 0.0) for record in records
        ),
        "benchmark_wall_seconds_through_aggregation": benchmark_wall_seconds,
        "epoch_timing_csv": str(epoch_path),
        "timing_summary_csv": str(summary_path),
    }
    return runtime_summary


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
    folds: int = 5,
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

    benchmark_start = time.perf_counter()
    benchmark_start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    if not models or not seeds:
        raise ValueError("At least one model and one seed are required")
    if epochs <= 0 or batch_size <= 0 or val_per_class <= 0 or folds <= 0:
        raise ValueError("epochs, batch-size, val-per-class, and folds must be positive")
    if (
        taxonomy_rule_weight < 0
        or taxonomy_consistency_weight < 0
        or concept_loss_weight < 0
    ):
        raise ValueError("taxonomy weights must be non-negative")

    device = resolve_device(torch, device_name)
    device_label = str(device)
    visible_gpu_count = 0
    gpu_index_used = None
    if device.type == "cuda":
        device_label = f"{device} ({torch.cuda.get_device_name(device)})"
        visible_gpu_count = int(torch.cuda.device_count())
        gpu_index_used = (
            int(device.index) if device.index is not None else int(torch.cuda.current_device())
        )
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
        for fold_index in range(folds):
            for seed in seeds:
                run_dir = root / model_name
                if folds > 1:
                    run_dir = run_dir / f"fold-{fold_index + 1}"
                run_dir = run_dir / f"seed-{seed}"
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
                        "split_audit_version": "sha256_indices_v1",
                        "validation_split_strategy": (
                            "fixed_stratified_holdout" if folds == 1 else "stratified_k_fold"
                        ),
                        "folds": folds,
                        "fold_index": fold_index + 1,
                        "validation_images_per_class": val_per_class if folds == 1 else None,
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
                        print(
                            f"Reusing completed run: {model_name}, fold {fold_index + 1}/{folds}, seed {seed}",
                            flush=True,
                        )
                        records.append(reused)
                        continue
                    _preserve_interrupted_run(run_dir)
                print(
                    f"Starting {model_name}, fold {fold_index + 1}/{folds}, seed {seed} on {device_label}",
                    flush=True,
                )
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
                    folds=folds,
                    fold_index=fold_index,
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

    split_hashes_by_fold: dict[int, dict[str, str]] = {}
    for record in records:
        audit = record.get("data_split_audit") or {}
        if int(audit.get("train_validation_overlap_count", -1)) != 0:
            raise RuntimeError(
                f"Data leakage audit failed for {record['model']} fold {record.get('fold_index')}: "
                "train/validation indices overlap"
            )
        fold_id = int(record.get("fold_index", 1))
        split_signature = {
            "training_indices_sha256": str(audit.get("training_indices_sha256", "")),
            "validation_indices_sha256": str(audit.get("validation_indices_sha256", "")),
        }
        if not all(split_signature.values()):
            raise RuntimeError(
                f"Data leakage audit hashes are missing for {record['model']} fold {fold_id}"
            )
        previous_signature = split_hashes_by_fold.setdefault(fold_id, split_signature)
        if previous_signature != split_signature:
            raise RuntimeError(
                f"Models/seeds do not share identical data indices in fold {fold_id}"
            )
    if set(split_hashes_by_fold) != set(range(1, folds + 1)):
        raise RuntimeError("The benchmark is missing one or more requested validation folds")
    if folds > 1 and len(
        {signature["validation_indices_sha256"] for signature in split_hashes_by_fold.values()}
    ) != folds:
        raise RuntimeError("Different CV folds unexpectedly share identical validation indices")

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
            fold_values_by_seed: dict[int, list[float]] = defaultdict(list)
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
            for record in model_records:
                if (
                    validation_only
                    and record.get("validation") is not None
                    and metric in record["validation"]
                ):
                    value = record["validation"][metric]
                elif record.get("test") is not None and metric in record["test"]:
                    value = record["test"][metric]
                else:
                    value = record.get(metric)
                if value is not None:
                    fold_values_by_seed[int(record["seed"])].append(float(value))
            if not fold_values_by_seed:
                continue
            seed_values = []
            for fold_values in fold_values_by_seed.values():
                if metric == "training_seconds_including_validation":
                    seed_values.append(sum(fold_values))
                elif metric in {"train_peak_gpu_memory_mb", "peak_gpu_memory_mb"}:
                    seed_values.append(max(fold_values))
                else:
                    seed_values.append(statistics.mean(fold_values))
            aggregate[model_name][metric] = {
                "mean": statistics.mean(seed_values),
                "sample_std": statistics.stdev(seed_values) if len(seed_values) > 1 else None,
                "n": len(seed_values),
                "fold_observations": len(values),
                "fold_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "folds": folds,
            }

    runtime_summary = _write_timing_artifacts(
        records,
        Path(report_path).parent,
        benchmark_wall_seconds=time.perf_counter() - benchmark_start,
    )
    runtime_summary.update(
        {
            "benchmark_started_at_utc": benchmark_start_utc,
            "models_requested": len(models),
            "seeds_requested": len(seeds),
            "folds_requested": folds,
            "epochs_requested_per_fit": epochs,
            "gpu_devices_used_by_runner": 1 if device.type == "cuda" else 0,
            "gpu_visible_device_count": visible_gpu_count,
            "gpu_device_index_used": gpu_index_used,
        }
    )
    data_leakage_audit = {
        "train_validation_overlap_count_per_fit": 0,
        "train_and_validation_cover_training_set_per_fit": True,
        "each_training_example_is_validation_once_across_folds": folds > 1,
        "same_index_hashes_across_all_models_and_seeds_per_fold": True,
        "fold_partition_audit_passed_for_every_fit": folds > 1,
        "official_test_dataset_loaded_for_post_training_evaluation": not validation_only,
        "official_test_used_for_training_or_checkpoint_selection": False,
        "official_test_used_for_gradient_updates_or_checkpoint_selection": False,
        "official_test_had_prior_project_exposure": True,
        "interpretation": (
            "No direct train/validation overlap was detected within any fit. "
            "The official test set is not a pristine confirmatory holdout because "
            "earlier project runs inspected it; validation results are primary."
        ),
    }
    report: dict[str, Any] = {
        "benchmark_name": "CIFAR-100 controlled comparison",
        "protocol": {
            "dataset": "CIFAR-100 fine labels",
            "train_validation": (
                "legacy fixed stratified 45,000/5,000 holdout from official 50,000 training images"
                if folds == 1
                else f"stratified {folds}-fold cross-validation over all 50,000 official training images; each fold trains on approximately {(folds - 1) / folds:.0%} and validates on {1 / folds:.0%}"
            ),
            "validation_split_strategy": (
                "fixed_stratified_holdout" if folds == 1 else "stratified_k_fold"
            ),
            "folds": folds,
            "validation_split_seed": 1729,
            "fold_assignment": "identical deterministic fine-label stratified folds for every model and seed",
            "split_index_sha256_by_fold": split_hashes_by_fold,
            "data_leakage_audit": data_leakage_audit,
            "validation_aggregation": "average quality and timing metrics within each seed across folds; sum training time and take peak memory within each seed across folds; then report mean and sample SD across seeds. Fold-level rows remain in runs and CSV.",
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
        "gpu_devices_used": 1 if device.type == "cuda" else 0,
        "gpu_visible_device_count": visible_gpu_count,
        "gpu_device_index_used": gpu_index_used,
        "runtime_summary": runtime_summary,
        "data_leakage_audit": data_leakage_audit,
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
                "fold_index",
                "folds",
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
                "fold_index": record.get("fold_index", 1),
                "folds": record.get("folds", 1),
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
    print(f"Epoch timing and GPU memory CSV: {runtime_summary['epoch_timing_csv']}", flush=True)
    print(f"Timing summary CSV: {runtime_summary['timing_summary_csv']}", flush=True)
    print(
        "Runtime summary: "
        + json.dumps(runtime_summary, indent=2, sort_keys=True),
        flush=True,
    )
    return report
