from __future__ import annotations

import json
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import save_metrics_json
from .config import NesyCVTConfig
from .runtime import resolve_device


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "synthetic"
    data_dir: str = "data"
    download: bool = False
    output_dir: str = "runs/nesy-cvt-tiny"
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    seed: int = 42
    synthetic_samples: int = 256
    device: str = "auto"


def _build_dataloader(config: TrainConfig, model_config: NesyCVTConfig):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if config.dataset == "synthetic":
        pixels = torch.randn(
            config.synthetic_samples,
            model_config.in_channels,
            model_config.image_size,
            model_config.image_size,
        )
        labels = torch.randint(0, model_config.num_classes, (config.synthetic_samples,))
        generator = torch.Generator().manual_seed(config.seed)
        return DataLoader(
            TensorDataset(pixels, labels),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )

    if config.dataset == "cifar100":
        try:
            from torchvision import datasets, transforms
        except ModuleNotFoundError as exc:
            raise RuntimeError("CIFAR-100 training requires torchvision.") from exc
        transform = transforms.Compose(
            [
                transforms.Resize((model_config.image_size, model_config.image_size)),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.CIFAR100(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=transform,
        )
        generator = torch.Generator().manual_seed(config.seed)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            generator=generator,
        )

    raise ValueError("dataset must be either 'synthetic' or 'cifar100'")


def train(config: TrainConfig, model_config: NesyCVTConfig | None = None) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("Training requires PyTorch. Install the training extras first.") from exc

    from .model import NesyCVTForImageClassification

    random.seed(config.seed)
    try:
        import numpy as np

        np.random.seed(config.seed)
    except ModuleNotFoundError:
        pass
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = resolve_device(torch, config.device)
    device_label = str(device)
    if device.type == "cuda":
        device_label = f"{device} ({torch.cuda.get_device_name(device)})"
    print(f"Using device: {device_label}", flush=True)
    model_config = model_config or NesyCVTConfig()
    model = NesyCVTForImageClassification(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader = _build_dataloader(config, model_config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    first_loss: float | None = None
    last_loss = 0.0
    epoch_history: list[dict[str, float | int]] = []
    train_start = time.perf_counter()
    for epoch in range(config.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_steps = 0
        samples_seen = 0
        for batch_pixels, batch_labels in loader:
            batch_pixels = batch_pixels.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_pixels, labels=batch_labels)
            assert output.loss is not None
            output.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = float(output.loss.detach().cpu())
            batch_samples = batch_labels.shape[0]
            epoch_loss += batch_loss * batch_samples
            epoch_steps += 1
            samples_seen += batch_samples

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        avg_epoch_loss = epoch_loss / max(samples_seen, 1)
        last_loss = avg_epoch_loss
        if first_loss is None:
            first_loss = avg_epoch_loss
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_result = {
            "epoch": epoch + 1,
            "avg_loss": avg_epoch_loss,
            "last_batch_loss": batch_loss if epoch_steps else 0.0,
            "seconds": epoch_seconds,
            "images_per_second": samples_seen / max(epoch_seconds, 1e-9),
        }
        epoch_history.append(epoch_result)
        print(
            f"epoch {epoch + 1}/{config.epochs} "
            f"avg_loss={avg_epoch_loss:.4f} last_loss={last_loss:.4f}",
            flush=True,
        )
        model_config.save_pretrained(output_dir)
        checkpoint_tmp = output_dir / "pytorch_model.bin.tmp"
        torch.save(model.state_dict(), checkpoint_tmp)
        checkpoint_tmp.replace(output_dir / "pytorch_model.bin")
        (output_dir / "progress.json").write_text(
            json.dumps(
                {"completed_epochs": epoch + 1, "epochs": epoch_history},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - train_start
    train_samples_per_epoch = (
        config.synthetic_samples if config.dataset == "synthetic" else len(loader.dataset)
    )
    model_config.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    metrics = {
        "dataset": config.dataset,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_decreased": last_loss <= first_loss if first_loss is not None else False,
        "epochs": epoch_history,
        "training_seconds": training_seconds,
        "mean_epoch_seconds": statistics.mean(row["seconds"] for row in epoch_history)
        if epoch_history
        else 0.0,
        "mean_train_images_per_second": train_samples_per_epoch
        * config.epochs
        / max(training_seconds, 1e-9),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
    }
    training_config = {
        "dataset": config.dataset,
        "data_dir": config.data_dir,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": device_label,
        "input_size": f"RGB {model_config.image_size}x{model_config.image_size}",
        "train_transform": (
            "Resize(32), AutoAugment(CIFAR10 policy), ToTensor"
            if config.dataset == "cifar100"
            else "synthetic random normal tensors"
        ),
        "model_config": model_config.to_dict(),
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_runtime": torch.version.cuda,
        "torchvision": None,
        "training_seconds": training_seconds,
    }
    try:
        import torchvision

        training_config["torchvision"] = torchvision.__version__
    except ModuleNotFoundError:
        pass
    (output_dir / "run_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_metrics_json(
        output_dir / "metrics.json",
        metrics,
        seed=config.seed,
        hardware=device_label,
        extra={"training_config": training_config},
    )
    return metrics
