from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import save_metrics_json
from .config import NesyCVTConfig
from .runtime import resolve_device


@dataclass(frozen=True)
class EvalConfig:
    checkpoint_dir: str
    output_path: str = "runs/eval/metrics.json"
    seed: int = 42
    samples: int = 128
    device: str = "auto"
    dataset: str = "synthetic"
    data_dir: str = "data"
    batch_size: int = 64
    download: bool = False


def evaluate(config: EvalConfig) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Evaluation requires PyTorch. Install the training extras first.") from exc

    from .metrics import accuracy_at_k, macro_f1
    from .model import NesyCVTForImageClassification

    torch.manual_seed(config.seed)
    device = resolve_device(torch, config.device)
    device_label = str(device)
    if device.type == "cuda":
        device_label = f"{device} ({torch.cuda.get_device_name(device)})"
    model_config = NesyCVTConfig.from_pretrained(config.checkpoint_dir)
    model = NesyCVTForImageClassification(model_config).to(device)
    checkpoint = Path(config.checkpoint_dir) / "pytorch_model.bin"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    if config.dataset == "synthetic":
        pixels = torch.randn(
            config.samples,
            model_config.in_channels,
            model_config.image_size,
            model_config.image_size,
        )
        labels = torch.randint(0, model_config.num_classes, (config.samples,))
        batches = [(pixels, labels)]
    elif config.dataset == "cifar100":
        try:
            from torchvision import datasets, transforms
        except ModuleNotFoundError as exc:
            raise RuntimeError("CIFAR-100 evaluation requires torchvision.") from exc
        transform = transforms.Compose(
            [
                transforms.Resize((model_config.image_size, model_config.image_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.CIFAR100(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=transform,
        )
        batches = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        raise ValueError("dataset must be either 'synthetic' or 'cifar100'")

    all_logits: list[list[float]] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for pixels, labels in batches:
            output = model(pixels.to(device))
            all_logits.extend(output.logits.detach().cpu().tolist())
            all_labels.extend(labels.tolist())
    predictions = [max(range(len(row)), key=row.__getitem__) for row in all_logits]
    metrics = accuracy_at_k(
        all_logits,
        all_labels,
        topk=(1, min(5, model_config.num_classes)),
    )
    metrics["macro_f1"] = macro_f1(predictions, all_labels, model_config.num_classes)
    save_metrics_json(config.output_path, metrics, seed=config.seed, hardware=device_label)
    return metrics
