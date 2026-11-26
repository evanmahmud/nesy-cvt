"""CUDA-first CIFAR-100 training and evaluation commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import NesyCVTConfig


def _torch_and_model():
    try:
        import torch

        from .model import NesyCVTForImageClassification
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install a CUDA-enabled PyTorch build and the project extras before using this command."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this Python environment. Check `nvidia-smi` and install "
            "a CUDA-enabled PyTorch wheel."
        )
    return torch, NesyCVTForImageClassification


def _prepare_cuda(torch, model, *, downsample_attention=True):
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # CIFAR-100 images become 16x16 tokens instead of 32x32, reducing
    # quadratic attention work by 16x without changing checkpoint tensor shapes.
    if downsample_attention:
        first_conv = model.stem[0]
        if not isinstance(first_conv, torch.nn.Conv2d):
            raise RuntimeError("Unexpected model stem; cannot enable the fast 16x16 token path.")
        first_conv.stride = (2, 2)
    model = model.to(device=device, memory_format=torch.channels_last)
    return model, device


def _amp_settings(torch):
    use_bfloat16 = bool(torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bfloat16)
    return amp_dtype, scaler


def _data_loaders(args, config):
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError("CIFAR-100 requires torchvision; install the project training extras.") from exc

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ]
    )
    train_set = datasets.CIFAR100(
        root=args.data_dir,
        train=True,
        download=args.download,
        transform=train_transform,
    )
    eval_set = datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        download=args.download,
        transform=eval_transform,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **common)
    eval_loader = DataLoader(eval_set, shuffle=False, **common)
    return train_loader, eval_loader


def _save_metrics(path: Path, metrics: dict, *, seed: int, device_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "reproducibility": {
                    "seed": seed,
                    "device": device_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def train(args) -> dict:
    torch, Model = _torch_and_model()
    from torch import nn

    from .benchmark import save_metrics_json

    torch.manual_seed(args.seed)
    config = NesyCVTConfig()
    model, device = _prepare_cuda(torch, Model(config))
    train_loader, _ = _data_loaders(args, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    amp_dtype, scaler = _amp_settings(torch)
    device_name = torch.cuda.get_device_name(device)
    print(f"device={device} gpu={device_name} amp={amp_dtype}", flush=True)
    epoch_history = []
    model.train()

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        samples_seen = 0
        for pixels, labels in train_loader:
            pixels = pixels.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = model(pixels, labels=labels)
            if output.loss is None:
                raise RuntimeError("The model did not return a training loss.")
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_count = labels.shape[0]
            loss_sum += output.loss.detach().float() * batch_count
            samples_seen += batch_count

        avg_loss = float((loss_sum / max(samples_seen, 1)).item())
        seconds = time.perf_counter() - start
        epoch_result = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "seconds": seconds,
            "images_per_second": samples_seen / max(seconds, 1e-9),
        }
        epoch_history.append(epoch_result)
        print(
            f"epoch {epoch}/{args.epochs} train_loss={avg_loss:.4f} "
            f"time={seconds:.1f}s images_per_second={epoch_result['images_per_second']:.1f}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    metrics = {
        "dataset": "cifar100",
        "epochs": epoch_history,
        "attention_tokens_per_image": 16 * 16,
    }
    save_metrics_json(
        output_dir / "metrics.json",
        metrics,
        seed=args.seed,
        hardware=device_name,
    )
    return metrics


def evaluate(args) -> dict:
    torch, Model = _torch_and_model()
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from .metrics import accuracy_at_k, expected_calibration_error, macro_f1

    checkpoint_dir = Path(args.checkpoint_dir)
    config = NesyCVTConfig.from_pretrained(checkpoint_dir)
    run_metrics_path = checkpoint_dir / "metrics.json"
    downsample_attention = False
    if run_metrics_path.exists():
        run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
        downsample_attention = (
            run_metrics.get("metrics", {}).get("attention_tokens_per_image") == 256
        )
    model, device = _prepare_cuda(
        torch,
        Model(config),
        downsample_attention=downsample_attention,
    )
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        download=args.download,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    amp_dtype, _ = _amp_settings(torch)
    all_logits = []
    all_labels = []
    all_predictions = []
    all_confidences = []
    all_correct = []
    with torch.inference_mode():
        for pixels, labels in loader:
            pixels = pixels.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pixels).logits
            probabilities = logits.float().softmax(dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            label_values = labels.cpu().tolist()
            prediction_values = prediction.cpu().tolist()
            all_logits.extend(logits.float().cpu().tolist())
            all_labels.extend(label_values)
            all_predictions.extend(prediction_values)
            all_confidences.extend(confidence.cpu().tolist())
            all_correct.extend(p == y for p, y in zip(prediction_values, label_values))

    metrics = accuracy_at_k(all_logits, all_labels, topk=(1, 5))
    metrics["macro_f1"] = macro_f1(all_predictions, all_labels, config.num_classes)
    metrics["ece"] = expected_calibration_error(all_confidences, all_correct)
    metrics["samples"] = len(all_labels)
    metrics["device"] = torch.cuda.get_device_name(device)
    _save_metrics(Path(args.output), metrics, seed=args.seed, device_name=metrics["device"])
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m nesy_cvt.fast")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train CIFAR-100 with CUDA")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--download", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--output-dir", default="runs/cifar100-rtx5070")

    eval_parser = commands.add_parser("eval", help="Score a checkpoint on CIFAR-100 test split")
    eval_parser.add_argument("--checkpoint-dir", required=True)
    eval_parser.add_argument("--data-dir", default="data")
    eval_parser.add_argument("--download", action="store_true")
    eval_parser.add_argument("--batch-size", type=int, default=256)
    eval_parser.add_argument("--num-workers", type=int, default=4)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--output", default="runs/cifar100-test-metrics.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train(args)
    else:
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
