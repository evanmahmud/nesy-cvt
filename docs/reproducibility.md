# Reproducibility

## Environment

Use Python 3.10–3.12 and a PyTorch wheel that supports the target GPU. For Blackwell RTX 50-series cards, use a current CUDA-enabled PyTorch build; CPU-only and older CUDA wheels will not use this GPU. Install a CUDA-compatible PyTorch/torchvision pair first, then install the package and development tools.

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --upgrade pip
# Select the matching current CUDA wheel from https://pytorch.org/get-started/locally/
.\.venv-win\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
.\.venv-win\Scripts\python.exe -m pip install -e ".[train,dev]"
```

Record Python, PyTorch, torchvision, CUDA runtime, driver, GPU model, OS, and package version with every run.

## Smoke Checks

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"
python -m nesy_cvt train --device cuda --epochs 1 --synthetic-samples 8 --batch-size 4 --output-dir runs/smoke
python -m nesy_cvt eval --device cuda --checkpoint-dir runs/smoke
```

## Controlled Comparison

See [`benchmarking.md`](benchmarking.md). The comparison command writes the fixed split protocol, all run metadata, per-seed results, and aggregate mean/sample standard deviation. The first model/seed run took about 1 hour 49 minutes on the RTX 5070; extrapolating to all 12 runs suggests roughly a day, though time varies by architecture. If interrupted, use the documented recovery launcher: it keeps completed runs and restarts an incomplete seed from epoch 1 because optimizer state is not saved.

## Pilot Record

The initial user-run CIFAR-100 experiments are preserved in [`../results/cifar100_pilot.json`](../results/cifar100_pilot.json). The 50-epoch result is user-reported wall time, one seed, and the test split was viewed after both 5 and 50 epochs. Treat those metrics as exploratory. Future publication results should use the fixed train/validation/test protocol and multiple seeds.
