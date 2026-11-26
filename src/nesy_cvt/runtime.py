from __future__ import annotations


def resolve_device(torch, requested: str):
    """Choose CPU/CUDA, with an actionable error when CUDA was explicitly requested."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU. "
                "Install a CUDA-enabled PyTorch build and check `nvidia-smi`."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, cpu")
