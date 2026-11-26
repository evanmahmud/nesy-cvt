"""NeSy-CVT public API."""

from .config import NesyCVTConfig

__all__ = [
    "NesyCVTConfig",
    "NesyCVTForImageClassification",
    "NesyCVTOutput",
    "create_model",
    "load_pretrained",
    "nesy_cvt_base",
    "nesy_cvt_huge",
    "nesy_cvt_large",
    "nesy_cvt_nano",
    "nesy_cvt_small",
    "nesy_cvt_tiny",
]


def __getattr__(name):
    if name in {"NesyCVTForImageClassification", "NesyCVTOutput"}:
        from .model import NesyCVTForImageClassification, NesyCVTOutput

        values = {
            "NesyCVTForImageClassification": NesyCVTForImageClassification,
            "NesyCVTOutput": NesyCVTOutput,
        }
        return values[name]
    if name in {
        "create_model",
        "load_pretrained",
        "nesy_cvt_base",
        "nesy_cvt_huge",
        "nesy_cvt_large",
        "nesy_cvt_nano",
        "nesy_cvt_small",
        "nesy_cvt_tiny",
    }:
        from . import model_factory

        return getattr(model_factory, name)
    raise AttributeError(f"module 'nesy_cvt' has no attribute {name!r}")
