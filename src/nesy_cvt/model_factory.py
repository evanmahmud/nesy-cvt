from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .config import NesyCVTConfig

if TYPE_CHECKING:
    from .model import NesyCVTForImageClassification

MODEL_SIZE_PRESETS = {
    "nano": {"embed_dim": 64, "depths": (1, 1), "num_heads": 4},
    "tiny": {"embed_dim": 96, "depths": (2, 2), "num_heads": 4},
    "small": {"embed_dim": 128, "depths": (3, 3), "num_heads": 4},
    "base": {"embed_dim": 192, "depths": (4, 4), "num_heads": 6},
    "large": {"embed_dim": 256, "depths": (6, 6), "num_heads": 8},
    "huge": {"embed_dim": 320, "depths": (8, 8), "num_heads": 10},
}

MODEL_NAMES = (
    "nesy_cvt",
    "nesy_cvt_tiny",
    "nesy_cvt_small",
    "nesy_cvt_base",
    "nesy_cvt_large",
    "nesy_cvt_nano",
    "nesy_cvt_huge",
    "nesy_cvt_coarse_concepts",
    "nesy_cvt_no_fft",
    "nesy_cvt_no_transformer",
    "nesy_cvt_no_concept_auxiliary",
    "nesy_cvt_linear_classifier",
    "nesy_cvt_no_rule_fusion",
    "nesy_cvt_no_hierarchy_consistency",
    "nesy_cvt_no_coarse_supervision",
    "resnet18_cifar",
    "mobilenet_v3_small",
    "densenet121_cifar",
    "convnext_tiny_cifar",
    "swin_t_cifar",
    "vit_tiny_patch4",
    "wide_resnet28x10_cifar",
)

TAXONOMY_ABLATIONS = {
    "nesy_cvt_no_fft",
    "nesy_cvt_no_transformer",
    "nesy_cvt_no_concept_auxiliary",
    "nesy_cvt_linear_classifier",
    "nesy_cvt_no_rule_fusion",
    "nesy_cvt_no_hierarchy_consistency",
    "nesy_cvt_no_coarse_supervision",
}


def config_for_model(model_name: str, config: NesyCVTConfig) -> NesyCVTConfig:
    """Return the architecture config for a named benchmark variant."""
    size = model_name.removeprefix("nesy_cvt_")
    if size in MODEL_SIZE_PRESETS:
        return replace(
            config,
            model_size=size,
            concept_dim=20,
            concept_schema="cifar100_coarse20",
            reasoning_mode="cifar100_taxonomy",
            **MODEL_SIZE_PRESETS[size],
        )
    if model_name in TAXONOMY_ABLATIONS:
        if model_name == "nesy_cvt_no_concept_auxiliary":
            return replace(
                config,
                model_size="tiny",
                concept_dim=32,
                concept_schema="latent",
                reasoning_mode="learned_projection",
                use_concept_auxiliary=False,
                **MODEL_SIZE_PRESETS["tiny"],
            )
        return replace(
            config,
            model_size="tiny",
            concept_dim=20,
            concept_schema="cifar100_coarse20",
            reasoning_mode="cifar100_taxonomy",
            **MODEL_SIZE_PRESETS["tiny"],
        )
    if model_name == "nesy_cvt_coarse_concepts":
        return replace(config, concept_dim=20, concept_schema="cifar100_coarse20")
    return config


def create_model(size: str = "tiny", *, num_classes: int = 100, **config_overrides):
    """Create a randomly initialized NeSy-CVT CIFAR-100 model using a size preset."""
    if size not in MODEL_SIZE_PRESETS:
        raise ValueError(
            f"Unknown model size {size!r}; choose nano, tiny, small, base, large, or huge"
        )

    from .model import NesyCVTForImageClassification

    reasoning_mode = config_overrides.pop("reasoning_mode", "cifar100_taxonomy")
    values = {
        "num_classes": num_classes,
        "model_size": size,
        "reasoning_mode": reasoning_mode,
        **MODEL_SIZE_PRESETS[size],
    }
    if reasoning_mode == "cifar100_taxonomy":
        values.update(concept_dim=20, concept_schema="cifar100_coarse20")
    values.update(config_overrides)
    # Custom dimensions should not continue to claim one of the stock presets.
    if any(key in config_overrides for key in ("embed_dim", "depths", "num_heads")):
        values["model_size"] = "custom"
    return NesyCVTForImageClassification(NesyCVTConfig(**values))


def nesy_cvt_tiny(*, num_classes: int = 100, **config_overrides):
    """Create a randomly initialized NeSy-CVT Tiny CIFAR-100 classifier."""
    return create_model("tiny", num_classes=num_classes, **config_overrides)


def nesy_cvt_nano(*, num_classes: int = 100, **config_overrides):
    """Create a randomly initialized NeSy-CVT Nano CIFAR-100 classifier."""
    return create_model("nano", num_classes=num_classes, **config_overrides)


def nesy_cvt_small(*, num_classes: int = 100, **config_overrides):
    """Create a randomly initialized NeSy-CVT Small CIFAR-100 classifier."""
    return create_model("small", num_classes=num_classes, **config_overrides)


def nesy_cvt_base(*, num_classes: int = 100, **config_overrides):
    """Create a randomly initialized NeSy-CVT Base CIFAR-100 classifier."""
    return create_model("base", num_classes=num_classes, **config_overrides)


def nesy_cvt_large(*, num_classes: int = 100, **config_overrides):
    """Create a randomly initialized NeSy-CVT Large CIFAR-100 classifier."""
    return create_model("large", num_classes=num_classes, **config_overrides)


def nesy_cvt_huge(*, num_classes: int = 100, **config_overrides):
    """Create a randomly initialized NeSy-CVT Huge CIFAR-100 classifier."""
    return create_model("huge", num_classes=num_classes, **config_overrides)


def load_pretrained(
    size: str = "tiny",
    *,
    repo_id: str = "evanmahmud/nesy-cvt",
    device: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> NesyCVTForImageClassification:
    """Load a trained Hub variant and return it in evaluation mode.

    Use ``nesy_cvt_tiny()`` (or another size constructor) for random initialization.
    """
    if size not in MODEL_SIZE_PRESETS:
        choices = ", ".join(MODEL_SIZE_PRESETS)
        raise ValueError(f"Unknown pretrained size {size!r}; choose one of: {choices}")

    from .model import NesyCVTForImageClassification

    model = NesyCVTForImageClassification.from_pretrained(
        repo_id,
        revision=revision,
        token=token,
        subfolder=size,
    )
    if device is not None:
        model = model.to(device)
    return model.eval()


def build_classifier(model_name: str, config: NesyCVTConfig):
    """Build NeSy-CVT or a torchvision reference model for CIFAR-sized inputs."""
    if model_name.startswith("nesy_cvt"):
        from .model import NesyCVTForImageClassification

        model_config = config_for_model(model_name, config)
        ablations = {
            "nesy_cvt_no_fft": {"use_spectral_mixer": False},
            "nesy_cvt_no_transformer": {"use_transformer": False},
            "nesy_cvt_linear_classifier": {"classifier_type": "linear"},
        }
        if model_name in ablations:
            from dataclasses import replace

            model_config = replace(model_config, **ablations[model_name])
        return NesyCVTForImageClassification(model_config)

    try:
        from torch import nn
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reference models require torchvision.") from exc

    if model_name == "resnet18_cifar":
        model = models.resnet18(weights=None, num_classes=config.num_classes)
        model.conv1 = nn.Conv2d(
            config.in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        return model

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None, num_classes=config.num_classes)
        model.features[0][0].stride = (1, 1)
        return model

    if model_name == "densenet121_cifar":
        model = models.densenet121(weights=None, num_classes=config.num_classes)
        model.features.conv0 = nn.Conv2d(
            config.in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.features.pool0 = nn.Identity()
        return model

    if model_name == "convnext_tiny_cifar":
        return models.convnext_tiny(weights=None, num_classes=config.num_classes)

    if model_name == "swin_t_cifar":
        return models.swin_t(weights=None, num_classes=config.num_classes)

    if model_name == "vit_tiny_patch4":
        return models.VisionTransformer(
            image_size=config.image_size,
            patch_size=4,
            num_layers=6,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=384,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=config.num_classes,
        )

    if model_name == "wide_resnet28x10_cifar":
        from .cifar_models import wide_resnet28x10_cifar

        return wide_resnet28x10_cifar(num_classes=config.num_classes)

    raise ValueError(f"Unknown model {model_name!r}; choose from {', '.join(MODEL_NAMES)}")
