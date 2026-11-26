from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NesyCVTConfig:
    """Configuration for the experimental NeSy-CVT hybrid image classifier."""

    model_type: str = "nesy-cvt"
    model_size: str = "tiny"
    reasoning_mode: str = "learned_projection"
    image_size: int = 32
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2)
    num_heads: int = 4
    mlp_ratio: float = 4.0
    concept_dim: int = 32
    concept_schema: str = "latent"
    dropout: float = 0.0
    rule_loss_weight: float = 0.05
    concept_loss_weight: float = 0.1
    prototype_loss_weight: float = 0.05
    spectral_loss_weight: float = 0.01
    layer_norm_eps: float = 1e-6
    spectral_mixer_mode: str = "residual_identity"
    use_spectral_mixer: bool = True
    use_transformer: bool = True
    use_concept_auxiliary: bool = True
    classifier_type: str = "cosine"
    taxonomy_rule_weight: float = 0.5
    taxonomy_consistency_weight: float = 0.1

    def __post_init__(self):
        if self.model_size not in {"nano", "tiny", "small", "base", "large", "huge", "custom"}:
            raise ValueError("model_size must be nano, tiny, small, base, large, huge, or custom")
        if self.reasoning_mode not in {"learned_projection", "cifar100_taxonomy"}:
            raise ValueError("reasoning_mode must be learned_projection or cifar100_taxonomy")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not self.depths or any(depth <= 0 for depth in self.depths):
            raise ValueError("depths must contain positive integers")
        if self.concept_dim <= 0:
            raise ValueError("concept_dim must be positive")
        if self.concept_schema not in {"latent", "cifar100_coarse20"}:
            raise ValueError("concept_schema must be latent or cifar100_coarse20")
        if self.concept_schema == "cifar100_coarse20" and self.concept_dim != 20:
            raise ValueError("cifar100_coarse20 requires concept_dim=20")
        if self.reasoning_mode == "cifar100_taxonomy":
            if self.num_classes != 100:
                raise ValueError("cifar100_taxonomy requires num_classes=100")
            if self.concept_schema != "cifar100_coarse20" or self.concept_dim != 20:
                raise ValueError("cifar100_taxonomy requires 20 CIFAR-100 coarse concepts")
            if not self.use_concept_auxiliary:
                raise ValueError("cifar100_taxonomy requires the concept auxiliary head")
        if self.taxonomy_rule_weight < 0 or self.taxonomy_consistency_weight < 0:
            raise ValueError("taxonomy loss weights must be non-negative")
        if self.spectral_mixer_mode not in {"residual_identity", "legacy_double"}:
            raise ValueError("spectral_mixer_mode must be residual_identity or legacy_double")
        if self.classifier_type not in {"cosine", "linear"}:
            raise ValueError("classifier_type must be cosine or linear")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["depths"] = list(self.depths)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NesyCVTConfig:
        normalized = dict(data)
        if "depths" in normalized:
            normalized["depths"] = tuple(normalized["depths"])
        # Pre-0.2 checkpoints used x + IFFT(FFT(x)) at zero spectral gates (2x).
        # Keep loading them with their historical forward behavior.
        normalized.setdefault("spectral_mixer_mode", "legacy_double")
        return cls(**normalized)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> NesyCVTConfig:
        return cls.from_dict(json.loads(value))

    def save_pretrained(self, path: str | Path) -> Path:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        config_path.write_text(self.to_json() + "\n", encoding="utf-8")
        return config_path

    @classmethod
    def from_pretrained(cls, path: str | Path) -> NesyCVTConfig:
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        return cls.from_json(config_path.read_text(encoding="utf-8"))
