from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ModuleNotFoundError(
        "PyTorch is required for nesy_cvt.model. Install with `pip install nesy-cvt[train]` "
        "or install torch from https://pytorch.org/."
    ) from exc

from .config import NesyCVTConfig


@dataclass
class NesyCVTOutput:
    logits: torch.Tensor
    concept_logits: torch.Tensor
    features: torch.Tensor
    loss: torch.Tensor | None = None
    loss_components: dict[str, torch.Tensor] | None = None
    reasoning_trace: dict[str, torch.Tensor] | None = None


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SpectralMixer(nn.Module):
    """Complex FFT mixer with bounded learnable real and imaginary gates."""

    def __init__(self, channels: int, mode: str = "residual_identity"):
        super().__init__()
        self.mode = mode
        self.real_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.imag_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        gate = torch.complex(torch.tanh(self.real_gate), torch.tanh(self.imag_gate))
        mixed = spectrum * (1.0 + gate)
        restored = torch.fft.irfft2(mixed, s=x.shape[-2:], norm="ortho").to(dtype=x.dtype)
        smoothness = (mixed.abs().mean() - spectrum.abs().mean()).abs()
        if self.mode == "legacy_double":
            return x + restored, smoothness
        return restored, smoothness


class TransformerBlock(nn.Module):
    def __init__(self, config: NesyCVTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            config.embed_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class CosinePrototypeClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return self.logit_scale.clamp(1.0, 100.0) * features @ prototypes.t()


class NesyCVTForImageClassification(nn.Module):
    def __init__(self, config: NesyCVTConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.embed_dim),
            nn.GELU(),
            *[
                DepthwiseResidualBlock(config.embed_dim, config.dropout)
                for _ in range(sum(config.depths[:1]))
            ],
        )
        self.spectral_mixer = (
            SpectralMixer(config.embed_dim, mode=config.spectral_mixer_mode)
            if config.use_spectral_mixer
            else None
        )
        self.transformer = nn.ModuleList(
            TransformerBlock(config) for _ in range(sum(config.depths) if config.use_transformer else 0)
        )
        self.feature_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.concept_head = (
            nn.Linear(config.embed_dim, config.concept_dim)
            if config.use_concept_auxiliary
            else None
        )
        self.rule_projection = None
        if config.use_concept_auxiliary and config.reasoning_mode == "learned_projection":
            self.rule_projection = nn.Linear(config.concept_dim, config.num_classes, bias=False)
        if config.reasoning_mode == "cifar100_taxonomy":
            from .taxonomy import CIFAR100_FINE_TO_COARSE

            self.register_buffer(
                "fine_to_coarse",
                torch.tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long),
            )
        self.classifier = (
            CosinePrototypeClassifier(config.embed_dim, config.num_classes)
            if config.classifier_type == "cosine"
            else nn.Linear(config.embed_dim, config.num_classes)
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_repo_id: str | Path,
        *,
        revision: str | None = None,
        token: str | None = None,
        subfolder: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> NesyCVTForImageClassification:
        """Load a saved NeSy-CVT folder or a Hugging Face Hub repository."""
        model_dir = Path(model_path_or_repo_id)
        if not model_dir.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Loading from the Hugging Face Hub requires `pip install nesy-cvt[hub]`."
                ) from exc
            model_dir = Path(
                snapshot_download(
                    repo_id=str(model_path_or_repo_id),
                    revision=revision,
                    token=token,
                    allow_patterns=(
                        ["config.json", "pytorch_model.bin"]
                        if not subfolder
                        else [
                            f"{subfolder.strip('/')}/config.json",
                            f"{subfolder.strip('/')}/pytorch_model.bin",
                        ]
                    ),
                )
            )
        if subfolder:
            model_dir = model_dir / subfolder

        config_path = model_dir / "config.json"
        weights_path = model_dir / "pytorch_model.bin"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Expected config.json and pytorch_model.bin in {model_dir}"
            )

        config = NesyCVTConfig.from_pretrained(model_dir)
        model = cls(config)
        state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        concept_targets: torch.Tensor | None = None,
    ) -> NesyCVTOutput:
        x = self.stem(pixel_values)
        if self.spectral_mixer is not None:
            x, spectral_smoothness = self.spectral_mixer(x)
        else:
            spectral_smoothness = x.new_zeros(())
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.transformer:
            tokens = block(tokens)
        features = self.feature_norm(tokens.mean(dim=1))
        concept_logits = (
            self.concept_head(features)
            if self.concept_head is not None
            else features.new_zeros((features.shape[0], self.config.concept_dim))
        )
        neural_logits = self.classifier(features)
        reasoning_trace = None
        if self.config.reasoning_mode == "cifar100_taxonomy":
            concept_probabilities = torch.softmax(concept_logits, dim=-1)
            concept_log_probabilities = torch.log_softmax(concept_logits, dim=-1)
            rule_log_scores = concept_log_probabilities[:, self.fine_to_coarse]
            logits = neural_logits + self.config.taxonomy_rule_weight * rule_log_scores
            reasoning_trace = {
                "neural_logits": neural_logits,
                "concept_probabilities": concept_probabilities,
                "rule_log_scores": rule_log_scores,
                "fine_to_coarse": self.fine_to_coarse,
            }
        else:
            logits = neural_logits

        loss = None
        components: dict[str, torch.Tensor] = {}
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            concept_loss = None
            if self.config.reasoning_mode == "cifar100_taxonomy":
                neural_class_probabilities = torch.softmax(neural_logits, dim=-1)
                coarse_from_classes = torch.zeros_like(concept_probabilities).scatter_add(
                    1,
                    self.fine_to_coarse.unsqueeze(0).expand(labels.shape[0], -1),
                    neural_class_probabilities,
                )
                rule_loss = F.mse_loss(concept_probabilities, coarse_from_classes)
                coarse_targets = (
                    concept_targets.argmax(dim=-1).to(dtype=torch.long)
                    if concept_targets is not None
                    else self.fine_to_coarse[labels]
                )
                concept_loss = F.cross_entropy(concept_logits, coarse_targets)
            elif self.rule_projection is not None:
                rule_logits = self.rule_projection(torch.sigmoid(concept_logits))
                rule_loss = F.cross_entropy(rule_logits, labels)
            else:
                rule_loss = cls_loss.new_zeros(())
            if isinstance(self.classifier, CosinePrototypeClassifier):
                prototype_loss = 1.0 - F.cosine_similarity(
                    F.normalize(features, dim=-1),
                    F.normalize(self.classifier.prototypes[labels], dim=-1),
                    dim=-1,
                ).mean()
            else:
                prototype_loss = cls_loss.new_zeros(())
            components = {
                "classification": cls_loss,
                "rule": rule_loss,
                "prototype": prototype_loss,
                "spectral": spectral_smoothness,
            }
            loss = cls_loss + self.config.spectral_loss_weight * spectral_smoothness
            if self.config.reasoning_mode == "cifar100_taxonomy":
                components["rule"] = rule_loss
                components["concept"] = concept_loss
                loss = loss + self.config.taxonomy_consistency_weight * rule_loss
            elif self.rule_projection is not None:
                loss = loss + self.config.rule_loss_weight * rule_loss
            if isinstance(self.classifier, CosinePrototypeClassifier):
                loss = loss + self.config.prototype_loss_weight * prototype_loss
            if self.config.reasoning_mode == "cifar100_taxonomy":
                loss = loss + self.config.concept_loss_weight * concept_loss
            elif concept_targets is not None and self.concept_head is not None:
                if self.config.concept_schema == "cifar100_coarse20":
                    concept_loss = F.cross_entropy(
                        concept_logits,
                        concept_targets.argmax(dim=-1).to(dtype=torch.long),
                    )
                else:
                    concept_loss = F.binary_cross_entropy_with_logits(
                        concept_logits,
                        concept_targets.to(dtype=concept_logits.dtype),
                    )
                components["concept"] = concept_loss
                loss = loss + self.config.concept_loss_weight * concept_loss

        return NesyCVTOutput(
            logits=logits,
            concept_logits=concept_logits,
            features=features,
            loss=loss,
            loss_components=components,
            reasoning_trace=reasoning_trace,
        )
