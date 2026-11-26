---
library_name: pytorch
pipeline_tag: image-classification
license: apache-2.0
datasets:
  - cifar100
metrics:
  - accuracy
  - f1
tags:
  - computer-vision
  - image-classification
  - neuro-symbolic-ai
  - hierarchical-classification
  - cifar100
  - convolutional-neural-network
  - vision-transformer
  - spectral-mixing
---

# NeSy-CVT

NeSy-CVT (Neuro-Symbolic Convolutional Vision Transformer) is an experimental model family for CIFAR-100 classification. Its taxonomy variants combine a convolutional feature stem, a gated Fourier-domain mixer, Transformer blocks, a cosine-prototype fine-classifier, a supervised 20-way superclass head, and a differentiable rule layer that uses CIFAR-100's fixed fine-to-coarse taxonomy.

> **Research status.** This release does not claim state-of-the-art performance. The CIFAR-100 test set was inspected in earlier exploratory project runs, so the metrics below are exploratory matched comparisons, not results from an untouched confirmatory holdout. The fixed rules encode one dataset hierarchy and do not establish general-purpose reasoning, human interpretability, robustness, or generalization to other datasets.

## Architecture

![NeSy-CVT taxonomy-mode architecture](architecture_overview.svg)

### Forward path

1. A 3×3 convolution, batch normalization, GELU, and depthwise residual blocks form local visual features.
2. A 2-D real FFT applies bounded real and imaginary channel gates; an inverse FFT returns the spatial feature map. New taxonomy checkpoints use `residual_identity`, where zero-initialized gates yield an identity transform at initialization.
3. Flattened spatial positions become tokens for pre-normalized multi-head self-attention and MLP blocks.
4. Mean token pooling and LayerNorm produce the image embedding.
5. A cosine-prototype head produces 100 fine-class logits. A supervised linear concept head predicts 20 CIFAR-100 superclass probabilities.
6. A fixed map connects each fine class to its known superclass. For fine class `c`, the fused score is `z[c] = z_neural[c] + 0.5 × log P(superclass(c) | image)`.

In taxonomy mode, the optional `reasoning_trace` contains neural logits, concept probabilities, rule scores, and the fine-to-coarse map. These are inspectable model intermediates; they are not by themselves a human-validated explanation.

The taxonomy training objective combines fine-class cross-entropy, coarse-superclass cross-entropy, hierarchy-consistency mean squared error, prototype regularization, and a spectral-magnitude regularizer. Historical learned-projection checkpoints use a different configuration; inspect `config.json` for `reasoning_mode` and `spectral_mixer_mode`.

## Available checkpoints

The root checkpoint is NeSy-CVT Huge. The model repository also contains all six trained size variants in explicit subfolders:

| Variant | Width | Transformer blocks | Heads | Parameters (M) | Hub folder |
|---|---:|---:|---:|---:|---|
| Nano | 64 | 2 | 4 | 0.115 | `nano` |
| Tiny | 96 | 4 | 4 | 0.483 | `tiny` |
| Small | 128 | 6 | 4 | 1.264 | `small` |
| Base | 192 | 8 | 6 | 3.746 | `base` |
| Large | 256 | 12 | 8 | 9.930 | `large` |
| Huge | 320 | 16 | 10 | 20.629 | `huge` (or repository root) |

The checkpoint in each variant folder is seed 42, selected at that seed's best validation top-1 epoch. The multi-seed result table aggregates test metrics across seeds 42, 43, and 44. The Python constructors such as `nesy_cvt_tiny()` return random initialization; use `from_pretrained` to load trained weights.

## Load and run inference

Install a compatible PyTorch build and `huggingface_hub`, then install this project's package from [GitHub](https://github.com/evanmahmud/nesy-cvt):

```bash
pip install "git+https://github.com/evanmahmud/nesy-cvt.git" "huggingface_hub>=0.30"
```

Example with the Tiny checkpoint:

```python
import torch
from nesy_cvt import NesyCVTForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NesyCVTForImageClassification.from_pretrained(
    "evanmahmud/nesy-cvt", subfolder="tiny"
).to(device).eval()

# Replace random input with RGB images resized to 32 x 32 and converted to [0, 1].
pixel_values = torch.rand(2, 3, 32, 32, device=device)
with torch.inference_mode():
    output = model(pixel_values)

predicted_class_ids = output.logits.argmax(dim=-1)
predicted_superclass_ids = output.reasoning_trace["concept_probabilities"].argmax(dim=-1)
print("class IDs:", predicted_class_ids.tolist())
print("superclass IDs:", predicted_superclass_ids.tolist())
```

Set `subfolder` to `nano`, `tiny`, `small`, `base`, `large`, or `huge`. Omitting it loads the Huge checkpoint at the repository root. Inputs must be float tensors of shape `[N, 3, 32, 32]` in `[0, 1]`, matching `ToTensor` preprocessing. Do not apply ImageNet mean/std normalization. These checkpoints were trained for CIFAR-100; they are not validated for other datasets, resolutions, or image domains.

The model returns 100 fine-class logits in `output.logits`, 20 raw superclass scores in `output.concept_logits`, and, for taxonomy variants, a dictionary of neural and rule intermediates in `output.reasoning_trace`. Map predicted IDs to human-readable names using CIFAR-100's official fine and coarse label lists.

## Evaluation

The comparison includes 13 scratch-trained configurations: six NeSy-CVT sizes, a latent-concept NeSy-CVT baseline, a supervised-concept baseline with a learned projection, ResNet-18, MobileNetV3-Small, DenseNet-121, ConvNeXt-Tiny, Swin-T, and ViT-Tiny/4. All models used the same stratified 45,000/5,000 train/validation split, 32×32 inputs, three seeds (42, 43, 44), 50 epochs, batch size 16, and official 10,000-example test set. The checkpoints were selected using validation top-1. All baseline models were trained from scratch under the same fixed recipe; it was not separately optimized for every architecture.

| Model | Top-1 (%) | Macro F1 (%) | Parameters (M) | Est. FLOPs (G/image) | Checkpoint (MB) | Batch-1 p50 (ms) | Batch-16 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeSy-CVT Nano | 51.31 ± 0.89 | 50.79 ± 1.10 | 0.11 | 0.755 | 0.48 | 1.57 | 3,362 |
| NeSy-CVT Tiny | 59.12 ± 0.39 | 58.82 ± 0.53 | 0.48 | 2.568 | 1.96 | 2.17 | 1,569 |
| NeSy-CVT Small | 62.17 ± 0.65 | 62.02 ± 0.77 | 1.26 | 5.759 | 5.09 | 3.01 | 970 |
| NeSy-CVT Base | 65.95 ± 0.32 | 65.93 ± 0.44 | 3.75 | 14.027 | 15.04 | 3.92 | 453 |
| NeSy-CVT Large | 68.75 ± 0.33 | 68.66 ± 0.36 | 9.93 | 33.074 | 39.80 | 6.61 | 218 |
| NeSy-CVT Huge | 69.58 ± 0.44 | 69.45 ± 0.40 | 20.63 | 63.500 | 82.63 | 12.52 | 121 |
| ResNet-18 | 61.25 ± 0.46 | 61.06 ± 0.45 | 11.22 | 1.111 | 44.96 | 1.74 | 9,030 |
| DenseNet-121 | 68.13 ± 0.43 | 68.00 ± 0.37 | 7.05 | 1.777 | 28.77 | 10.04 | 1,597 |
| MobileNetV3-Small | 47.47 ± 0.46 | 46.94 ± 0.48 | 1.62 | 0.011 | 6.60 | 3.36 | 4,561 |

Values are mean ± sample standard deviation across three seeds. The full 13-model table also includes top-5 accuracy, macro precision, macro recall, macro one-vs-rest AUC, peak allocated GPU memory, and training time. It is available as [Markdown](https://github.com/evanmahmud/nesy-cvt/blob/main/docs/benchmark/cifar100-taxonomy-family/comparison_table.md), [CSV](https://github.com/evanmahmud/nesy-cvt/blob/main/docs/benchmark/cifar100-taxonomy-family/comparison_table.csv), and [seed-level JSON with test-set fingerprint](https://github.com/evanmahmud/nesy-cvt/blob/main/docs/benchmark/cifar100-taxonomy-family/cifar100-taxonomy-detailed.json).

These values show resource trade-offs under a fixed recipe. NeSy-CVT Huge has the highest observed mean top-1 in this roster, while using more parameters, far more estimated FLOPs, a longer training time, and lower batched throughput than several baselines. They do not establish general or state-of-the-art superiority.

### Figures

![CIFAR-100 quality metrics](benchmark/figures/quality_metrics.png)

![Top-1 accuracy versus estimated FLOPs](benchmark/figures/flops.png)

![Top-1 accuracy versus batch-1 latency](benchmark/figures/latency.png)

![Top-1 accuracy versus batched FPS](benchmark/figures/fps.png)

The full interactive Plotly dashboard and individual figures are hosted in the [GitHub benchmark artifacts](https://github.com/evanmahmud/nesy-cvt/tree/main/docs/benchmark/cifar100-taxonomy-family). FLOPs are analytical estimates and omit several operations. Latency is batch-1 GPU model execution excluding transfer; FPS is batched model-only throughput at batch size 16. Hardware measurements were collected on one NVIDIA GeForce RTX 5070 system and should not be generalized to other setups.

### Test-set exposure

Prior exploratory project runs evaluated the official CIFAR-100 test set before this controlled comparison. Although the compared models share the same held-out test examples and those examples were not used for training or checkpoint selection in these runs, prior project-wide inspection means this is not an untouched, blinded test set. Treat all test metrics as exploratory. Independent data and a preregistered evaluation are needed for confirmatory generalization claims.

## Training details

- Dataset: CIFAR-100 (50,000 training images; official 10,000-image test set).
- Split for the comparison: stratified 45,000 training / 5,000 validation, with 50 validation images per class; split seed 1729.
- Input and preprocessing: RGB 32×32; `Resize(32)`, CIFAR-10 AutoAugment policy for training, and `ToTensor`; no channel normalization.
- Recipe: AdamW, learning rate `3e-4`, batch size 16, 50 epochs, gradient clipping at 1.0, float32, no learning-rate scheduler.
- Seeds: 42, 43, and 44.
- Checkpoint selection: highest validation top-1 for each model/seed.
- Hardware: NVIDIA GeForce RTX 5070, CUDA-enabled PyTorch. The complete software and per-run records are in the benchmark JSON and `run_config.json` files.

## Limitations and intended use

This release is intended for research and education on small-image classification. The taxonomy is dataset-specific and performance on CIFAR-100 does not establish performance elsewhere. The published comparison is a fixed-recipe study, has prior test-set exposure, and lacks component ablations needed to establish which model components cause the observed results. FLOPs are estimates. Inference and training measurements depend on the reported hardware/software setup. Do not use these results as evidence for safety-critical or real-world deployment.

## Files

- `config.json`: model architecture and reasoning configuration.
- `pytorch_model.bin`: PyTorch state dictionary for the Huge root checkpoint.
- `nano/`, `tiny/`, `small/`, `base/`, `large/`, `huge/`: variant configs, checkpoints, and run records.
- `nesy_cvt/`: Python model implementation for direct source use.
- `architecture_overview.svg`: vector architecture diagram.
- `architecture_overview.png`: 1600-pixel-wide raster copy of the architecture diagram.
- `benchmark/`: benchmark tables, raw report, and static Plotly-generated PNG figures.
- `validation_benchmark_results.json`: three-seed validation-only report for the 13-model roster.
- `inference_example.py`: minimal local checkpoint loading example.
- `LICENSE`: Apache License 2.0 for project code and weights.
