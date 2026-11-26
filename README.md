# NeSy-CVT

NeSy-CVT (Neuro-Symbolic Convolutional Vision Transformer) is a research model family for image classification. Its CIFAR-100 taxonomy mode combines convolutional and Transformer features with a 20-way semantic concept head and an explicit differentiable rule layer using the dataset's fine-to-coarse hierarchy. The legacy baseline and the currently exported Tiny checkpoint use a learned concept projection instead; their cards identify that distinction.

The taxonomy rule layer is specific to CIFAR-100 and only encodes its official fine-to-coarse hierarchy. It is not a general knowledge base. Interpretability, robustness, calibration, and accuracy improvements remain empirical questions. NeSy-CVT has not been shown to outperform all CNN or Transformer models, and no such claim is made.

## Current Pilot

One CIFAR-100 run trained for 50 epochs on an RTX 5070 achieved 59.29% top-1, 85.40% top-5, and 0.591 macro F1 on the official test split. Training took 1:47:57, as measured by the user. This is a pilot result, not a controlled comparison: it used one seed, no validation split, and the test set was also evaluated after the 5-epoch run. Benchmark-pipeline smoke validation later exercised the test loader for one epoch on the corrected model and a ResNet baseline; those numbers are not research results. Treat CIFAR-100 test metrics as exploratory because the split has already been inspected. The pilot used the historical FFT residual behavior. New training defaults to a corrected identity-at-initialization FFT residual, so the pilot checkpoint and future runs are different model revisions.

## Architecture

NeSy-CVT has five parts:

1. CNN stem for edges, textures, and local image features.
2. Transformer encoder for global relationships between image regions.
3. FFT spectral mixer using complex-valued gates over frequency-domain features.
4. Semantic concept branch: for CIFAR-100, predicts the 20 official coarse categories.
5. Cosine prototype classifier using normalized class vectors.
6. CIFAR-100 taxonomy reasoner: maps each fine class to its known superclass, adds the concept log-probability as a rule score, and trains a hierarchy consistency loss.

![NeSy-CVT architecture overview](docs/figures/architecture_overview.svg)

## Install

Library-only install:

```bash
pip install -e .
```

Training install:

```bash
pip install -e ".[train,dev]"
```

For an RTX 50-series GPU, install a CUDA-enabled PyTorch wheel (PyTorch 2.7 or newer) using the [official selector](https://pytorch.org/get-started/locally/) before installing the project extras. The project extras do not choose a CUDA wheel for you.

## Quickstart

### Model sizes and Python API

The public size presets are `nesy_cvt_nano`, `nesy_cvt_tiny`, `nesy_cvt_small`, `nesy_cvt_base`,
`nesy_cvt_large`, and `nesy_cvt_huge`. They create **randomly initialized** CIFAR-100 models with
the explicit taxonomy reasoner. Only a Tiny checkpoint is currently exported, and that older
checkpoint uses supervised concepts with a learned projection; it does not contain the new explicit
rule layer. Train and benchmark the taxonomy mode before presenting it as a trained improvement.

- Nano: width 64, 2 Transformer blocks, 4 heads (~0.115M parameters).
- Tiny: width 96, 4 Transformer blocks, 4 heads (~0.483M parameters).
- Small: width 128, 6 Transformer blocks, 4 heads (~1.264M parameters).
- Base: width 192, 8 Transformer blocks, 6 heads (~3.746M parameters).
- Large: width 256, 12 Transformer blocks, 8 heads (~9.930M parameters).
- Huge: width 320, 16 Transformer blocks, 10 heads (~20.629M parameters).

```python
import torch
from nesy_cvt import nesy_cvt_tiny

model = nesy_cvt_tiny(num_classes=100).eval()  # new, untrained model
images = torch.randn(2, 3, 32, 32)
with torch.inference_mode():
    output = model(images)
    predicted_class_ids = output.logits.argmax(dim=-1)
    predicted_superclasses = output.reasoning_trace["concept_probabilities"].argmax(dim=-1)
```

Load the trained Tiny checkpoint from the exported folder:

```python
from nesy_cvt import NesyCVTForImageClassification

model = NesyCVTForImageClassification.from_pretrained("hf_export/nesy-cvt").eval()
```

After the model is uploaded, the same loader accepts its Hub ID (install `pip install -e ".[hub]"`):

```python
model = NesyCVTForImageClassification.from_pretrained("YOUR_HF_USERNAME/nesy-cvt").eval()
```

The model takes float `NCHW` image tensors in the `[0, 1]` range and returns an output object; class
scores are in `output.logits`. The current trained weights are for 32×32 CIFAR-100 images and were
trained without mean/std normalization, so use that same preprocessing and the CIFAR-100 label map
when evaluating them. A new constructor has random weights; use `from_pretrained` to load trained
weights. For another number of classes, use the learned-projection mode explicitly, for example
`nesy_cvt_tiny(num_classes=10, reasoning_mode="learned_projection")`; the fixed CIFAR-100 taxonomy
reasoner only applies to CIFAR-100's 100 fine labels and 20 coarse concepts.

Train a taxonomy-reasoning size preset with the CLI (Tiny is the default):

```bash
nesy-cvt train --model-size tiny --dataset cifar100 --data-dir data --download --epochs 50 --batch-size 16 --device cuda --output-dir runs/nesy-cvt-tiny-cifar100
```

Use `nano`, `small`, `base`, `large`, or `huge` for the other presets. This basic `train` command
does not perform validation-based checkpoint selection; use `benchmark` for controlled comparisons.
Large variants require more GPU memory and training time. All six presets are new configurations,
not pretrained weights.

### Hugging Face variant layout

Use one model repository named `OWNER/nesy-cvt`. Keep the currently released checkpoint at the
repository root; put additional trained checkpoints in folders named for their size, such as
`base/` or `large/`. Export each variant after training, then upload the whole family folder. For
example:

```bash
nesy-cvt export --checkpoint-dir runs/cifar100-final-benchmark/nesy_cvt_base/seed-42 --output-dir hf_export/nesy-cvt/base --model-name "NeSy-CVT Base" --results-path results/cifar100-final-benchmark.json
nesy-cvt push-hf --repo-id OWNER/nesy-cvt --local-dir hf_export/nesy-cvt
```

After that checkpoint is present in the Hub repository, load it with:

```python
from nesy_cvt import NesyCVTForImageClassification

model = NesyCVTForImageClassification.from_pretrained(
    "OWNER/nesy-cvt", subfolder="base"
)
```

Replace `base` with `nano`, `tiny`, `small`, `large`, or `huge`. A Hub folder only has pretrained
weights after you export and upload a trained checkpoint there; the constructor alone creates
random weights. Keep each size's `config.json` beside its matching checkpoint.

To compare one taxonomy variant against the existing controls without touching the official test
split, run a validation-only benchmark first:

```bash
nesy-cvt benchmark --models nesy_cvt_tiny resnet18_cifar mobilenet_v3_small densenet121_cifar convnext_tiny_cifar swin_t_cifar vit_tiny_patch4 --seeds 42 43 44 --epochs 50 --batch-size 16 --device cuda --validation-only --download --runs-dir runs/tiny-taxonomy-validation --report results/tiny-taxonomy-validation.json
```

Rule fusion and consistency-loss weights are adjustable with `--taxonomy-rule-weight` and
`--taxonomy-consistency-weight`. Defaults 0.5 and 0.1 are unvalidated starting values; select them on
validation only. Set one to `0` to measure the matching ablation.

Export a checkpoint and its code/model card into a Hugging Face-ready folder:

```bash
nesy-cvt export --checkpoint-dir runs/cifar100-final-benchmark/nesy_cvt_coarse_concepts/seed-44 --output-dir hf_export/nesy-cvt
```

Run a short benchmark-pipeline smoke check (not scientific results):

```bash
nesy-cvt benchmark --models nesy_cvt resnet18_cifar --seeds 42 --epochs 1 --device cuda --download --validation-only --runs-dir runs/benchmark-smoke --report runs/benchmark-smoke.json
```

Check v0.2 coarse-concept supervision without evaluating the CIFAR-100 test split:

```bash
nesy-cvt benchmark --models nesy_cvt_coarse_concepts --seeds 42 --epochs 1 --device cuda --validation-only --runs-dir runs/cifar100-v02-validation --report results/cifar100-v02-validation.json
```

Run a small synthetic training smoke test after installing PyTorch:

```bash
nesy-cvt train --device auto --epochs 1 --synthetic-samples 256 --output-dir runs/nesy-cvt-tiny
nesy-cvt eval --device auto --checkpoint-dir runs/nesy-cvt-tiny
```

`--device auto` uses CUDA when PyTorch can access an NVIDIA GPU, otherwise CPU. Set
`--device cuda` to require CUDA or `--device cpu` to force CPU execution.

Run the full controlled CIFAR-100 comparison (four models, three seeds, 50 epochs per run):

```bash
nesy-cvt benchmark --device cuda --download --epochs 50 --batch-size 16 --seeds 42 43 44 --models nesy_cvt resnet18_cifar mobilenet_v3_small vit_tiny_patch4 --runs-dir runs/cifar100-comparison --report results/cifar100-comparison.json
```

The comparison uses a fixed, stratified 45,000/5,000 train/validation split, selects each run's checkpoint by validation top-1, then evaluates the official 10,000-image test set. Add the `nesy_cvt_no_*` and `nesy_cvt_linear_classifier` variants in a separate invocation for ablations. On the RTX 5070, the first model/seed run took about 1 hour 49 minutes; budget roughly a day for all 12 runs. Use `python -m nesy_cvt.resume_comparison` to keep completed runs and restart an interrupted seed from epoch 1.

Evaluate one trained checkpoint on the CIFAR-100 test split:

```bash
nesy-cvt eval --device cuda --dataset cifar100 --data-dir data --batch-size 16 --checkpoint-dir runs/cifar100-comparison/nesy_cvt/seed-42 --output-path results/nesy-cvt-seed42-test.json
```

## Benchmarking

“Top of all CNN and Transformer models” is not a measurable finite benchmark claim. Define a
dataset, input resolution, allowed training data/pretraining, compute budget, and a fixed set of
strong baselines first. The existing CIFAR-100 report is a limited controlled comparison, and its
test split has already been inspected. Do not turn it into a universal ranking or a confirmatory
paper claim.

Use two tables:

- Controlled benchmark: rerun NeSy-CVT and baselines with the same code, data, hardware, seeds, and image resolution.
- Literature SOTA reference: report published leaderboard numbers separately and mark them as not rerun.

Controlled baselines and ablations implemented by the benchmark command:

- CIFAR-stem ResNet-18
- MobileNetV3-Small
- small ViT with 4x4 patches
- NeSy-CVT without FFT, Transformer, or concept auxiliary branch
- NeSy-CVT with a linear classifier instead of cosine prototypes

## Publishing Order

1. GitHub for code, docs, and scripts.
2. Hugging Face Model Hub for model weights and model card.
3. PyPI for the Python library only.
4. arXiv after real experiments exist.
5. A top venue such as NeurIPS, CVPR, ICLR, ICCV, ECCV, ICML, or TMLR.

Check anonymity and dual-submission rules before public release around conference deadlines.
