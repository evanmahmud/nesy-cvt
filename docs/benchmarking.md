# Benchmark Protocol

## Status of the Existing Pilot

The 50-epoch RTX 5070 result is a pilot, not a controlled comparison. It used one seed, trained on all 50,000 official training images, selected 50 epochs without a validation set, and evaluated the official test set after a 5-epoch run had already been evaluated. It reports 59.29% top-1, 85.40% top-5, and 0.591 macro F1. In addition, one-epoch pipeline smoke runs exercised the official test loader for the corrected NeSy-CVT and ResNet-18. Their metrics are validation artifacts, not research evidence. The CIFAR-100 test split has therefore already been inspected; do not describe it as untouched, and use an additional independent dataset for confirmatory generalization claims. Preserve pilot values as historical results, not final estimates or comparisons against baselines.

The pilot checkpoint predates `spectral_mixer_mode` in its config. The loader keeps its historical `x + IFFT(FFT(x))` behavior. New checkpoints default to an identity-at-zero-gates residual. The old and corrected configurations must be reported as different model revisions.

## Controlled CIFAR-100 Comparison

Run the `nesy-cvt benchmark` command. It uses a fixed stratified split of the 50,000 official training images: 45,000 for training and 5,000 for validation, with 50 validation images from each of the 100 classes. The split is held constant across model runs. Each seed controls initialization, augmentation randomness, and training order.

Every model uses 32x32 inputs, CIFAR-10 AutoAugment policy, `ToTensor`, batch size 16, AdamW with learning rate 0.0003, gradient clipping at 1.0, and 50 epochs. The best checkpoint is selected by validation top-1. The official 10,000-image test split is evaluated after training. Do not use test metrics to select hyperparameters.

The historical controlled comparison includes:

- NeSy-CVT with the corrected spectral residual.
- ResNet-18 with a CIFAR stem.
- MobileNetV3-Small.
- DenseNet-121 with a CIFAR stem.
- ConvNeXt-Tiny.
- Swin-Tiny.
- A small ViT with 4x4 patches.
- `nesy_cvt_coarse_concepts`, an experimental variant with a 20-dimensional concept head supervised by the CIFAR-100 coarse superclass labels.

The original `nesy_cvt` remains the latent-concept baseline. `nesy_cvt_coarse_concepts` adds supervised superclass concepts but keeps a learned projection. New `nesy_cvt_nano`, `nesy_cvt_tiny`, `nesy_cvt_small`, `nesy_cvt_base`, `nesy_cvt_large`, and `nesy_cvt_huge` variants use an explicit, fixed CIFAR-100 fine-to-coarse rule layer plus a hierarchy-consistency loss. These are new, untrained model configurations; they have no measured advantage until benchmarked. DenseNet-121, ConvNeXt-Tiny, and Swin-Tiny are available as additional selectable references. The roster is stronger but still finite and cannot establish superiority over every image model.

### Short validation-only check

Use this first for the taxonomy-reasoning Tiny variant. It trains one seed and skips construction and evaluation of the official test split:

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark `
  --device cuda --epochs 1 --batch-size 16 --seeds 42 `
  --models nesy_cvt_tiny `
  --validation-only `
  --runs-dir runs/cifar100-v02-validation `
  --report results/cifar100-v02-validation.json
```

This is a pipeline/learning check, not a publishable benchmark. It writes the validation-selected checkpoint, validation metrics, and a report that explicitly records `validation_only: true`. Use validation-only runs to compare the taxonomy family, legacy variants, and baselines during development. Do not select the release checkpoint using test metrics.

Run three seeds (42, 43, 44) and report mean and sample standard deviation. For the architecture ablation, run the same settings with `nesy_cvt_no_fft`, `nesy_cvt_no_transformer`, `nesy_cvt_no_concept_auxiliary`, and `nesy_cvt_linear_classifier`.

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark `
  --device cuda --download --epochs 50 --batch-size 16 `
  --seeds 42 43 44 `
  --models nesy_cvt_tiny resnet18_cifar mobilenet_v3_small densenet121_cifar `
    convnext_tiny_cifar swin_t_cifar vit_tiny_patch4 `
  --validation-only `
  --runs-dir runs/cifar100-comparison `
  --report results/cifar100-comparison.json
```

The command writes a JSON report and a flat CSV. Each run records validation-selected epoch, test top-1/top-5, macro F1, 15-bin ECE, NLL, inference throughput, batch latency percentiles, peak allocated GPU memory, parameter count, checkpoint size, training time, model config, seed, and software/hardware metadata. FLOPs are not currently reported; do not estimate them from parameter count.

`--taxonomy-rule-weight` and `--taxonomy-consistency-weight` expose the rule-fusion and hierarchy-loss strengths. Their defaults (0.5 and 0.1) are starting values, not validated optima. Tune them only on the fixed validation split; set either to `0` for its ablation. Record every tried configuration and do not select by the previously inspected CIFAR-100 test split.

An equivalent invocation for ablations should use a separate output path:

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark `
  --device cuda --epochs 50 --batch-size 16 --seeds 42 43 44 `
  --models nesy_cvt nesy_cvt_no_fft nesy_cvt_no_transformer `
    nesy_cvt_no_concept_auxiliary nesy_cvt_linear_classifier `
  --runs-dir runs/cifar100-ablations `
  --report results/cifar100-ablations.json
```

Each run uses one GPU and one fixed recipe. The comparison is controlled under that recipe; it is not a claim that the recipe is optimal for every model. The standard `benchmark` command writes the best validation checkpoint and progress after each epoch, but does not save optimizer state or resume a partial seed. Start with one seed and one epoch to check runtime and memory before requesting the full roster; ConvNeXt-Tiny, Swin-Tiny, and Huge are more demanding.

No finite experiment can establish that a model is “top of all CNN and Transformer models.” Define a bounded roster and matched training-data/compute rules. The current runner compares a small set of references, not every published architecture. A CIFAR-100 result also cannot support rankings on other datasets or input resolutions.

### Recovering an interrupted comparison

Use `resume_comparison` with the same settings and output paths after an interruption:

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt.resume_comparison `
  --device cuda --epochs 50 --batch-size 16 --seeds 42 43 44 `
  --models nesy_cvt resnet18_cifar mobilenet_v3_small vit_tiny_patch4 `
  --runs-dir runs/cifar100-comparison `
  --report results/cifar100-comparison.json
```

The recovery launcher validates and keeps complete model/seed records, then restarts each incomplete model/seed from epoch 1 because its optimizer state was not saved. It writes the combined report and CSV only after every requested run has completed.

## Metrics

Report classification quality (top-1, top-5, macro F1), calibration (ECE and NLL), efficiency (training wall time, inference throughput, batch latency, peak allocated GPU memory, parameters, checkpoint bytes), and uncertainty across seeds. Keep training-time and inference-time measurements separate. Record batch size, warm-up count, precision mode, input size, and whether data-loading time is included.

The current harness measures evaluation throughput with data loading included and per-batch latency around transfer plus model execution. It runs float32 inference. The reported memory value is PyTorch's peak allocated memory for that process, not total card utilization.

## Additional Evidence Needed for a Paper

- Compare against suitable recent baselines with the same split, preprocessing, epoch budget, and seeds; also state whether baselines are scratch-trained or pretrained.
- Add at least one larger or more realistic dataset after the CIFAR-100 protocol is stable.
- Use corruption/shift tests before making robustness claims, and calibration metrics before making calibration claims.
- Add concept semantics and explicit rule evaluation before describing the model as interpretable or neuro-symbolic. The current concept branch is a learned auxiliary head.
- For taxonomy-mode runs, report fine accuracy, coarse-concept accuracy, rule/hierarchy consistency, and ablations that remove the rule scores and consistency loss. The CIFAR-100 test split has prior exploratory exposure, so use validation and independent data for development and confirmation.
- Run component ablations and report uncertainty; include a compute/parameter budget comparison.
- Freeze settings using validation data. Reserve test data for final reporting.
- Publish scripts, exact environment, data split definition, seeds, raw run JSON/CSV, and checkpoints or hashes.

Do not mix literature values with reruns. Cite source, dataset, resolution, training recipe, and whether the number is reproduced. CIFAR-100 contains 50,000 training and 10,000 test images across 100 fine classes; see the [CIFAR-100 dataset description](https://www.cs.toronto.edu/~kriz/cifar.html).
