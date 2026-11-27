# Benchmark Protocol

## Status of the Existing Pilot

The 50-epoch RTX 5070 result is a pilot, not a controlled comparison. It used one seed, trained on all 50,000 official training images, selected 50 epochs without a validation set, and evaluated the official test set after a 5-epoch run had already been evaluated. It reports 59.29% top-1, 85.40% top-5, and 0.591 macro F1. In addition, one-epoch pipeline smoke runs exercised the official test loader for the corrected NeSy-CVT and ResNet-18. Their metrics are validation artifacts, not research evidence. The CIFAR-100 test split has therefore already been inspected; do not describe it as untouched, and use an additional independent dataset for confirmatory generalization claims. Preserve pilot values as historical results, not final estimates or comparisons against baselines.

The pilot checkpoint predates `spectral_mixer_mode` in its config. The loader keeps its historical `x + IFFT(FFT(x))` behavior. New checkpoints default to an identity-at-zero-gates residual. The old and corrected configurations must be reported as different model revisions.

## Controlled CIFAR-100 Comparison

The historical runs in this repository used a fixed stratified 45,000/5,000 holdout. For the new paper study, use stratified **5-fold cross-validation** over all 50,000 official training images. Each fold trains on about 40,000 images and validates on 10,000 (100 per class). Fold assignments are deterministic (seed 1729) and shared across every model and training seed. The benchmark command now defaults to `--folds 5`; pass `--folds 1` only to reproduce the historical holdout behavior.

The cross-validation run directory is organized as `model/fold-N/seed-N`. The official CIFAR-100 test split is not loaded during training when `--validation-only` is set. Once the training and model-selection plan is locked, `benchmark-eval` can score all fold checkpoints on the same official 10,000 test images. Report validation CV results as primary; test results are secondary and exploratory because this project inspected the test split in earlier runs. Quality and timing metrics are averaged within each seed across folds; training time is summed and peak memory is maximized within each seed. The report then gives mean and sample standard deviation across seeds, with every fold still listed in JSON and CSV.

For a short, test-free five-fold pipeline check:

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark `
  --device cuda --epochs 1 --batch-size 16 --seeds 42 --folds 5 `
  --models nesy_cvt_tiny `
  --validation-only `
  --runs-dir runs/cifar100-cv-preflight `
  --report results/cifar100-cv-preflight.json
```

This check fits five one-epoch models. It checks the CV pipeline and is not a performance result.

The earlier fixed-holdout comparisons are retained as historical exploratory results; do not mix their holdout metrics with the new five-fold CV estimates as if they were one experiment.

Within a given historical holdout run, checkpoints shared the same train/validation indices, evaluation transform, official 10,000-image CIFAR-100 test examples, and seed set. In the new cross-validation study, each fold is a separate train/validation partition shared across all models and seeds, and every fold checkpoint can be scored on the same official test examples after selection. Earlier NeSy-CVT pilot and smoke evaluations inspected this official test split, so it is no longer a blinded, untouched test set for this project. Results on it are useful for matched comparison but should not be presented as confirmatory evidence from a pristine holdout.

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

The original `nesy_cvt` remains the latent-concept baseline. `nesy_cvt_coarse_concepts` adds supervised superclass concepts but keeps a learned projection. The `nesy_cvt_nano`, `nesy_cvt_tiny`, `nesy_cvt_small`, `nesy_cvt_base`, `nesy_cvt_large`, and `nesy_cvt_huge` variants use an explicit, fixed CIFAR-100 fine-to-coarse rule layer plus a hierarchy-consistency loss. Trained checkpoints and a 13-model, three-seed exploratory comparison are included in this repository. The roster is finite and cannot establish superiority over every image model; the shared recipe does not guarantee architecture-specific optimal tuning.

### Short validation-only check

Use this first for the taxonomy-reasoning Tiny variant. It trains one seed over all five folds and skips construction and evaluation of the official test split:

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark `
  --device cuda --epochs 1 --batch-size 16 --seeds 42 --folds 5 `
  --models nesy_cvt_tiny `
  --validation-only `
  --runs-dir runs/cifar100-v02-validation `
  --report results/cifar100-v02-validation.json
```

This is a pipeline/learning check, not a publishable benchmark. It writes the validation-selected checkpoint, validation metrics, and a report that explicitly records `validation_only: true`. Use validation-only runs to compare the taxonomy family, legacy variants, and baselines during development. Do not select the release checkpoint using test metrics.

Run at least three seeds (42, 43, 44) with five folds and report the average across folds within each seed, then mean and sample standard deviation across seeds. For the architecture ablation, run the same settings with `nesy_cvt_no_fft`, `nesy_cvt_no_transformer`, `nesy_cvt_no_concept_auxiliary`, and `nesy_cvt_linear_classifier`.

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark `
  --device cuda --download --epochs 50 --batch-size 16 --folds 5 `
  --seeds 42 43 44 `
  --models nesy_cvt_tiny resnet18_cifar mobilenet_v3_small densenet121_cifar `
    convnext_tiny_cifar swin_t_cifar vit_tiny_patch4 `
  --validation-only `
  --runs-dir runs/cifar100-comparison `
  --report results/cifar100-comparison.json
```

Each completed run records the validation-selected epoch, training configuration, seed, and software/hardware metadata. A test-evaluating run records test metrics and performance measurements; `--validation-only` skips the official test split and records validation metrics only. The current saved-checkpoint `benchmark-eval` report adds macro precision/recall/F1, macro one-vs-rest AUC, estimated FLOPs, checkpoint size, batch-1 latency, batched FPS, training peak allocated GPU memory (when the original run metadata contains it), inference peak allocated memory, training time, and the shared test-data fingerprint. Historical runs do not contain training-peak-memory metadata. FLOPs are analytical estimates under the documented convention below, not estimates inferred from parameter count.

`--taxonomy-rule-weight` and `--taxonomy-consistency-weight` expose the rule-fusion and hierarchy-loss strengths. Their defaults (0.5 and 0.1) are starting values, not validated optima. Tune them only using the training and validation folds; set either to `0` for its ablation. Record every tried configuration and do not select by the previously inspected CIFAR-100 test split.

An equivalent invocation for ablations should use a separate output path:

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark `
  --device cuda --epochs 50 --batch-size 16 --seeds 42 43 44 --folds 5 `
  --models nesy_cvt nesy_cvt_no_fft nesy_cvt_no_transformer `
    nesy_cvt_no_concept_auxiliary nesy_cvt_linear_classifier `
  --validation-only `
  --runs-dir runs/cifar100-ablations `
  --report results/cifar100-ablations.json
```

Each run uses one GPU and one fixed recipe. The comparison is controlled under that recipe; it is not a claim that the recipe is optimal for every model. The `benchmark` command writes the best validation checkpoint and progress after each epoch. With `--resume`, it reuses completed model/fold/seed runs whose recorded settings and GPU match, and preserves the checkpoint/progress files of an interrupted fold before restarting that fold from epoch 1. It does not restore optimizer state or continue from the interrupted epoch. Start with one seed, one epoch, and five folds to check runtime and memory before requesting the full roster; ConvNeXt-Tiny, Swin-Tiny, and Huge are more demanding.

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

The older recovery launcher is for the historical single-holdout layout. For the new cross-validation layout, rerun the same `benchmark` command with `--resume`; it reuses completed model/fold/seed records and restarts incomplete folds from epoch 1 because optimizer state is not saved.

## Metrics

The detailed saved-checkpoint comparison reports top-1 and top-5 accuracy, macro precision, macro recall, macro F1, macro one-vs-rest ROC AUC, calibration (ECE and NLL), trainable/total parameters, checkpoint size, estimated FLOPs, batch-1 latency, batched inference FPS, evaluation-pipeline throughput, training and inference peak allocated GPU memory, and training time. Score metrics are aggregated over the same seeds as mean ± sample standard deviation. Precision, recall, and F1 are unweighted macro averages over the 100 CIFAR-100 classes; AUC is the unweighted mean of 100 one-vs-rest ROC AUC values computed from softmax probabilities.

`single_image_latency_*` is CUDA-event model execution time on one image already resident on the GPU; it excludes data loading and host-to-device transfer. `inference_fps` is model-only wall-time throughput at the common evaluation batch size. `pipeline_images_per_second` includes test-loader iteration and transfers. The report records batch size and warm-up count. All reported inference uses float32. Inference peak memory is PyTorch's peak allocated memory during evaluation; training peak memory is measured during optimization. Neither is total card utilization.

FLOPs are a static analytical estimate per 32×32 image: Conv2d and Linear multiply-adds, attention matrix operations, cosine-classifier multiplication, and an approximate 2-D FFT count are included. Normalization, activations, softmax, pooling, most elementwise work, and backend-specific implementation details are omitted. The value is therefore a consistently defined estimate, not an exact hardware instruction count. Checkpoint size is `pytorch_model.bin` bytes divided by 1,000,000 (decimal MB).

### Re-score existing trained checkpoints and make plots

This does not train the models again. It checks that the chosen run directory contains the same training/validation protocol and seeds, then evaluates each saved checkpoint against the same official CIFAR-100 test images. The report stores a SHA-256 fingerprint of those test pixels and labels.

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark-eval `
  --device cuda --batch-size 16 `
  --runs-dir runs/cifar100-final-benchmark `
  --data-dir data `
  --report results/cifar100-detailed.json

.\.venv-win\Scripts\python.exe -m pip install -e ".[reports]"
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark-report `
  --input results/cifar100-detailed.json `
  --output-dir results/cifar100-detailed-report
```

The output folder contains detailed Markdown and CSV tables, a self-contained interactive Plotly Express dashboard, individual interactive HTML figures, and high-resolution PNG versions of every figure. Five-fold reports also include `fold_scores.html` and `fold_scores.png`, showing each fold/seed observation. The PNG exporter uses Kaleido and a local Chrome/Chromium installation. Hover for exact values in HTML and use the legend to show or hide models and metrics. No labels are placed on top of nearby points. Missing metrics remain `N/A`; FLOPs are explicitly labeled analytical estimates.

The existing `runs/cifar100-final-benchmark` roster contains NeSy-CVT, its coarse-concept variant, ResNet-18, MobileNetV3-Small, and ViT-Tiny/4, with seeds 42, 43, and 44. The separate `runs/cifar100-taxonomy-family` directory contains validation-only checkpoints for more variants. They can also be rescored with the same command by changing `--runs-dir` and `--report`, but those results should be labeled exploratory because the CIFAR-100 test split has already been inspected in earlier project runs.

The expanded family report, figures, and tables were generated with:

```powershell
.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark-eval `
  --device cuda --batch-size 16 `
  --runs-dir runs/cifar100-taxonomy-family `
  --data-dir data `
  --report results/cifar100-taxonomy-detailed.json

.\.venv-win\Scripts\python.exe -m nesy_cvt benchmark-report `
  --input results/cifar100-taxonomy-detailed.json `
  --output-dir results/cifar100-taxonomy-detailed-report
```

Matching data, split, input size, seeds, and a shared training recipe makes the comparison reproducible and controlled under that recipe. It does not prove that the recipe is equally optimal for each architecture. Report this limit, and avoid claiming general superiority from a single dataset or metric. In particular, compare accuracy against parameters, estimated FLOPs, checkpoint size, latency, FPS, and allocated GPU memory.

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
