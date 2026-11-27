# Paper figure and result archive

This archive keeps prior exploratory comparisons and the new v2 pipeline preflight in separate folders. It preserves the existing tracked comparison under `docs/benchmark/` and copies older local `results/` reports here. These files make no claim that the v2 study has finished.

Additional pilot and validation outputs that were previously only under the ignored local `results/` directory are copied into [`historical/`](historical/). This includes their raw JSON/CSV and existing report figures where available.

## `before-v1/`

Paper-ready re-render of the historical 13-configuration CIFAR-100 comparison. It includes the detailed table, raw metrics plots, Plotly Express HTML figures, PNG exports, and dashboard. The original tracked benchmark figures were preserved unchanged. The official test set had prior exploratory project-level exposure, so those test metrics are not confirmatory.

## `after-v2-preflight/`

One epoch, one seed (42), validation-only pipeline preflight for NeSy-CVT Tiny, the no-concept branch, Wide-ResNet-28-10, ConvNeXt-Tiny, and Swin-T on the NVIDIA GeForce RTX 5070. It exercised the `cifar100_standard_v2` transforms, SGD, cosine schedule, validation checkpointing, and report export. It did not evaluate the official test split. One epoch is far too short to compare learning quality; do not cite these values as model performance. The single seed provides no estimate of seed-to-seed uncertainty.

The JSON file preserves per-model epoch metrics, resource measurements, checkpoint hashes, and the run configuration. This preflight predates the corrected phase-specific memory capture, so its saved peak-memory value is a fit-phase high-water mark that may include validation. The full v2 runner measures training and inference peaks separately. The `report/` subfolder contains the validation-only Markdown/CSV table, interactive Plotly Express dashboard and charts, plus PNG exports.

## Rebuilding the reports

From the repository root, use new output paths if you want to preserve this archive:

```powershell
python -m nesy_cvt benchmark-report `
  --input docs/benchmark/cifar100-taxonomy-family/cifar100-taxonomy-detailed.json `
  --output-dir results/cifar100-paper-before-v1

python -m nesy_cvt benchmark-report `
  --input results/cifar100-paper-v2-preflight.json `
  --output-dir results/cifar100-paper-v2-preflight-report
```
