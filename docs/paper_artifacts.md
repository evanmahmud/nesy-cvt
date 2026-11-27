# Paper Result and Visualization Archive

This index keeps historical fixed-holdout results separate from the planned five-fold cross-validation study. Existing files are retained in their original locations; new reports use versioned paths and must not overwrite the historical artifacts. The study is designed to test a bounded set of comparisons; it does not guarantee or presume a state-of-the-art result.

## Before: historical exploratory results

### Primary 13-configuration comparison

- **Raw report:** [`cifar100-taxonomy-detailed.json`](benchmark/cifar100-taxonomy-family/cifar100-taxonomy-detailed.json)
- **Full table:** [`docs/benchmark/cifar100-taxonomy-family/comparison_table.md`](benchmark/cifar100-taxonomy-family/comparison_table.md)
- **CSV table:** [`docs/benchmark/cifar100-taxonomy-family/comparison_table.csv`](benchmark/cifar100-taxonomy-family/comparison_table.csv)
- **Original Plotly Express dashboard:** [`docs/benchmark/cifar100-taxonomy-family/benchmark_dashboard.html`](benchmark/cifar100-taxonomy-family/benchmark_dashboard.html)
- **Original static plots:** `quality_metrics.png`, `parameters.png`, `flops.png`, `size.png`, `latency.png`, `fps.png`, and `memory.png` in that directory; each has a matching Plotly HTML figure.
- **Paper-ready re-render:** [`paper-artifacts/before-v1/`](paper-artifacts/before-v1/) adds ECE/NLL and taxonomy metrics as separate figures and table columns. The original table and figures above were left unchanged.

### Other historical comparisons

- **Five-model pilot/test comparison:** raw JSON/CSV and the original report are archived in [`paper-artifacts/historical/five-model-pilot/`](paper-artifacts/historical/five-model-pilot/).
- **Earlier final-benchmark output:** raw JSON/CSV are archived in [`paper-artifacts/historical/final-benchmark/`](paper-artifacts/historical/final-benchmark/).
- **Validation-only taxonomy-family run:** JSON/CSV and a Plotly Express/PNG re-render are archived in [`paper-artifacts/historical/taxonomy-validation/`](paper-artifacts/historical/taxonomy-validation/). Its source aggregate only contains validation top-1/top-5 plus limited run-level resource data; unavailable test metrics appear as N/A.
- **Original taxonomy detailed render:** all existing files from `results/cifar100-taxonomy-detailed-report/` are preserved in [`paper-artifacts/historical/taxonomy-detailed-report/`](paper-artifacts/historical/taxonomy-detailed-report/). The matching raw JSON is already tracked in `docs/benchmark/cifar100-taxonomy-family/`.
- Checkpoints and epoch progress for the historical study remain on the local machine under `runs/cifar100-final-benchmark/` and `runs/cifar100-taxonomy-family/`; large checkpoints are not copied into this paper-figure archive.
- The older Tiny learned-projection checkpoint and later fixed-taxonomy Tiny are distinct revisions; keep them labeled separately.

## After: v2 recipe diagnostics and final runs

- **Preflight raw results:** [JSON](paper-artifacts/after-v2-preflight/cifar100-paper-v2-preflight.json) and [CSV](paper-artifacts/after-v2-preflight/cifar100-paper-v2-preflight.csv), with per-epoch logs/checkpoints under `runs/cifar100-paper-v2-preflight/`.
- **Preflight tables and plots:** [`paper-artifacts/after-v2-preflight/report/`](paper-artifacts/after-v2-preflight/report/). This is a one-epoch, one-seed validation-only pipeline check on five selected architectures; it is not a performance result and does not estimate seed variance.
- **Full v2 five-fold validation study:** reserved paths are `runs/cifar100-paper-v2-main/` (organized by model/fold/seed) and `results/cifar100-paper-v2-main-validation.json`. It uses the same deterministic stratified folds for every architecture and seed; validation summaries average folds within each seed before seed-level statistics.
- **Detailed evaluation and plots:** after model selection is locked, use a separate report path such as `results/cifar100-paper-v2-detailed.json` and `results/cifar100-paper-v2-detailed-report/`.
- **Learning curves:** use `benchmark-curves` to create separate “before” and “after” validation-curve facets from each run's `progress.json`.
- **Before/after metrics and figures:** use `benchmark-compare` to write `results/cifar100-paper-before-after/`. The command rejects reports with different test-data fingerprints.

## Saved paper figures

The archive folders contain the Markdown/CSV tables, Plotly Express HTML dashboard and charts, and matching PNG exports. The historical exploratory comparison is the “before” reference. The only completed v2 output so far is the one-epoch validation preflight; it is labeled as a pipeline check throughout. Do not present that preflight as a result table or place it in the same accuracy ranking as converged models.

- [Before v1 table](paper-artifacts/before-v1/comparison_table.md)
- [Before v1 interactive dashboard](paper-artifacts/before-v1/benchmark_dashboard.html)
- [Before v1 quality chart (PNG)](paper-artifacts/before-v1/quality_metrics.png)
- [V2 preflight table](paper-artifacts/after-v2-preflight/report/comparison_table.md)
- [V2 preflight interactive dashboard](paper-artifacts/after-v2-preflight/report/benchmark_dashboard.html)
- [V2 preflight quality chart (PNG)](paper-artifacts/after-v2-preflight/report/quality_metrics.png)

Existing images are never replaced by new runs. The comparison plots are descriptive: the CIFAR-100 official test set had prior project-level exposure, and the v2 recipe changes several training choices at once. They cannot identify a causal component or establish confirmatory SOTA by themselves.

## Paper-ready metric inventory

The v2 detailed-table schema includes top-1/top-5 accuracy, macro precision/recall/F1, macro one-vs-rest ROC AUC, 15-bin ECE, NLL, coarse top-1, fine/coarse agreement, hierarchy-consistency MAE, parameter counts, analytical FLOPs, checkpoint size, batch-1 latency, batched model-only FPS, pipeline images/s, training and inference peak allocated GPU memory, and training duration. The historical reports predate training-memory recording and show inference peak memory only. For cross-validation, report quality metrics averaged over folds within each seed and then mean/sample standard deviation across seeds; training time is summed and peak memory maximized within seed. Keep fold-level values and run configuration in JSON/CSV.

FLOPs are an analytical estimate under the repository's documented convention. The speed and memory figures are specific to the NVIDIA GeForce RTX 5070 and the recorded batch/precision settings.
