# Historical CIFAR-100 outputs

These are preserved copies of the project's pre-v2 outputs. They are exploratory and the official CIFAR-100 test set had prior project-level exposure. Keep each protocol labeled; these reports are not one pooled benchmark.

| Folder | Contents |
|---|---|
| `five-model-pilot/` | Five-model saved-checkpoint JSON/CSV and the original Plotly/PNG report. |
| `final-benchmark/` | Earlier five-model training JSON/CSV. |
| `taxonomy-validation/` | 13-model validation-only JSON/CSV and a Plotly/PNG re-render. Its source aggregate contains validation top-1/top-5 and limited run-level resource data; unavailable test metrics appear as N/A. |
| `taxonomy-detailed-report/` | Original detailed taxonomy report plots and tables. The associated raw JSON is already under `docs/benchmark/cifar100-taxonomy-family/`. |

Historical checkpoints and epoch logs remain in the original local `runs/` folders. The public paper archive preserves result files and visualizations, not checkpoint binaries.
