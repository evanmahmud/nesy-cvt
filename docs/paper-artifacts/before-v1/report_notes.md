Charts use Plotly Express. Hover a point or bar for its values and seed count; the legend can hide or show model/metric series. Points have no printed labels, keeping close results readable. Error bars show sample standard deviation across 3 seeds.

The official test data were excluded from training and validation but were inspected in earlier project runs; this is not a pristine confirmatory holdout.

Each figure is available as an interactive HTML file and as a high-resolution PNG.

Evaluation split: same official CIFAR-100 10,000-image test split for every checkpoint

Precision, recall, F1, and AUC are macro-averaged by class; AUC is macro one-vs-rest. FLOPs are a documented analytical estimate. Checkpoint size is decimal MB. Batch-1 latency excludes data loading and host-to-device transfer; batched FPS is model-only. Training peak memory is measured during optimization; peak allocated inference memory is measured during evaluation.
