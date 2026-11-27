| Model | Seeds | Validation top-1 (%) | Validation top-5 (%) | Precision macro (%) | Recall macro (%) | F1 macro (%) | AUC OVR macro (%) | ECE (15-bin) | NLL | Coarse top-1 (%) | Fine/coarse agreement (%) | Hierarchy consistency MAE | Parameters (M) | Estimated FLOPs (G/image) | Checkpoint size (MB) | Batch-1 latency p50 (ms) | FPS (model-only, batched) | Training peak GPU memory (MB) | Inference peak GPU memory (MB) | Training time (hours) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 1 | 5.44 | 20.58 | 3.83 | 5.44 | 2.19 | 75.09 | 0.03 | 4.22 | N/A | N/A | N/A | 27.90 | 0.182 | 111.64 | 2.93 | 4131.63 | 493.40 | 395.53 | 0.02 |
| NeSy-CVT w/o concept branch | 1 | 4.80 | 19.02 | 2.44 | 4.80 | 2.65 | 72.21 | 0.03 | 4.35 | N/A | N/A | N/A | 0.48 | 2.568 | 1.95 | 2.09 | 1564.83 | 638.83 | 147.84 | 0.04 |
| NeSy-CVT Tiny | 1 | 5.40 | 19.54 | 4.01 | 5.40 | 2.79 | 78.10 | 0.01 | 4.17 | N/A | 86.78 | 0.04 | 0.48 | 2.568 | 1.96 | 2.16 | 1556.56 | 638.87 | 147.90 | 0.04 |
| Swin-T | 1 | 3.64 | 13.22 | 0.68 | 3.64 | 0.86 | 68.13 | 0.01 | 4.42 | N/A | N/A | N/A | 27.60 | 0.169 | 110.67 | 5.65 | 2370.53 | 493.54 | 405.02 | 0.02 |
| Wide-ResNet-28-10 | 1 | 4.32 | 18.64 | 2.70 | 4.32 | 1.38 | 74.64 | 0.06 | 4.25 | N/A | N/A | N/A | 36.54 | 10.487 | 146.27 | 2.35 | 1819.74 | 796.28 | 545.92 | 0.03 |
