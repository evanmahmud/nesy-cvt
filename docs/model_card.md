# Hugging Face Model Card Draft

The export command generates the upload-ready `README.md` from the selected checkpoint and its metrics. Use the exported card as the source of truth for a model repository; do not upload this draft independently.

## Pilot values

- Dataset: CIFAR-100 fine labels.
- Input: 32x32 RGB.
- Training: 50 epochs, batch 16, AdamW at 3e-4, seed 42, CIFAR10 AutoAugment policy.
- Test: top-1 59.29%, top-5 85.40%, macro F1 0.5914 on 10,000 images.
- Hardware/time: RTX 5070, user-reported 1:47:57.
- Revision: `legacy_double` spectral residual.

This is one exploratory pilot with no validation split, and the test set was examined after an earlier 5-epoch checkpoint. It is not a controlled benchmark and must not be presented as a final or SOTA result.

## Limitations

The existing Tiny checkpoint has a concept vector supervised with CIFAR-100 coarse labels, but its concept-to-class projection remains learned. New size presets use an explicit differentiable CIFAR-100 taxonomy mode: 20 supervised superclass concepts, a fixed mapping from fine labels to superclasses, rule scores added to fine-class logits, and a hierarchy-consistency loss. That rule layer is a real symbolic prior, but it is limited to this taxonomy. The newer mode has not been shown to increase accuracy until it is trained and evaluated against the same baselines. Neither mode has been validated on larger datasets, corruption benchmarks, or domain shifts. Interpretability, calibration, and robustness claims require additional evidence.
