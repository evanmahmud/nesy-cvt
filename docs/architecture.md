# Architecture

NeSy-CVT is an image-classification family combining convolutional processing, global attention, FFT-based channel mixing, a cosine classifier, and a semantic concept branch. The CIFAR-100 taxonomy mode adds explicit differentiable reasoning over the official fine-to-coarse label hierarchy. Claims about general reasoning, robustness, calibration, interpretability, and accuracy gains still require experiments.

## Components

The CNN stem maps images into dense feature maps. Depthwise residual blocks keep local texture and edge information while controlling parameter count.

The FFT spectral mixer transforms feature maps into the frequency domain with `torch.fft.rfft2`, applies bounded learnable real and imaginary gates, and returns to the spatial domain with `torch.fft.irfft2`. New model configurations use a residual form that is the identity when the gates are zero. Older checkpoints without an explicit `spectral_mixer_mode` load in `legacy_double` mode to preserve their historical behavior.

The Transformer encoder flattens feature maps into tokens and applies multi-head self-attention. This gives the model global context after the CNN has already organized local evidence.

The legacy `nesy_cvt` baseline predicts latent concepts and uses a learned projection. `nesy_cvt_coarse_concepts` supervises 20 concepts using CIFAR-100 coarse labels but also retains a learned projection. The size variants `nesy_cvt_nano`, `nesy_cvt_tiny`, `nesy_cvt_small`, `nesy_cvt_base`, `nesy_cvt_large`, and `nesy_cvt_huge` use `reasoning_mode=cifar100_taxonomy` by default:

1. The neural classifier produces fine-class logits.
2. The concept head predicts 20 coarse superclass probabilities.
3. A fixed table maps each CIFAR-100 fine class to its official superclass.
4. The reasoner adds the log probability of the mapped superclass to each fine-class score.
5. A consistency loss aligns the 20 concept probabilities with the summed fine-class probabilities for each superclass.

The taxonomy is an explicit symbolic prior; the vision features and concept probabilities are learned. The reasoner is specific to CIFAR-100, and its correctness or accuracy benefit must be measured. The previously exported Tiny checkpoint predates this rule layer and remains loadable through its saved `reasoning_mode` configuration.

The cosine classifier normalizes image embeddings and learned class prototypes, then uses cosine similarity as logits. Prototype similarities are inspectable numerically, but this alone does not establish human interpretability.

## Data Flow

```text
image -> CNN stem -> FFT spectral mixer -> Transformer tokens -> pooled features
                                                          ├── cosine prototype classifier -> neural fine logits ─┐
                                                          └── 20 semantic concepts -> CIFAR-100 taxonomy rules ──┴-> fused logits
```

For taxonomy mode, concept logits correspond to the 20 CIFAR-100 coarse categories. The output includes an optional `reasoning_trace` with neural logits, concept probabilities, per-fine-class rule scores, and the fixed fine-to-coarse map. The benchmark records concept validation accuracy separately from fine-label accuracy.

## First Model Size

The size family offers Nano, Tiny, Small, Base, Large, and Huge presets. Their dimensions are architecture choices, not evidence that every variant is pretrained or better than a baseline. CIFAR-100 taxonomy reasoning cannot be applied to another label set unless its semantic concepts and rules are supplied.
