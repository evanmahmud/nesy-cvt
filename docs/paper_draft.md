# Manuscript Draft: NeSy-CVT

> **Status: working draft, not submission-ready.** Author details, a defensible novelty claim, completed experiments, results, figures, and target-journal formatting are still required. Do not cite smoke-run or exploratory test metrics as final results.

The repository now includes an experimental CIFAR-100 taxonomy-reasoning path and six size presets. These additions have not yet completed a controlled benchmark. The draft's claims and results must identify the exact checkpoint config; the previous Tiny checkpoint uses a learned projection and is a different revision.

## Provisional title

**NeSy-CVT: A Taxonomy-Grounded Neuro-Symbolic Convolutional Vision Transformer for CIFAR-100**

This title describes the implemented fixed CIFAR-100 rule path. It does not make an accuracy, interpretability, or state-of-the-art claim. Complete a related-work review before claiming novelty.

## Authors and affiliations

`[Author names, order, affiliations, corresponding author, and ORCID IDs to be supplied by the authors.]`

## Abstract

`[Complete only after the new taxonomy mode's multi-seed validation benchmark and an independent dataset are evaluated.]` This work studies an image classifier that combines a convolutional stem, an identity-initialized Fourier feature mixer, Transformer blocks, a cosine-prototype classifier, and a supervised concept head. In taxonomy mode, a fixed CIFAR-100 fine-to-coarse map supplies differentiable class rules and a hierarchy-consistency loss. We compare size variants with scratch-trained convolutional and Transformer references under a fixed protocol. ` [Insert verified results and uncertainty; make no universal ranking claim.] `The CIFAR-100 test split was inspected in exploratory runs, so final claims must disclose this exposure and rely on an independent dataset for confirmatory generalization claims.`

## 1. Introduction

`[Motivate the specific research gap only after completing the literature review.]` The current implementation is a compact classification prototype. Its components—convolutional processing, spectral mixing, self-attention, cosine prototypes, and a learned auxiliary classifier—have precedents in the literature. A paper must identify and empirically establish a distinct contribution beyond combining these ingredients.

The study should answer whether the proposed design has a reproducible accuracy/efficiency trade-off against suitable baselines and which components account for it. Claims about generalization, robustness, calibration, interpretability, or neuro-symbolic reasoning require separate evidence.

## 2. Related work

Cover convolutional vision Transformers, Fourier and spectral feature mixing, efficient image classifiers, concept bottlenecks, and neuro-symbolic visual reasoning. Close starting points include CvT (Wu et al., 2021), Scattering Vision Transformer (Patro and Agneeswaran, 2023), Concept-Centric Transformers (Hong et al., 2024), and NeuroRule (Zarar et al., 2026); BibTeX entries are in [`references.bib`](references.bib). Separate direct architectural comparisons from papers that solve different tasks or use pretraining. Complete a broader search before making priority or novelty claims.

## 3. Method

### Figure 1. Model overview

![NeSy-CVT model overview](figures/architecture_overview.svg)

The taxonomy path uses the 20 official CIFAR-100 coarse labels and their fixed mapping to the 100 fine classes. The previously released Tiny checkpoint predates this path.

### 3.1 Architecture

The default CIFAR-100 configuration uses 32x32 RGB images, an embedding width of 96, two depthwise residual blocks after a 3x3 convolutional stem, four Transformer blocks with four attention heads and an MLP expansion ratio of four, mean token pooling, and LayerNorm. The spectral mixer has per-channel real and imaginary gates initialized to zero. The classifier normalizes features and learned class prototypes and applies a learned, clamped cosine-logit scale.

For a feature map `x`, the spectral operation is

```text
X = rFFT2(x, norm="ortho")
G = tanh(A) + i tanh(B)
y = irFFT2(X * (1 + G), norm="ortho")
```

The current default returns `y`, so zero gates make the mixer an identity. Historical checkpoints without `spectral_mixer_mode` retain the former `x + y` behavior and must be described as a separate revision.

In the v0.1 baseline, the auxiliary head maps the 96-dimensional pooled feature to 32 latent values. After a sigmoid, a learned linear projection maps those values to class logits. It is trained with the class labels through an auxiliary cross-entropy term. CIFAR-100 training provides no semantic concept targets for this baseline.

The legacy `nesy_cvt_coarse_concepts` variant supervises 20 superclass values but retains its learned concept-to-class projection. The new size presets instead use an explicit taxonomy mode: the official fine-to-coarse map assigns a superclass to each fine class, and the log probability of that superclass contributes a rule score to the corresponding fine-class logit. The implementation returns the neural logits, concept probabilities, rule scores, and mapping in a reasoning trace.

### 3.2 Objective

For taxonomy mode, the objective is

```text
L = L_fused-class + 0.10 L_coarse-concept + 0.10 L_hierarchy + 0.05 L_prototype + 0.01 L_spectral
```

`L_hierarchy` is mean squared error between concept probabilities and superclass probabilities aggregated from the neural fine-class distribution. Fine logits are fused by adding `0.5 × log P(superclass(fine class) | image)` to the neural fine-class logits. The older learned-projection baseline uses its separate objective and must be reported as a separate model revision.

`L_class` is cross-entropy on cosine-prototype logits. `L_aux-class` is cross-entropy on the learned auxiliary projection. `L_prototype` is one minus cosine similarity between an image feature and its labeled prototype. `L_spectral` is the absolute difference between the mean spectral magnitudes before and after gating. The v0.1 baseline has no semantic concept targets. The experimental coarse-concept variant adds cross-entropy on the 20 CIFAR-100 superclass labels with weight 0.1; report this as a separate model revision.

### 3.3 Training and implementation

The controlled CIFAR-100 runner uses batch size 16, AdamW at learning rate 0.0003, gradient clipping at 1.0, float32, deterministic cuDNN settings, and no learning-rate scheduler. Training examples use `Resize(32)`, CIFAR-10 AutoAugment policy, and `ToTensor`. State exact hardware, software, data checksums, seeds, and code hashes alongside each reported result. Report the fixed rule system and consistency weights as hyperparameters.

## 4. Experimental protocol

The current controlled comparison splits the official 50,000-image training set into 45,000 training and 5,000 validation images using 50 validation examples per class and split seed 1729. Checkpoints are selected by validation top-1. The planned comparison includes the taxonomy variants, CIFAR-stem ResNet-18, MobileNetV3-Small, DenseNet-121, ConvNeXt-Tiny, Swin-Tiny, and a small patch-4 ViT for multiple seeds. It reports top-1, top-5, macro F1, superclass accuracy, class/superclass agreement, hierarchy consistency, 15-bin ECE, NLL, parameter count, checkpoint size, training duration, inference throughput/latency, and peak allocated GPU memory. This bounded roster is not every published CNN or Transformer.

The official CIFAR-100 test split has already been inspected during the historical pilot and one-epoch pipeline checks. Disclose that exploratory exposure; do not call the split untouched. Add at least one additional independent dataset for confirmatory generalization claims. State that all baseline models are trained from scratch and that the shared recipe is a controlled comparison, not a claim that each baseline received optimal hyperparameter tuning.

## 5. Results

`[Insert only after the full multi-seed run finishes and the report is audited.]` Include mean and sample standard deviation across seeds. Keep the one-epoch smoke scores out of the results section. Report failed, interrupted, or excluded runs and explain exclusions before examining test outcomes.

## 6. Ablations and analysis

Run the same fixed protocol for the no-FFT, no-Transformer, no-concept-head, no-rule-fusion, no-hierarchy-loss, and linear-classifier variants. Explain whether results support the importance of each component; do not infer causality from the full-model comparison alone. Add parameter-matched comparisons if making compactness claims.

## 7. Limitations and responsible use

The new taxonomy mode implements a fixed, inspectable label hierarchy but does not establish general logical reasoning, human interpretability, robustness, or calibrated confidence. Its accuracy effect is unknown until measured. CIFAR-100 is a small-image benchmark and is insufficient evidence for broad computer-vision generalization. The test split has prior exploratory exposure. The single-GPU performance measurements are hardware-specific and must not be compared directly with published numbers measured under different protocols.

## 8. Data, code, and reproducibility

Release the exact source revision, environment, benchmark JSON/CSV, per-run records, checkpoints or hashes, data split seed, and model card. The local run manifest records code and CIFAR-100 file hashes. Confirm dataset redistribution terms, code/weight license, author contributions, conflicts of interest, and target-journal policies before submission.

## 9. Submission checklist

- Replace all bracketed placeholders with verified information.
- Establish a specific contribution with the literature review and controlled experiments.
- Add the independent dataset and any evaluations needed for the claims.
- Supply author names, affiliations, ORCIDs, acknowledgments, and funding details.
- Choose a journal and verify its current quartile in the selected ranking system, year, and subject category.
- Format the manuscript and supplementary material for that journal, and follow its anonymization, preprint, data/code, and AI-assistance policies.
