# Manuscript Draft: NeSy-CVT

> **Status: working draft, not submission-ready.** A 13-model, three-seed CIFAR-100 comparison, detailed metrics table, and figures are now recorded below and in `docs/benchmark/cifar100-taxonomy-family/`. The official test set had prior exploratory exposure; these numbers are exploratory, not confirmatory. A related-work review, component ablations, independent-dataset evaluation, author details, and target-venue formatting are still required.

The repository includes an experimental CIFAR-100 taxonomy-reasoning path, six trained size presets, and a fixed-recipe controlled comparison. The draft must identify the exact checkpoint config and disclose prior test-set exposure; the older Tiny checkpoint uses a learned projection and is a different revision.

## Provisional title

**NeSy-CVT: A Taxonomy-Grounded Neuro-Symbolic Convolutional Vision Transformer for CIFAR-100**

This title describes the implemented fixed CIFAR-100 rule path. It does not make an accuracy, interpretability, or state-of-the-art claim. Complete a related-work review before claiming novelty.

## Authors and affiliations

`[Author names, order, affiliations, corresponding author, and ORCID IDs to be supplied by the authors.]`

## Abstract

`[Complete after the independent-dataset study and literature review.]` This work studies an image classifier that combines a convolutional stem, an identity-initialized Fourier feature mixer, Transformer blocks, a cosine-prototype classifier, and a supervised concept head. In taxonomy mode, a fixed CIFAR-100 fine-to-coarse map supplies differentiable class rules and a hierarchy-consistency loss. A fixed-recipe comparison evaluates 13 scratch-trained model configurations over three seeds. NeSy-CVT Huge reaches 69.58 ± 0.44% top-1 in this roster, with substantially higher estimated compute and training time than several references. The CIFAR-100 official test split was previously inspected in exploratory runs; these results are exploratory and do not establish confirmatory generalization or universal superiority. Independent data, component ablations, and a completed novelty review are needed before submission claims are finalized.

## 1. Introduction

`[Motivate the specific research gap only after completing the literature review.]` The current implementation is a research prototype. Its components—convolutional processing, spectral mixing, self-attention, cosine prototypes, supervised concepts, and taxonomy rules—have precedents in related areas. A paper must identify and empirically establish a distinct contribution beyond combining these ingredients.

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

The controlled comparison splits the official 50,000-image training set into 45,000 training and 5,000 validation images using 50 validation examples per class and split seed 1729. Checkpoints are selected by validation top-1. The compared checkpoints share this split, input transform, seed set, and official 10,000-image test examples. Report top-1/top-5, macro precision/recall/F1, macro one-vs-rest ROC AUC, superclass accuracy, class/superclass agreement, hierarchy consistency, 15-bin ECE, NLL, total/trainable parameters, checkpoint size, estimated FLOPs, training duration, batch-1 latency, batched FPS, pipeline throughput, and peak allocated GPU memory. Give means and sample standard deviations across seeds. The FLOP count is a documented analytical estimate, not an exact hardware count. The roster is bounded and does not represent every published CNN or Transformer.

The official CIFAR-100 test split has already been inspected during the historical pilot and one-epoch pipeline checks. Disclose that exploratory exposure; do not call the split an untouched or blinded confirmatory holdout. Add at least one additional independent dataset for confirmatory generalization claims. State that all baseline models are trained from scratch and that the shared recipe is a fixed-recipe comparison, not a claim that each baseline received optimal hyperparameter tuning.

## 5. Results

### Exploratory CIFAR-100 comparison

The expanded comparison evaluated 13 model configurations over seeds 42, 43, and 44. Every checkpoint used the same 45,000/5,000 train/validation split, training recipe, and official 10,000-image test set. Reported values are mean ± sample standard deviation across seeds. NeSy-CVT Huge had the highest top-1 accuracy in this roster (69.58 ± 0.44%) and macro one-vs-rest AUC of 98.89 ± 0.02%. DenseNet-121 reached 68.13 ± 0.43% top-1 with 7.05M parameters, an estimated 1.777G FLOPs/image, a 28.77MB checkpoint, and 1,597 model-only images/s at batch size 16. NeSy-CVT Huge used 20.63M parameters, an estimated 63.500G FLOPs/image, an 82.63MB checkpoint, and reached 121 model-only images/s at batch size 16; its mean training time was 19.06 hours per seed versus 1.68 hours for DenseNet-121. These results show a quality/resource trade-off, not an overall win for NeSy-CVT.

At the smaller end, NeSy-CVT Nano reached 51.31 ± 0.89% top-1 with 0.11M parameters and a 0.48MB checkpoint, compared with 47.47 ± 0.46% for MobileNetV3-Small at 1.62M parameters and 6.60MB. The Nano variant's estimated FLOPs/image were much higher (0.755G versus 0.011G), and its batch-16 throughput was lower (3,362 versus 4,561 images/s). This is another example where accuracy and stored model size do not describe runtime cost by themselves.

These CIFAR-100 test results are exploratory because this project's earlier pilot and smoke runs already inspected the official test split. They are not an untouched confirmatory result and should not be used to claim statistical or generalization superiority. The complete per-model metrics, Plotly Express HTML dashboard, static PNG plots, and tables are in `docs/benchmark/cifar100-taxonomy-family/`. Confirmatory claims still require an independent dataset/test set, choices fixed before evaluation, and the planned component ablations.

## 6. Ablations and analysis

Run the same fixed protocol for the no-FFT, no-Transformer, no-concept-head, no-rule-fusion, no-hierarchy-loss, and linear-classifier variants. Explain whether results support the importance of each component; do not infer causality from the full-model comparison alone. Add parameter-matched comparisons if making compactness claims.

## 7. Limitations and responsible use

The new taxonomy mode implements a fixed, inspectable label hierarchy, but the current experiments do not establish general logical reasoning, human interpretability, robustness, or calibrated confidence. Ablations are needed to isolate the taxonomy's contribution. CIFAR-100 is a small-image benchmark and is insufficient evidence for broad computer-vision generalization. The test split has prior exploratory exposure. The single-GPU performance measurements are hardware-specific and must not be compared directly with published numbers measured under different protocols.

## 8. Data, code, and reproducibility

Release the exact source revision, environment, benchmark JSON/CSV, per-run records, checkpoints or hashes, data split seed, and model card. The local run manifest records code and CIFAR-100 file hashes. Confirm dataset redistribution terms, code/weight license, author contributions, conflicts of interest, and target-journal policies before submission.

## 9. Submission checklist

- Replace all bracketed placeholders with verified information.
- Establish a specific contribution with the literature review and controlled experiments.
- Add the independent dataset and any evaluations needed for the claims.
- Supply author names, affiliations, ORCIDs, acknowledgments, and funding details.
- Choose a journal and verify its current quartile in the selected ranking system, year, and subject category.
- Format the manuscript and supplementary material for that journal, and follow its anonymization, preprint, data/code, and AI-assistance policies.
