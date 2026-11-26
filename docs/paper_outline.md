# Paper Plan (Provisional)

## Working Title

**Evaluating Spectral and Concept-Auxiliary Components in a Compact Convolutional Transformer**

Keep “neuro-symbolic,” “interpretable,” “robust,” and “calibrated” out of the title and abstract until the method and evidence support those claims.

## Current Evidence

The pilot is a single CIFAR-100 seed and one 50-epoch schedule: 59.29% test top-1, 85.40% top-5, 0.591 macro F1, with 1.80 hours reported training time on an RTX 5070. The test set was evaluated for both the 5-epoch and 50-epoch checkpoints, and the run did not use a validation split. Later one-epoch benchmark-pipeline smoke runs also evaluated corrected NeSy-CVT and ResNet-18 checkpoints on the official test loader; these smoke results are not research evidence. The CIFAR-100 test split has been inspected during development, so do not call it untouched. Freeze all choices before any further evaluation and add an independent dataset for confirmatory generalization claims. Existing values are historical pilot numbers only. They do not establish superiority, statistical reliability, robustness, calibration, or interpretability.

The architecture combines known ingredients—convolution, self-attention, Fourier-domain feature mixing, latent auxiliary features, and cosine prototypes. Prior work already studies convolutional vision Transformers, Fourier-domain classifiers, and explicit neuro-symbolic image classifiers. The paper's novelty must be established by a careful related-work review and evidence of a specific contribution, not by the combination's name.

Preliminary primary references to cover include [CvT](https://arxiv.org/abs/2103.15808), [Frequency Learning for Image Classification](https://arxiv.org/abs/2006.15476), [Resolution-Invariant Image Classification with Fourier Neural Operators](https://arxiv.org/abs/2304.01227), [Scattering Vision Transformer](https://arxiv.org/abs/2311.01310), [NeSyFOLD](https://arxiv.org/abs/2301.12667), [Language in a Bottle: Language Model Guided Concept Bottlenecks](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Language_in_a_Bottle_Language_Model_Guided_Concept_Bottlenecks_for_CVPR_2023_paper.pdf), [Concept-Centric Transformers](https://openaccess.thecvf.com/content/WACV2024/html/Hong_Concept-Centric_Transformers_Enhancing_Model_Interpretability_Through_Object-Centric_Concept_Learning_Within_a_Shared_Global_Workspace_WACV_2024_paper.html), [NeuroRule](https://openaccess.thecvf.com/content/CVPR2026/html/Zarar_NeuroRule_Bridging_Vision_and_Logic_with_Differentiable_Rule_Induction_CVPR_2026_paper.html), and [NUViT](https://arxiv.org/abs/2610.32041). Scattering Vision Transformer is especially relevant because it combines spectral mixing and attention and reports CIFAR-100 transfer experiments; Concept-Centric Transformers are relevant because they learn semantic concepts for classification and include CIFAR-100; NeuroRule illustrates a much more explicit learned-rule approach, although it focuses on scene graphs; NUViT studies Fourier-domain inputs rather than FFT mixing inside a conventional RGB classifier. These methods are related work, not directly comparable numbers for the proposed scratch-trained CIFAR-100 protocol. This list is a starting point, not a completed novelty or priority search.

## Research Questions

1. Under a fixed CIFAR-100 protocol, how does the compact model compare with CNN and ViT baselines in accuracy, calibration, memory, and speed?
2. Which components account for any accuracy/efficiency trade-off? Use the implemented FFT, Transformer, concept-auxiliary, and classifier ablations.
3. Does the latent concept branch represent human-meaningful concepts? This question is unanswered until concepts are semantically supervised or evaluated against concept annotations.
4. Does performance transfer to larger datasets or distribution shifts? This requires additional datasets and robustness evaluations.

## Method Sections

- State exact image preprocessing, architecture widths/depths, spectral gate equations, classifier, loss terms, and numerical settings.
- Describe the corrected spectral residual, which is identity when its gates are zero. Explicitly separate historical checkpoints using the previous doubled-residual behavior.
- Describe the learned concept-to-class projection accurately. It is an auxiliary classifier and is not a hand-specified or extracted logical rule set.
- Explain that the concept loss is inactive unless concept targets are supplied. CIFAR-100 training currently does not supply concept targets.

## Experiments

- Primary initial benchmark: CIFAR-100, fixed 45k/5k stratified train/validation split, official test set held until the final evaluation.
- Controlled models: NeSy-CVT, CIFAR-stem ResNet-18, MobileNetV3-Small, and small ViT; same input, augmentation, optimizer, batch size, epoch budget, and seeds.
- Three seeds minimum for the first controlled table; report means and standard deviations, not the best seed.
- Report top-1/top-5, macro F1, ECE, NLL, parameter count, checkpoint size, training wall time, throughput, inference latency, peak allocated GPU memory, and exact software/hardware.
- Run component ablations under the same protocol.
- Add a larger dataset and corruption/shift datasets before claims about general vision or robustness.
- Include one final official test evaluation after the method and settings are fixed. Keep literature values in a separate table and label them as not rerun.

## Interpretation and Claim Boundaries

The current 50-epoch pilot reaches 59.29% top-1 on CIFAR-100. That number is the only current real-data quality result. It is not a state-of-the-art claim, and it does not support claims about robustness, calibration, or explanations. If the model is to be called neuro-symbolic, introduce explicit semantic concepts and logical constraints or revise the terminology.

## Submission Readiness

An eventual journal submission needs a contribution supported by a broader literature review, controlled experiments, ablations, multiple seeds, and limitations. Q1 is a journal quartile in a specific ranking system, year, and subject category; it is not a property of an individual paper and cannot be guaranteed. Verify a journal’s current Scopus/CiteScore or SJR quartile in the relevant category when selecting a venue.

Before submission, identify the actual authors and affiliations, agree on authorship and corresponding author, check journal scope and current Q1 category, and follow its template, anonymization, data/code, AI-assistance, and dual-submission policies.
