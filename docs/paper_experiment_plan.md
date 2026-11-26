# NeSy-CVT Paper: Architecture, Model Lineup, and SOTA Plan

## 1. What “SOTA” would mean here

“State of the art” is a scoped, evidence-based comparison, not a description of a model name. A defensible claim must name the dataset, metric, training-data regime, input resolution, and comparison date. It must distinguish scratch training from pretraining, additional data, test-time augmentation, and other extra resources. The claim must be supported by reproducible results against suitable current methods under comparable rules.

The current result does **not** establish SOTA. On the existing exploratory CIFAR-100 roster, NeSy-CVT Huge scored 69.58 ± 0.44% top-1, while DenseNet-121 scored 68.13 ± 0.43%. Huge used 20.63M parameters and an estimated 63.50G FLOPs/image; DenseNet-121 used 7.05M parameters and an estimated 1.777G FLOPs/image. These are results from the project's fixed recipe and selected roster, not a comparison against all current CIFAR-100 methods. In addition, the project's official test split was inspected in earlier runs. The values are therefore useful for diagnosis and matched analysis, but they cannot be described as an untouched confirmatory result.

The baseline roster needs inspection before it is used for any ranking claim. In particular, Swin-T's 1.19% top-1 is near random for 100 classes, and ConvNeXt-T's 31.48% is unexpectedly weak for a 50-epoch run. Treat those rows as failed or underfit reference runs until their data path, input adaptation, optimization, and learning curves have been checked. Do not delete or silently omit them from a report; document the diagnosis and rerun under a validated protocol.

## 2. Research questions and hypotheses

These are questions to test, not conclusions to write into the abstract in advance.

1. **Fine-label quality:** Does the fixed CIFAR-100 taxonomy path improve fine-class performance over an otherwise matched NeSy-CVT encoder?
2. **Coarse-label quality:** Does the supervised concept head improve prediction of the 20 coarse categories and class/superclass agreement?
3. **Resource trade-off:** How do accuracy, macro-F1, AUC, calibration, parameter count, estimated FLOPs, model size, latency, throughput, training time, and peak allocated GPU memory change across the six size presets?
4. **Component effect:** What changes when spectral gating, attention, concept supervision, rule fusion, hierarchy consistency, or the cosine classifier is removed?
5. **Generalization:** Does the method transfer to a separate dataset with an explicitly defined class hierarchy? This requires new data mapping and experiments; the released CIFAR-100 rule table cannot simply be reused.

## 3. Model lineup

### 3.1 Existing exploratory roster

Keep the existing 13-configuration, three-seed results in a clearly labeled exploratory section and supplement. Its shared 45,000/5,000 train/validation split, 32×32 evaluation images, 50-epoch budget, batch size 16, optimizer, and seed set make it a reproducible fixed-recipe comparison. The same official 10,000-image test examples were scored for all models, but the project had already inspected that test set.

| Group | Models in current artifacts | Role in the paper |
|---|---|---|
| NeSy-CVT size curve | Nano, Tiny, Small, Base, Large, Huge | Show the current accuracy/resource scaling curve. Do not imply every size is equally useful or that scaling alone proves a method contribution. |
| CNN references | CIFAR-stem ResNet-18, MobileNetV3-Small, CIFAR-stem DenseNet-121, ConvNeXt-Tiny | Standard reference families; validate each training and input path before drawing conclusions. |
| Transformer references | ViT-Tiny/4, Swin-Tiny | Transformer reference points. The current poor Swin run is a diagnostic result, not evidence that the architecture is weak. |
| Historical NeSy-CVT revisions | Learned-projection baseline and coarse-concept learned projection | Separate historical checkpoints from the newer fixed-taxonomy configurations. They are not interchangeable implementations of one model. |

### 3.2 Main controlled comparison to run next

For a compact main paper, use one designated NeSy-CVT size as the primary model (select it using validation performance and resource constraints, not test scores). Include the other five size presets in the scaling table or supplement. Use a validated, competitive set of scratch-trained references:

- NeSy-CVT primary size, plus the six-size scaling curve.
- ResNet-18 with the CIFAR stem and DenseNet-121 with the CIFAR stem.
- MobileNetV3-Small for the small-model efficiency point.
- ViT-Tiny/4 for a pure-transformer reference.
- ConvNeXt-Tiny and Swin-T only after their CIFAR-100 adaptation and training runs pass baseline sanity checks.
- Add a CIFAR-oriented high-performing CNN reference such as Wide-ResNet-28-10 or PreAct-ResNet, with an established CIFAR implementation and documented settings.

All scratch-trained runs should share exactly defined train/validation indices and evaluation images. Report two tracks if resources permit:

1. **Fixed-recipe track:** one shared schedule and optimizer, which answers how models behave under that recipe.
2. **Validation-tuned track:** a bounded, equal tuning budget per architecture (for example, the same number of learning-rate/schedule trials), selected only on validation data. This is the stronger track for a claim that one method outperforms well-trained references.

Do not use imported literature numbers as if they were reruns. Put them in a separate table with citations and record pretraining, extra data, resolution, augmentations, epochs, and test-time augmentation. Original ViT results, for example, use pretraining and transfer to smaller benchmarks; that is not directly comparable to a scratch run on 32×32 CIFAR images [Dosovitskiy et al., 2021].

### 3.3 Component ablations

Run these with the primary NeSy-CVT size, same data split, seeds, and training budget. Freeze all other settings. The current runner supports the FFT-off, Transformer-off, linear-classifier, rule-fusion-weight-zero, and hierarchy-consistency-weight-zero comparisons. Check the no-concept-head path before using it: taxonomy mode currently requires a concept head, so the existing `nesy_cvt_no_concept_auxiliary` name may not represent a valid isolated ablation without an implementation/configuration change.

| Ablation | Question answered | Interpretation limit |
|---|---|---|
| FFT gate off | Does the spectral gate add value? | Removing it also changes compute, so report both accuracy and resource metrics. |
| Transformer off | Does global attention help after the convolutional stem? | This is a different capacity profile; parameter counts must be shown. |
| Cosine classifier → linear | Does prototype normalization matter? | Keep initialization and schedule documented. |
| Rule fusion weight = 0 | Does the fixed superclass probability change fine-class decisions? | The concept head and its supervised loss remain active. |
| Hierarchy-consistency weight = 0 | Does the consistency objective help? | Rule fusion and concept supervision remain active. |
| Remove concept head and dependent rule path | Is the semantic branch useful as a whole? | This ablates the coupled branch, not only one isolated tensor; it needs a valid config and exact description. |
| Coarse-supervision off | Does ground-truth superclass supervision matter? | Keep the label hierarchy fixed; do not switch to the legacy learned projection silently. |

An ablation that changes several components at once should be labeled as a branch ablation, not evidence for one individual component. A useful factorial subset is rule fusion on/off × hierarchy loss on/off × coarse supervision on/off.

## 4. Protocol needed for a stronger claim

1. **Freeze the research question and comparison scope.** Specify “scratch-trained CIFAR-100 fine classification, no external data or pretraining” or a different precise regime. A broad “beats CNNs and Transformers” statement is not a defined benchmark.
2. **Validate data and training paths.** Check class counts, labels, transformations, model input sizes, logits, per-epoch train/validation curves, and that each reference can learn substantially above chance. Investigate the current weak ConvNeXt and Swin runs before rerunning the full roster.
3. **Use validation for decisions.** Keep a stratified train/validation split for model selection. Set augmentation, optimizer, schedule, rule weights, checkpoint rule, and epoch count using only training/validation data. Log every tried setting.
4. **Train enough to converge.** The current 50-epoch AdamW 3e-4 schedule has no learning-rate decay. It is a fixed comparison recipe, not a strong CIFAR-100 recipe for every architecture. Compare a longer schedule with a documented decay schedule and suitable CIFAR augmentation. Give each architecture a bounded, comparable validation-tuning budget.
5. **Repeat seeds.** Keep at least three seeds for the main table and preferably five for the final primary comparison and key ablations. Report mean, sample standard deviation, and confidence intervals where appropriate. For test-set differences, use paired uncertainty analysis on the same examples; do not interpret seed standard deviation alone as a significance test.
6. **Lock evaluation.** Use the official CIFAR-100 test set only after configuration selection. Because this project previously inspected it, state that fact in the paper. Do not make further development choices from the test metrics. Add a separate dataset with a documented hierarchy to evaluate transfer; that supports a generalization claim, not by itself CIFAR-100 SOTA.
7. **Keep comparisons comparable.** Use the same test examples and label mapping. State scratch/pretrained status, extra training data, resolution, augmentations, number of epochs, checkpoint selection, test-time augmentation, batch size, and precision. Do not mix timing from unlike hardware.
8. **Archive the experiment.** Record the Git commit, Python/PyTorch/CUDA/driver versions, GPU and CPU, random seeds, split indices or their hash, dataset file hash, full configuration, per-epoch logs, checkpoints or hashes, and evaluation JSON/CSV.

## 5. Metrics and reporting layout

### Main quality table

- Top-1 and top-5 accuracy.
- Macro precision, recall, and F1 over the 100 fine classes.
- Macro one-vs-rest ROC AUC from softmax probabilities.
- Negative log-likelihood and expected calibration error (state the binning convention).
- For taxonomy models: coarse top-1, fine/coarse agreement, and hierarchy-consistency error.

### Resource table

- Total/trainable parameters (M), checkpoint size (MB), estimated FLOPs (G/image), batch-1 device latency (p50/p95), model-only batched FPS, end-to-end pipeline images/s, training duration, and peak allocated GPU memory.
- Mark FLOPs as an analytical estimate with the exact counting convention; they exclude some elementwise and backend work.
- Keep batch-1 latency (milliseconds per image) separate from batched throughput. State that current single-image latency excludes data loading and host-to-device transfer.
- State the RTX 5070, software versions, precision, and timing batch size beside every speed figure.

Report per-seed results as supplementary material. Include uncertainty bars and avoid putting labels directly over crowded points. The existing Plotly Express figures and PNGs are a starting point, not a substitute for updated figures after the final benchmark.

## 6. Proposed paper tables and figures

| Item | Contents | Status |
|---|---|---|
| Figure 1 | Architecture: convolutional stem → depthwise residual blocks → per-channel complex FFT gate → Transformer encoder → cosine classifier; parallel 20-way superclass head and fixed fine-to-coarse rule fusion | Existing SVG and PNG; update caption to call the FFT operation a per-channel complex spectral gate, not a frequency-specific mixer. |
| Figure 2 | Data split, validation-based selection, final evaluation, and leakage disclosure | Create for the manuscript. |
| Figure 3 | Fine accuracy and macro-F1 against parameters and estimated FLOPs | Existing Plotly Express source figures; regenerate from final locked runs. |
| Figure 4 | Accuracy against batch-1 latency and batch throughput | Existing interactive and PNG figures; preserve distinct latency/throughput definitions. |
| Figure 5 | Calibration and hierarchy metrics, including coarse accuracy and agreement | Some taxonomy metrics are present in source JSON; build a focused final figure after runs are validated. |
| Figure 6 | Component ablations and seed-level uncertainty | Not complete; wait for valid, confirmed ablation runs. |

| Table | Contents |
|---|---|
| Table 1 | Exact model configurations, initialization/pretraining, CIFAR input adaptation, parameter count. |
| Table 2 | Main quality metrics, mean ± SD and seed count. |
| Table 3 | Resource and deployment metrics, measured on the RTX 5070 under stated settings. |
| Table 4 | Ablation results and differences from the primary model. |
| Supplement | Per-seed values, training curves, confusion matrices, class-level results, configs, and run hashes. |

## 7. Paper claim wording

Until the SOTA gate passes, suitable wording is:

> “In an exploratory, fixed-recipe CIFAR-100 comparison, NeSy-CVT Huge achieved 69.58 ± 0.44% top-1 accuracy among the 13 evaluated configurations. It required substantially more estimated computation and training time than DenseNet-121. The official test set had been inspected in earlier project runs; this result is not a pristine confirmatory estimate.”

Avoid “state of the art,” “best model,” “outperforms all CNNs/Transformers,” “interpretable,” and “general neuro-symbolic reasoning” in the title, abstract, README, and model card unless the relevant evidence is completed. A narrower result such as “highest top-1 in our named, fixed-recipe roster” is only appropriate after all listed methods pass baseline-health checks and the result has been independently reproduced.

## 8. Go/no-go gates for a SOTA claim

- [ ] Reproduce current checkpoints and validate reported numbers from the saved JSON and artifacts.
- [ ] Diagnose and rerun weak/failed reference models; establish a CIFAR-100 high-performing scratch baseline.
- [ ] Add appropriate high-performing CIFAR-specific CNN and current Transformer references.
- [ ] Tune each reference and NeSy-CVT with a bounded equal validation budget; use a suitable longer schedule.
- [ ] Run primary model and key baselines over five seeds; predefine the primary metric and resource limits.
- [ ] Complete taxonomy and architecture ablations, including a valid no-concept branch setting.
- [ ] Freeze all design choices before the final evaluation; document previous test exposure.
- [ ] Complete a dated literature review of directly comparable CIFAR-100 results and separate incompatible protocols.
- [ ] Confirm the result is still best for the defined regime and metric; otherwise report the trade-off or negative result.
- [ ] Add a separate hierarchical dataset before any cross-dataset or broad generalization claim.

Passing these gates would permit a scoped SOTA claim if the final result supports it. It cannot guarantee a Q1 journal or A* conference decision; those depend on novelty, evidence, venue scope, and review.
