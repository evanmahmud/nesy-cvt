# Paper Outline and Claim Plan

## Current status

The project has a useful exploratory CIFAR-100 comparison, but it does not currently establish state-of-the-art (SOTA) performance. The official CIFAR-100 test split was inspected in earlier project runs, so existing test scores are not a pristine confirmatory result. Several reference runs also need a baseline-health review before they can support a ranking claim.

The detailed experiment roster, paper tables and figure plan, and requirements for a future SOTA claim are in [`paper_experiment_plan.md`](paper_experiment_plan.md). The manuscript in [`paper_draft.md`](paper_draft.md) is a working draft that reports current evidence with these limits.

## Proposed manuscript structure

1. **Introduction** — define the narrow problem: using a supplied class hierarchy to inform fine-grained image classification; state the hypotheses without presuming that they are true.
2. **Related work** — convolutional and hybrid vision backbones; spectral/Fourier methods; concept-based and hierarchical classification; explicit neuro-symbolic methods. Distinguish related ideas from direct comparisons.
3. **Method** — encoder, per-channel complex spectral gate, cosine-prototype classifier, CIFAR-100 coarse-concept head, fixed fine-to-coarse rule fusion, and training objective.
4. **Experimental protocol** — data splits, scratch/pretrained status, model lineup, seeds, training and validation schedule, hardware, software, metrics, and timing definitions.
5. **Results** — matched scratch-trained comparison; size scaling; component ablations; quality/resource trade-offs; uncertainty and failure analysis.
6. **Discussion** — what the hierarchy contributes, which resources it costs, and what the results do not establish.
7. **Limitations and reproducibility** — prior test-set exposure, baseline tuning, small-image benchmark scope, code/checkpoints, and exact versions.

## Claim gate

Use “SOTA” only for a clearly named benchmark setting and metric, after a current literature review and reproducible comparisons under matching data, pretraining, augmentation, model-selection, and evaluation rules. Until that gate is passed, use “SOTA target” for the research goal and report measured results as exploratory or as results within the explicitly listed roster.
