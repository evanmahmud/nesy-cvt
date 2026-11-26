# Release Preparation

## Artifacts Prepared in the Repository

- Python package metadata and CLI.
- CIFAR-100 pilot record with training and test metrics separated.
- Controlled benchmark runner and result formats.
- Hugging Face export command that bundles config, checkpoint, source package, metrics, and an inference example.
- Model-card template with evaluation, training details, intended use, and limitations.
- Paper outline and methods-first draft with claim boundaries and explicit result placeholders.

## Final Checks Before a Public Release

### Research and checkpoint

- Rerun the corrected spectral-residual model; the 50-epoch checkpoint is from the historical implementation.
- Complete the validation-selected, multi-seed baseline comparison and component ablations.
- Freeze the training recipe before any further CIFAR-100 test evaluation. The official CIFAR-100 test split has already been inspected in the pilot and pipeline smoke runs; do not describe it as untouched, and add an independent dataset for confirmatory generalization claims.
- The existing Tiny checkpoint predates the explicit taxonomy-reasoning path. Its card must identify its `reasoning_mode=learned_projection` and must not imply that those saved weights use fixed symbolic rules.
- New size constructors use the explicit CIFAR-100 fine-to-coarse taxonomy rule layer. Train and validation-benchmark each released size before adding its weights to the Hub repo; do not describe random-initialized presets as pretrained variants.
- Report superclass accuracy, class/superclass agreement, and hierarchy-consistency metrics for taxonomy-mode checkpoints. The rule system only encodes CIFAR-100's taxonomy; it is not a general symbolic knowledge base.
- Do not claim “top of all CNNs and Transformers.” Publish the exact benchmark roster, compute/training protocol, uncertainty, and limitations. The current test split is not a clean confirmatory split.
- Confirm that all datasets and pretrained weights used are legal to redistribute, and document their licenses.

### GitHub source

- Add the real GitHub owner/repository URLs to `pyproject.toml` once the repository location is known.
- Add an actual remote, tagged version, license, citation file, and release notes.
- Run tests and package build in CI; verify the sdist contains source and docs but excludes local data/checkpoints.
- Keep experiment JSON/CSV and exact config files with the paper artifacts.

### Hugging Face model

- Create the model repository under the intended personal/organization namespace.
- Review `README.md` metadata, license, dataset, model revision, measured results, and limitations.
- The Hub README must explain the neuro-symbolic rule path and disclose whether the uploaded checkpoint actually enables it.
- Keep the base `OWNER/nesy-cvt` checkpoint at the repository root and upload additional trained sizes only under matching `nano/`, `tiny/`, `small/`, `base/`, `large/`, or `huge/` folders with a matching config and weights file. Verify each with `from_pretrained(..., subfolder="size")`.
- Download the exported folder into a clean environment and run `inference_example.py` before upload.
- Authenticate locally with `hf auth login` using a write-enabled token; do not store the token in the repository.
- For staging, use `nesy-cvt push-hf --repo-id NAMESPACE/REPOSITORY --local-dir EXPORTED_FOLDER --private`.
- Upload only the selected checkpoint and its matching config/source. Do not upload local datasets or unrelated checkpoints.
- Use a private repository for a staging review if desired; switch public only after checking the artifacts and rights.

### Python package index

- Check that the `nesy-cvt` name is available and select a release version.
- Build and inspect wheel and source distribution, test installation from TestPyPI, then publish through a configured trusted publisher.
- Configure the PyPI trusted publisher for the GitHub repository, workflow `.github/workflows/publish-pypi.yml`, and GitHub environment `pypi`; the workflow requires a published release tag matching the version in `pyproject.toml`.
- Keep PyTorch as an optional dependency and provide GPU-specific install instructions rather than promising one wheel works for every CUDA driver.

### DOI/archive and paper

- Link the tagged GitHub release to Zenodo for a versioned source archive and DOI after the repository is public.
- Prepare a manuscript source, figures, supplementary tables, and reproducibility statement. Do not upload to a preprint server until the author list and submission timing are approved.
- Select a journal only after confirming scope and its current quartile in the chosen ranking source, year, and subject category. Quartile status can differ by category and ranking system; acceptance cannot be guaranteed.
- Check the target journal's current policies on prior preprints, anonymous review, data/code, authorship, and simultaneous submissions.

## Metadata Needed from the Owner

- Author name(s), order, affiliations, corresponding-author email, and ORCID IDs.
- GitHub owner/repository and Hugging Face namespace/repository name.
- Copyright holder and confirmation that Apache-2.0 is the intended code and weight license.
- Whether the first release should be private for review or public.
- Preferred target journal and the ranking source/category that defines “Q1” for the intended evaluation.
