from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import NesyCVTConfig


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_model_card(
    output_dir: str | Path,
    config: NesyCVTConfig,
    *,
    model_name: str = "NeSy-CVT",
    results: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    benchmark_summary: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    card = output / "README.md"
    training_metadata = training.get("reproducibility", {}) if training else {}
    training_config = dict(training_config or {})
    if training and "metrics" in training:
        benchmark_record = training["metrics"]
        if results is None:
            results = benchmark_record.get("test")
        training_config = {
            **benchmark_record.get("config", {}),
            **training_config,
        }
        if not training_config.get("hardware") and benchmark_record.get("hardware"):
            training_config["hardware"] = benchmark_record["hardware"]
        if not training_config.get("pytorch") and benchmark_record.get("pytorch"):
            training_config["pytorch"] = benchmark_record["pytorch"]
        training = {
            **benchmark_record,
            "epochs": training.get("extra", {}).get("epochs", benchmark_record.get("epochs", [])),
        }
    if results and "metrics" in results:
        results = results["metrics"]

    architecture_figure = (
        "![NeSy-CVT architecture overview](architecture_overview.svg)"
        if config.reasoning_mode == "cifar100_taxonomy"
        and (output / "architecture_overview.svg").exists()
        else ""
    )

    eval_section = "No test-set evaluation was supplied for this export."
    if results:
        eval_section = "\n".join(
            [
                "Evaluation for this selected checkpoint on the official CIFAR-100 test split:",
                f"- Top-1 accuracy: {float(results['top1']) * 100:.2f}%",
                f"- Top-5 accuracy: {float(results['top5']) * 100:.2f}%",
                f"- Macro F1: {float(results['macro_f1']):.4f}",
                f"- Test examples: {int(results.get('samples', results.get('images', 10_000)))}",
            ]
        )
        if training and training.get("best_epoch_by_validation_top1"):
            eval_section += (
                "\nThe checkpoint was selected by validation top-1 at epoch "
                f"{int(training['best_epoch_by_validation_top1'])}."
            )
        eval_section += (
            "\nThese metrics describe this checkpoint only; they are not an aggregate across seeds "
            "or a state-of-the-art claim."
        )
    elif training and training.get("best_validation_top1") is not None:
        eval_section = (
            "Validation-only evaluation; the official CIFAR-100 test split was not evaluated for this run. "
            "Earlier exploratory runs of this project did inspect that test split, so this is not an untouched-test claim."
        )
        eval_section += (
            f"\n- Best validation top-1: {float(training['best_validation_top1']) * 100:.2f}%"
            f"\n- Validation top-5 at the selected epoch: {float(training.get('best_validation_top5', 0.0)) * 100:.2f}%"
        )
        if training.get("best_validation_coarse_top1") is not None:
            eval_section += (
                f"\n- Coarse-concept top-1 at the selected epoch: "
                f"{float(training['best_validation_coarse_top1']) * 100:.2f}%"
            )
        if training.get("best_validation_taxonomy_agreement") is not None:
            eval_section += (
                f"\n- Predicted class/superclass agreement: "
                f"{float(training['best_validation_taxonomy_agreement']) * 100:.2f}%"
            )
        if training.get("best_validation_hierarchy_consistency_mae") is not None:
            eval_section += (
                f"\n- Hierarchy consistency MAE: "
                f"{float(training['best_validation_hierarchy_consistency_mae']):.4f}"
            )
        if training.get("best_epoch_by_validation_top1"):
            eval_section += (
                "\nCheckpoint selected by validation top-1 at epoch "
                f"{int(training['best_epoch_by_validation_top1'])}."
            )

    benchmark_section = ""
    if benchmark_summary:
        aggregates = benchmark_summary.get("aggregate", {})

        def _mean_std(metric: str, *, percent: bool = False) -> str:
            values = aggregates.get(metric, {})
            mean = float(values.get("mean", float("nan")))
            sample_std = values.get("sample_std")
            scale = 100.0 if percent else 1.0
            suffix = "%" if percent else ""
            mean_text = f"{mean * scale:.2f}{suffix}"
            if sample_std is None:
                return f"{mean_text} (n={values.get('n', 0)}; standard deviation unavailable)"
            return f"{mean_text} ± {float(sample_std) * scale:.2f}{suffix}"

        seeds = benchmark_summary.get("seeds", [])
        evaluation_split = benchmark_summary.get("evaluation_split", "test")
        validation_report = evaluation_split == "validation"
        heading = "Controlled validation results" if validation_report else "Controlled multi-seed comparison"
        split_label = "validation" if validation_report else "official CIFAR-100 test"
        qualification = (
            "These validation results use the fixed split drawn from CIFAR-100's training set."
            if validation_report
            else (
                "These are results for the fixed CIFAR-100 protocol. The official test split was "
                "inspected in earlier exploratory runs, so treat this comparison as exploratory; "
                "it is not a state-of-the-art claim."
            )
        )
        benchmark_section = "\n".join(
            [
                f"\n## {heading}",
                (
                    f"Aggregate {split_label} results for `{benchmark_summary.get('model', 'model')}` "
                    f"over seeds {', '.join(str(seed) for seed in seeds)}:"
                ),
                f"- Top-1 accuracy: {_mean_std('best_validation_top1' if validation_report else 'top1', percent=True)}",
                f"- Top-5 accuracy: {_mean_std('best_validation_top5' if validation_report else 'top5', percent=True)}",
                *(
                    [f"- Coarse-concept top-1: {_mean_std('best_validation_coarse_top1', percent=True)}"]
                    if validation_report and "best_validation_coarse_top1" in aggregates
                    else []
                ),
                *(
                    [
                        f"- Class/superclass agreement: {_mean_std('best_validation_taxonomy_agreement', percent=True)}",
                        f"- Hierarchy consistency MAE: {_mean_std('best_validation_hierarchy_consistency_mae')}",
                    ]
                    if validation_report
                    and "best_validation_taxonomy_agreement" in aggregates
                    else []
                ),
                *(
                    []
                    if validation_report
                    else [
                        f"- Macro F1: {_mean_std('macro_f1')}",
                        f"- ECE (15 bins): {_mean_std('ece_15')}",
                    ]
                ),
                qualification,
            ]
        )

    training_section = "Training metadata was not supplied with this export."
    if training:
        epochs = training.get("epochs", [])
        epoch_count = (
            len(epochs)
            if isinstance(epochs, list) and epochs
            else training_config.get("epochs", training.get("epochs_requested", "not recorded"))
        )
        hardware = (
            training_config.get("hardware")
            or training_config.get("device")
            or training_metadata.get("hardware", "not recorded")
        )
        input_size = training_config.get("input_size", "not recorded")
        if isinstance(input_size, int):
            input_size = f"RGB {input_size}x{input_size}"
        training_section = "\n".join(
            [
                f"- Dataset: {training.get('dataset', 'CIFAR-100')}",
                f"- Epochs: {epoch_count}",
                f"- Batch size: {training_config.get('batch_size', 'not recorded')}",
                f"- Optimizer: {training_config.get('optimizer', 'not recorded')}",
                f"- Learning rate: {training_config.get('learning_rate', 'not recorded')}",
                f"- Input: {input_size}",
                f"- Training augmentation: {training_config.get('train_transform', 'not recorded')}",
                f"- Concept supervision: {training_config.get('concept_supervision', config.concept_schema)}",
                f"- Seed: {training_config.get('seed', training_metadata.get('seed', 'not recorded'))}",
                f"- Hardware: {hardware}",
                f"- PyTorch: {training_config.get('pytorch', 'not recorded')}",
            ]
        )
        training_seconds = training_config.get(
            "training_seconds", training.get("training_seconds_including_validation")
        )
        if training_seconds is not None:
            training_section += f"\n- Training time including validation: {float(training_seconds):.1f} seconds"

    if config.reasoning_mode == "cifar100_taxonomy":
        neuro_symbolic_section = (
            "This checkpoint uses a differentiable CIFAR-100 taxonomy reasoner. Its neural branch "
            "scores the 100 fine labels; a supervised concept branch scores the 20 official coarse "
            "superclasses. A fixed fine-to-coarse taxonomy supplies one rule per fine class, and the "
            "corresponding superclass log-probability adjusts that class score. A hierarchy consistency "
            "loss aligns the concept distribution with superclass probabilities aggregated from the "
            "fine-class branch. The rules are explicit and inspectable, but their knowledge is limited "
            "to this supplied CIFAR-100 taxonomy. This does not by itself prove human interpretability, "
            "causal reasoning, robustness, or improved accuracy."
        )
    elif config.concept_schema == "cifar100_coarse20":
        neuro_symbolic_section = (
            "This checkpoint has a 20-way concept head supervised by CIFAR-100 coarse superclass "
            "labels. Its concept-to-class projection is learned, so this checkpoint is a "
            "concept-supervised neuro-symbolic research baseline; it does not contain the explicit "
            "CIFAR-100 taxonomy reasoner available in newer model configs. It does not by itself "
            "prove human interpretability, causal reasoning, robustness, or improved accuracy."
        )
    else:
        neuro_symbolic_section = (
            "This checkpoint uses a latent concept head and a learned concept-to-class projection. "
            "Its concepts have no human-labeled meanings in this run, so describe it as a "
            "neuro-symbolic-inspired baseline rather than an explicit symbolic rule system."
        )

    variant_names = "Nano, Tiny, Small, Base, Large, Huge"
    variant_specs = """- Nano: width 64, 2 Transformer blocks, 4 heads (~0.115M parameters)
- Tiny: width 96, 4 Transformer blocks, 4 heads (~0.483M parameters)
- Small: width 128, 6 Transformer blocks, 4 heads (~1.264M parameters)
- Base: width 192, 8 Transformer blocks, 6 heads (~3.746M parameters)
- Large: width 256, 12 Transformer blocks, 8 heads (~9.930M parameters)
- Huge: width 320, 16 Transformer blocks, 10 heads (~20.629M parameters)"""
    variant_example_name = (
        config.model_size
        if config.model_size in {"nano", "tiny", "small", "base", "large", "huge"}
        else "tiny"
    )
    variant_imports = """from nesy_cvt import (
    nesy_cvt_nano, nesy_cvt_tiny, nesy_cvt_small,
    nesy_cvt_base, nesy_cvt_large, nesy_cvt_huge,
)

variant_constructors = {
    "nano": nesy_cvt_nano,
    "tiny": nesy_cvt_tiny,
    "small": nesy_cvt_small,
    "base": nesy_cvt_base,
    "large": nesy_cvt_large,
    "huge": nesy_cvt_huge,
}"""

    card.write_text(
        f"""---
library_name: pytorch
pipeline_tag: image-classification
license: apache-2.0
datasets:
- cifar100
metrics:
- accuracy
- f1
tags:
- computer-vision
- image-classification
- neuro-symbolic-ai
- hierarchical-image-classification
- cifar100
- convolution
- transformer
- fourier
---

# {model_name}

NeSy-CVT (Neuro-Symbolic Convolutional Vision Transformer) combines a convolutional stem, learnable FFT feature mixing, Transformer blocks, semantic concepts, and a cosine classifier. This repository is a CIFAR-100 research model; the exact reasoning mode and size are recorded in `config.json`.

## Model family and variants

Available architecture presets: **{variant_names}**. Constructors create random initial weights unless loaded with `from_pretrained`. Only weights actually present in this repository are pretrained; do not treat a size name as a claim that its checkpoint has been trained or benchmarked.

{variant_specs}

```python
{variant_imports}

model = variant_constructors["{variant_example_name}"](num_classes=100)  # random initialization
```

After trained variants are uploaded into this model repository under folders such as `tiny/` and `base/`, load one with `NesyCVTForImageClassification.from_pretrained("YOUR_HF_USERNAME/nesy-cvt", subfolder="base")`. The checkpoint stored at the repository root loads with `from_pretrained("YOUR_HF_USERNAME/nesy-cvt")`.

## Intended use

Research and educational use for small-image classification. The checkpoint was trained on CIFAR-100 and should not be assumed to generalize to other domains or resolutions.

## Evaluation

{eval_section}
{benchmark_section}

## Training details

{training_section}

## Neuro-Symbolic architecture and scope

{architecture_figure}

{neuro_symbolic_section}

The current checkpoint uses `model_size={config.model_size}`, `reasoning_mode={config.reasoning_mode}`, and `spectral_mixer_mode={config.spectral_mixer_mode}`. These settings are saved with its weights to reproduce the evaluated architecture.

## Load and run

Clone the model repository, install the listed requirements, and run the bundled smoke example:

```bash
git clone https://huggingface.co/YOUR_HF_USERNAME/nesy-cvt
cd nesy-cvt
pip install -r requirements.txt
python inference_example.py
```

From the cloned repository root (so its bundled `nesy_cvt` source is importable), load the checkpoint:

```python
import torch
from nesy_cvt import NesyCVTForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NesyCVTForImageClassification.from_pretrained("YOUR_HF_USERNAME/nesy-cvt")
model = model.eval().to(device)
pixels = torch.rand(1, 3, 32, 32, device=device)  # RGB, float in [0, 1]
with torch.inference_mode():
    output = model(pixels)
    prediction = {{"class_id": int(output.logits.argmax(dim=-1)[0])}}
    if output.reasoning_trace is not None:
        prediction["superclass_id"] = int(
            output.reasoning_trace["concept_probabilities"].argmax(dim=-1)[0]
        )
print(prediction)
```

For a released subfolder variant, pass `subfolder="nano"`, `"tiny"`, `"small"`, `"base"`, `"large"`, or `"huge"` to `from_pretrained`. The current CIFAR-100 weights use RGB 32×32 tensors in the `[0, 1]` range without mean/std normalization. The model returns logits and does not apply image transforms or map class IDs to names. Run `python inference_example.py` for the bundled local-folder example.

This repository uses the NeSy-CVT Python class; it is not registered as a Transformers `AutoModelForImageClassification` implementation. Use the bundled package API shown above. For human-readable labels, map the predicted class ID using CIFAR-100's official fine-label names.

## Files

- `config.json`: model configuration
- `pytorch_model.bin`: PyTorch state dictionary
- `nesy_cvt/`: source required to instantiate this model revision
- `architecture_overview.svg`: model-flow diagram, when available
- `training_metrics.json` and `evaluation_metrics.json`: raw run records, when supplied
- `validation_benchmark_results.json`: validation-only report, when supplied
- `benchmark_results.json`: multi-seed test report, when supplied
- `source_manifest.json`: source and dataset hashes, when supplied
- `inference_example.py`: minimal loading example
""",
        encoding="utf-8",
    )
    return card


def prepare_hf_export(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "NeSy-CVT",
    results_path: str | Path | None = None,
) -> Path:
    """Build a self-contained local Hugging Face model-repository folder."""
    import torch

    source = Path(checkpoint_dir)
    output = Path(output_dir)
    weights_path = source / "pytorch_model.bin"
    if not (source / "config.json").exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected config.json and pytorch_model.bin in checkpoint directory {source}"
        )
    config = NesyCVTConfig.from_pretrained(source)
    from .model import NesyCVTForImageClassification

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = NesyCVTForImageClassification(config)
    model.load_state_dict(state, strict=True)

    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    torch.save({key: value.detach().cpu() for key, value in state.items()}, output / "pytorch_model.bin")

    package_source = Path(__file__).resolve().parent
    shutil.copytree(
        package_source,
        output / "nesy_cvt",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    architecture_figure = (
        Path(__file__).resolve().parents[2] / "docs" / "figures" / "architecture_overview.svg"
    )
    if architecture_figure.is_file():
        shutil.copy2(architecture_figure, output / architecture_figure.name)
    (output / "requirements.txt").write_text(
        "torch>=2.7\ntorchvision>=0.22\nhuggingface_hub>=0.30\n",
        encoding="utf-8",
    )
    (output / "inference_example.py").write_text(
        """from pathlib import Path
import torch

from nesy_cvt import NesyCVTConfig, NesyCVTForImageClassification


def load_model(model_dir: str | Path | None = None):
    model_dir = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
    config = NesyCVTConfig.from_pretrained(model_dir)
    model = NesyCVTForImageClassification(config)
    weights = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    model = load_model()
    sample = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        logits = model(sample).logits
    print({"logits_shape": tuple(logits.shape), "predicted_class_id": int(logits.argmax(-1))})
""",
        encoding="utf-8",
    )

    training = _read_json(source / "metrics.json")
    if training:
        (output / "training_metrics.json").write_text(
            json.dumps(training, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    result_file = Path(results_path) if results_path else source / "test_metrics.json"
    results = _read_json(result_file)
    benchmark_report = None
    if results and (
        "aggregate_mean_and_sample_std" in results
        or "validation_aggregate_mean_and_sample_std" in results
    ):
        benchmark_report = results
        report_filename = (
            "validation_benchmark_results.json"
            if "validation_aggregate_mean_and_sample_std" in results
            else "benchmark_results.json"
        )
        (output / report_filename).write_text(
            json.dumps(benchmark_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results = None
    if results is None and training:
        results = training.get("metrics", {}).get("test")
    if results:
        (output / "evaluation_metrics.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    license_file = Path(__file__).resolve().parents[2] / "LICENSE"
    if license_file.exists():
        shutil.copy2(license_file, output / "LICENSE")
    run_config = _read_json(source / "run_config.json")
    if run_config is None and training:
        benchmark_record = training.get("metrics", {})
        run_config = {
            **benchmark_record.get("config", {}),
            "model": benchmark_record.get("model"),
            "seed": benchmark_record.get("seed"),
            "hardware": benchmark_record.get("hardware"),
            "python": benchmark_record.get("python"),
            "pytorch": benchmark_record.get("pytorch"),
            "torchvision": benchmark_record.get("torchvision"),
            "cuda_runtime": benchmark_record.get("cuda_runtime"),
            "platform": benchmark_record.get("platform"),
        }
        run_config = {key: value for key, value in run_config.items() if value is not None}
    if run_config:
        (output / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    manifest_candidates = [
        source / "source_manifest.json",
        source.parent.parent / "source_manifest.json",
    ]
    source_manifest = next((path for path in manifest_candidates if path.is_file()), None)
    if source_manifest is not None:
        shutil.copy2(source_manifest, output / "source_manifest.json")
    benchmark_summary = None
    if benchmark_report and training:
        record = training.get("metrics", {})
        model_key = record.get("model")
        evaluation_split = (
            "validation"
            if "validation_aggregate_mean_and_sample_std" in benchmark_report
            else "test"
        )
        aggregate_key = (
            "validation_aggregate_mean_and_sample_std"
            if evaluation_split == "validation"
            else "aggregate_mean_and_sample_std"
        )
        aggregate = benchmark_report.get(aggregate_key, {}).get(model_key)
        if aggregate:
            benchmark_summary = {
                "model": model_key,
                "seeds": benchmark_report.get("protocol", {}).get("seeds", []),
                "aggregate": aggregate,
                "evaluation_split": evaluation_split,
            }
    write_model_card(
        output,
        config,
        model_name=model_name,
        results=results,
        training=training,
        training_config=run_config,
        benchmark_summary=benchmark_summary,
    )
    return output


def push_to_hub(repo_id: str, local_dir: str | Path, *, private: bool = False) -> None:
    try:
        from huggingface_hub import create_repo, upload_folder
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to push model files to Hugging Face.") from exc
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    variant_dirs = ("nano", "tiny", "small", "base", "large", "huge")
    delete_patterns = [
        "**/__pycache__/**",
        "**/*.pyc",
        "evaluation_metrics.json",
        *(
            pattern
            for variant in variant_dirs
            for pattern in (
                f"{variant}/nesy_cvt/**",
                f"{variant}/validation_benchmark_results.json",
            )
        ),
    ]
    upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        repo_type="model",
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        delete_patterns=delete_patterns,
        commit_message="Update NeSy-CVT checkpoints, model card, and benchmark figures",
    )
