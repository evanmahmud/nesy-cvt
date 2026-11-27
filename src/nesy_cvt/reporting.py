from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


TABLE_COLUMNS = (
    ("Accuracy (%)", "top1", "percent"),
    ("Top-5 (%)", "top5", "percent"),
    ("Precision macro (%)", "precision_macro", "percent"),
    ("Recall macro (%)", "recall_macro", "percent"),
    ("F1 macro (%)", "macro_f1", "percent"),
    ("AUC OVR macro (%)", "auc_ovr_macro", "percent"),
    ("ECE (15-bin)", "ece_15", "identity"),
    ("NLL", "nll", "identity"),
    ("Coarse top-1 (%)", "coarse_top1", "percent"),
    ("Fine/coarse agreement (%)", "taxonomy_agreement", "percent"),
    ("Hierarchy consistency MAE", "hierarchy_consistency_mae", "identity"),
    ("Parameters (M)", "num_parameters", "millions"),
    ("Estimated FLOPs (G/image)", "flops_g_per_image", "gflops"),
    ("Checkpoint size (MB)", "model_size_mb", "identity"),
    ("Batch-1 latency p50 (ms)", "single_image_latency_p50_ms", "identity"),
    ("FPS (model-only, batched)", "inference_fps", "identity"),
    ("Training peak GPU memory (MB)", "train_peak_gpu_memory_mb", "identity"),
    ("Inference peak GPU memory (MB)", "peak_gpu_memory_mb", "identity"),
    ("Training time (hours)", "training_seconds_including_validation", "hours"),
)

EFFICIENCY_PLOTS = (
    ("num_parameters", "Parameters (M)", "millions", True, "parameters"),
    ("flops_g_per_image", "Estimated FLOPs (G/image)", "identity", True, "flops"),
    ("model_size_mb", "Checkpoint size (MB)", "identity", True, "size"),
    (
        "single_image_latency_p50_ms",
        "Batch-1 latency p50 (ms)",
        "identity",
        False,
        "latency",
    ),
    ("inference_fps", "Model-only FPS (batch=protocol size)", "identity", False, "fps"),
    ("peak_gpu_memory_mb", "Peak allocated GPU memory (MB)", "identity", True, "memory"),
    (
        "train_peak_gpu_memory_mb",
        "Training peak allocated GPU memory (MB)",
        "identity",
        True,
        "train_memory",
    ),
)


def _aggregate(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate_mean_and_sample_std") or report.get(
        "validation_aggregate_mean_and_sample_std"
    )
    if not isinstance(aggregate, dict) or not aggregate:
        raise ValueError("The report has no aggregate_mean_and_sample_std section")
    for metrics in aggregate.values():
        if "top1" not in metrics and "best_validation_top1" in metrics:
            metrics["top1"] = metrics["best_validation_top1"]
        if "top5" not in metrics and "best_validation_top5" in metrics:
            metrics["top5"] = metrics["best_validation_top5"]
    training_memory: dict[str, list[float]] = {}
    for run in report.get("runs", []):
        value = run.get("train_peak_gpu_memory_mb")
        if value is not None:
            training_memory.setdefault(str(run["model"]), []).append(float(value))
    for model_name, values in training_memory.items():
        model_metrics = aggregate.setdefault(model_name, {})
        model_metrics.setdefault(
            "train_peak_gpu_memory_mb",
            {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
            },
        )
    return aggregate


def _label(model_name: str) -> str:
    known = {
        "nesy_cvt": "NeSy-CVT (latent)",
        "nesy_cvt_coarse_concepts": "NeSy-CVT (coarse concepts)",
        "nesy_cvt_nano": "NeSy-CVT Nano",
        "nesy_cvt_tiny": "NeSy-CVT Tiny",
        "nesy_cvt_small": "NeSy-CVT Small",
        "nesy_cvt_base": "NeSy-CVT Base",
        "nesy_cvt_large": "NeSy-CVT Large",
        "nesy_cvt_huge": "NeSy-CVT Huge",
        "nesy_cvt_no_fft": "NeSy-CVT w/o spectral gate",
        "nesy_cvt_no_transformer": "NeSy-CVT w/o Transformer",
        "nesy_cvt_no_concept_auxiliary": "NeSy-CVT w/o concept branch",
        "nesy_cvt_linear_classifier": "NeSy-CVT linear classifier",
        "nesy_cvt_no_rule_fusion": "NeSy-CVT w/o rule fusion",
        "nesy_cvt_no_hierarchy_consistency": "NeSy-CVT w/o hierarchy consistency",
        "nesy_cvt_no_coarse_supervision": "NeSy-CVT w/o coarse supervision",
        "resnet18_cifar": "ResNet-18",
        "mobilenet_v3_small": "MobileNetV3-Small",
        "densenet121_cifar": "DenseNet-121",
        "convnext_tiny_cifar": "ConvNeXt-Tiny",
        "swin_t_cifar": "Swin-T",
        "vit_tiny_patch4": "ViT-Tiny/4",
        "wide_resnet28x10_cifar": "Wide-ResNet-28-10",
    }
    return known.get(model_name, model_name.replace("_", "-").title())


def _convert(value: float, conversion: str) -> float:
    if conversion == "percent":
        return value * 100.0
    if conversion == "millions":
        return value / 1_000_000.0
    if conversion == "hours":
        return value / 3600.0
    return value


def _summary_cell(metric: dict[str, Any] | None, conversion: str, digits: int = 2) -> str:
    if not metric or metric.get("mean") is None:
        return "N/A"
    mean = _convert(float(metric["mean"]), conversion)
    sample_std = metric.get("sample_std")
    if sample_std is None:
        return f"{mean:.{digits}f}"
    std = _convert(float(sample_std), conversion)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _columnar(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row records to Plotly's dependency-free dict-of-columns input."""
    return {key: [row[key] for row in rows] for key in rows[0]} if rows else {}


def _seed_counts(aggregate: dict[str, Any]) -> set[int]:
    return {
        int(metrics["top1"]["n"])
        for metrics in aggregate.values()
        if isinstance(metrics.get("top1"), dict) and metrics["top1"].get("n")
    }


def _uncertainty_notice(aggregate: dict[str, Any]) -> str:
    counts = _seed_counts(aggregate)
    fold_counts = {
        int(metric.get("folds", 1))
        for metrics in aggregate.values()
        if isinstance(metric := metrics.get("top1"), dict)
    }
    fold_count = next(iter(fold_counts)) if len(fold_counts) == 1 else 1
    fold_note = (
        f" Fold metrics are averaged within each seed across {fold_count} folds first."
        if fold_count > 1
        else ""
    )
    if counts == {1}:
        return "Only one seed was run per model; seed-to-seed uncertainty is not estimated." + fold_note
    if len(counts) == 1:
        return f"Error bars show sample standard deviation across {next(iter(counts))} seeds." + fold_note
    return "Seed counts vary by model; error bars show sample standard deviation where estimable." + fold_note


def _markdown_table(
    aggregate: dict[str, Any], *, score_split: str = "test", protocol: dict[str, Any] | None = None
) -> tuple[str, list[dict[str, Any]]]:
    columns = list(TABLE_COLUMNS)
    columns[0] = (f"{score_split.title()} top-1 (%)", "top1", "percent")
    columns[1] = (f"{score_split.title()} top-5 (%)", "top5", "percent")
    protocol = protocol or {}
    headers = ["Model", "CV design", *(column[0] for column in columns)]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    csv_rows: list[dict[str, Any]] = []
    for model_name, metrics in sorted(aggregate.items()):
        top1 = metrics.get("top1", {})
        seed_count = int(top1.get("n", 0))
        fold_count = int(protocol.get("folds", top1.get("folds", 1)))
        cv_design = f"{fold_count} folds × {seed_count} seeds" if fold_count > 1 else f"{seed_count} seeds"
        row = {"Model": _label(model_name), "CV design": cv_design}
        values = []
        for title, key, conversion in columns:
            digits = 3 if conversion == "gflops" else 2
            cell = _summary_cell(metrics.get(key), conversion, digits)
            row[title] = cell
            values.append(cell)
        lines.append("| " + " | ".join([row["Model"], cv_design, *values]) + " |")
        csv_rows.append(row)
    return "\n".join(lines) + "\n", csv_rows


def _write_tables(
    output_dir: Path,
    aggregate: dict[str, Any],
    *,
    score_split: str = "test",
    protocol: dict[str, Any] | None = None,
) -> None:
    protocol = protocol or {}
    markdown, rows = _markdown_table(aggregate, score_split=score_split, protocol=protocol)
    (output_dir / "comparison_table.md").write_text(markdown, encoding="utf-8")
    table_columns = list(TABLE_COLUMNS)
    table_columns[0] = (f"{score_split.title()} top-1 (%)", "top1", "percent")
    table_columns[1] = (f"{score_split.title()} top-5 (%)", "top5", "percent")
    columns = ["Model", "CV design", *(column[0] for column in table_columns)]
    with (output_dir / "comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _quality_figure(aggregate: dict[str, Any], *, score_split: str = "test"):
    import plotly.express as px

    metric_columns = (
        ("top1", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("macro_f1", "Macro F1"),
        ("auc_ovr_macro", "Macro OVR AUC"),
    )
    rows = []
    for model_name, metrics in sorted(aggregate.items()):
        for key, label in metric_columns:
            value = metrics.get(key)
            if value is None:
                continue
            rows.append(
                {
                    "model": _label(model_name),
                    "metric": label,
                    "mean_percent": float(value["mean"]) * 100.0,
                    "sample_std_percent": (
                        float(value["sample_std"]) * 100.0
                        if value.get("sample_std") is not None
                        else 0.0
                    ),
                    "seeds": int(value["n"]),
                }
            )
    seed_counts = _seed_counts(aggregate)
    if seed_counts == {1}:
        uncertainty_title = "single seed; no uncertainty estimate"
    elif len(seed_counts) == 1 and seed_counts:
        fold_counts = {
            int(metric.get("folds", 1))
            for metrics in aggregate.values()
            if isinstance(metric := metrics.get("top1"), dict)
        }
        fold_count = next(iter(fold_counts)) if len(fold_counts) == 1 else 1
        uncertainty_title = f"mean ± sample SD across {next(iter(seed_counts))} seeds"
        if fold_count > 1:
            uncertainty_title += f" (each averaged over {fold_count} folds)"
    else:
        uncertainty_title = "mean and sample SD across seeds"
    figure = px.bar(
        _columnar(rows),
        x="mean_percent",
        y="model",
        color="metric",
        barmode="group",
        error_x="sample_std_percent",
        orientation="h",
        hover_data={
            "mean_percent": ":.2f",
            "sample_std_percent": ":.2f",
            "seeds": True,
        },
        labels={
            "model": "Model",
            "mean_percent": "Score (%)",
            "metric": "Metric",
            "sample_std_percent": "Sample SD (%)",
            "seeds": "Seeds",
        },
        title=f"CIFAR-100 {score_split} quality metrics ({uncertainty_title})",
        template="plotly_white",
    )
    figure.update_layout(
        xaxis_range=[0, 100],
        yaxis_categoryorder="array",
        yaxis_categoryarray=[_label(name) for name in sorted(aggregate)],
        legend_title_text="Metric",
        legend_orientation="h",
        legend_y=-0.2,
        legend_x=0,
        hovermode="y unified",
        margin={"l": 220, "r": 35, "t": 80, "b": 150},
        height=max(650, 48 * len(aggregate) + 220),
    )
    return figure


def _fold_score_figure(report: dict[str, Any], *, score_split: str):
    """Show individual fold-by-seed scores without treating folds as independent seeds."""
    import plotly.express as px

    rows = []
    score_key = "validation" if score_split == "validation" else "test"
    for run in report.get("runs", []):
        folds = int(run.get("folds", 1))
        if folds <= 1:
            continue
        metrics = run.get(score_key) or {}
        score = metrics.get("top1")
        if score is None and score_split == "validation":
            score = run.get("best_validation_top1")
        if score is None:
            continue
        rows.append(
            {
                "model": _label(str(run["model"])),
                "top1_percent": float(score) * 100.0,
                "fold": f"Fold {int(run.get('fold_index', 1))}",
                "seed": int(run["seed"]),
                "fold_index": int(run.get("fold_index", 1)),
            }
        )
    if not rows:
        return None
    figure = px.strip(
        _columnar(rows),
        x="top1_percent",
        y="model",
        color="fold",
        stripmode="overlay",
        jitter=0.35,
        hover_data={"seed": True, "fold_index": True, "top1_percent": ":.2f"},
        labels={
            "top1_percent": f"{score_split.title()} top-1 accuracy (%)",
            "model": "Model",
            "fold": "Fold",
            "seed": "Seed",
            "fold_index": "Fold index",
        },
        title=f"Per-fold {score_split} scores across seeds",
        template="plotly_white",
        category_orders={"model": sorted({row["model"] for row in rows})},
    )
    figure.update_layout(
        height=max(650, 34 * len({row["model"] for row in rows}) + 220),
        width=1200,
        margin={"l": 210, "r": 35, "t": 85, "b": 90},
        legend_title_text="Cross-validation fold",
    )
    return figure


def _calibration_figure(aggregate: dict[str, Any]):
    import plotly.express as px

    rows = []
    for name, metrics in sorted(aggregate.items()):
        ece = metrics.get("ece_15")
        nll = metrics.get("nll")
        accuracy = metrics.get("top1")
        if ece is None or nll is None or accuracy is None:
            continue
        rows.append(
            {
                "model": _label(name),
                "ece_percent": float(ece["mean"]) * 100,
                "nll": float(nll["mean"]),
                "top1_percent": float(accuracy["mean"]) * 100,
                "seeds": int(accuracy["n"]),
            }
        )
    if not rows:
        return None
    figure = px.scatter(
        _columnar(rows),
        x="ece_percent",
        y="nll",
        color="model",
        symbol="model",
        size="top1_percent",
        hover_name="model",
        hover_data={"ece_percent": ":.2f", "nll": ":.3f", "top1_percent": ":.2f", "seeds": True},
        labels={
            "ece_percent": "Expected calibration error (%) · lower is better",
            "nll": "Negative log-likelihood · lower is better",
            "top1_percent": "Top-1 accuracy (%)",
            "model": "Model",
        },
        title="CIFAR-100 calibration and predictive quality",
        template="plotly_white",
    )
    figure.update_layout(
        legend_orientation="h",
        legend_y=-0.25,
        margin={"l": 85, "r": 30, "t": 80, "b": 150},
        height=600,
    )
    return figure


def _taxonomy_figures(aggregate: dict[str, Any]):
    import plotly.express as px

    rows = []
    consistency_rows = []
    for name, metrics in sorted(aggregate.items()):
        for key, label in (
            ("coarse_top1", "Coarse top-1"),
            ("taxonomy_agreement", "Fine/coarse agreement"),
        ):
            value = metrics.get(key)
            if value:
                rows.append(
                    {
                        "model": _label(name),
                        "metric": label,
                        "mean_percent": float(value["mean"]) * 100,
                        "sample_std_percent": (
                            float(value["sample_std"]) * 100
                            if value.get("sample_std") is not None
                            else 0.0
                        ),
                        "seeds": int(value["n"]),
                    }
                )
        consistency = metrics.get("hierarchy_consistency_mae")
        accuracy = metrics.get("top1")
        if consistency and accuracy:
            consistency_rows.append(
                {
                    "model": _label(name),
                    "top1_percent": float(accuracy["mean"]) * 100,
                    "hierarchy_consistency_mae": float(consistency["mean"]),
                    "seeds": int(accuracy["n"]),
                }
            )
    figures = {}
    if rows:
        figure = px.bar(
            _columnar(rows),
            x="mean_percent",
            y="model",
            color="metric",
            barmode="group",
            error_x="sample_std_percent",
            orientation="h",
            hover_data={"mean_percent": ":.2f", "sample_std_percent": ":.2f", "seeds": True},
            labels={
                "mean_percent": "Score (%)",
                "model": "Model",
                "metric": "Hierarchy metric",
                "sample_std_percent": "Sample SD (pp)",
            },
            title="CIFAR-100 coarse classification and hierarchy agreement",
            template="plotly_white",
        )
        figure.update_layout(
            legend_orientation="h",
            legend_y=-0.2,
            margin={"l": 220, "r": 35, "t": 80, "b": 145},
            height=max(500, 46 * len({row["model"] for row in rows}) + 200),
        )
        figures["taxonomy_metrics"] = figure
    if consistency_rows:
        figure = px.scatter(
            _columnar(consistency_rows),
            x="top1_percent",
            y="hierarchy_consistency_mae",
            color="model",
            symbol="model",
            hover_name="model",
            hover_data={"top1_percent": ":.2f", "hierarchy_consistency_mae": ":.4f", "seeds": True},
            labels={
                "top1_percent": "Fine-class top-1 accuracy (%)",
                "hierarchy_consistency_mae": "Hierarchy consistency MAE · lower is better",
                "model": "Model",
            },
            title="Fine-class accuracy and hierarchy consistency",
            template="plotly_white",
        )
        figure.update_layout(
            legend_orientation="h",
            legend_y=-0.25,
            margin={"l": 85, "r": 30, "t": 80, "b": 150},
            height=580,
        )
        figures["hierarchy_consistency"] = figure
    return figures


def _efficiency_figures(aggregate: dict[str, Any], batch_size: int):
    import plotly.express as px

    output = {}
    for key, axis_label, conversion, log_x, suffix in EFFICIENCY_PLOTS:
        if suffix == "fps":
            axis_label = f"Model-only FPS (batch={batch_size})"
        rows = []
        for model_name, metrics in sorted(aggregate.items()):
            accuracy = metrics.get("top1")
            cost = metrics.get(key)
            if accuracy is None or cost is None:
                continue
            rows.append(
                {
                    "model": _label(model_name),
                    "accuracy_percent": float(accuracy["mean"]) * 100.0,
                    "accuracy_sample_std_percent": (
                        float(accuracy["sample_std"]) * 100.0
                        if accuracy.get("sample_std") is not None
                        else 0.0
                    ),
                    "cost_mean": _convert(float(cost["mean"]), conversion),
                    "cost_sample_std": (
                        _convert(float(cost["sample_std"]), conversion)
                        if cost.get("sample_std") is not None
                        else 0.0
                    ),
                    "seeds": int(accuracy["n"]),
                }
            )
        if not rows:
            continue
        figure = px.scatter(
            _columnar(rows),
            x="cost_mean",
            y="accuracy_percent",
            color="model",
            symbol="model",
            error_x="cost_sample_std",
            error_y="accuracy_sample_std_percent",
            hover_name="model",
            hover_data={
                "cost_mean": ":.4f",
                "cost_sample_std": ":.4f",
                "accuracy_percent": ":.2f",
                "accuracy_sample_std_percent": ":.2f",
                "seeds": True,
            },
            labels={
                "cost_mean": axis_label,
                "accuracy_percent": "Top-1 accuracy (%)",
                "cost_sample_std": "Sample SD",
                "accuracy_sample_std_percent": "Top-1 sample SD (pp)",
                "model": "Model",
                "seeds": "Seeds",
            },
            title=f"Top-1 accuracy vs {axis_label.lower()}",
            template="plotly_white",
        )
        if log_x:
            tick_sets = {
                "parameters": [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50],
                "flops": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
                "size": [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
                "memory": [25, 40, 60, 80, 100, 150, 200, 300, 400],
                "train_memory": [100, 200, 400, 800, 1600, 3200, 6400, 12800],
            }
            ticks = tick_sets[suffix]
            tick_labels = [f"{value:g}" for value in ticks]
            figure.update_xaxes(
                type="log", tickmode="array", tickvals=ticks, ticktext=tick_labels
            )
        figure.update_traces(
            marker={"size": 11, "opacity": 0.78, "line": {"color": "white", "width": 1.2}}
        )
        figure.update_layout(
            legend_title_text="Model",
            legend_orientation="h",
            legend_y=-0.28,
            legend_x=0,
            hovermode="closest",
            margin={"l": 80, "r": 30, "t": 80, "b": 145},
            height=560,
        )
        output[suffix] = figure
    return output


def _write_plotly_files(
    output_dir: Path,
    quality_figure,
    efficiency_figures: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, str]:
    import plotly.io as pio
    try:
        import kaleido  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNG export requires Kaleido. Install it with `pip install -e \".[reports]\"`."
        ) from exc

    figure_paths: dict[str, str] = {}
    figures = {"quality_metrics": quality_figure, **efficiency_figures}
    if int(report.get("protocol", {}).get("folds", 1)) > 1:
        fold_score_figure = _fold_score_figure(
            report,
            score_split="validation"
            if report.get("protocol", {}).get("validation_only", False)
            else "test",
        )
        if fold_score_figure is not None:
            figures["fold_scores"] = fold_score_figure
    report_aggregate = report.get("aggregate_mean_and_sample_std") or report.get(
        "validation_aggregate_mean_and_sample_std", {}
    )
    calibration = _calibration_figure(report_aggregate)
    if calibration is not None:
        figures["calibration"] = calibration
    figures.update(_taxonomy_figures(report_aggregate))
    for name, figure in figures.items():
        path = output_dir / f"{name}.html"
        figure.write_html(path, include_plotlyjs="cdn", full_html=True)
        figure_paths[name] = str(path)
        png_path = output_dir / f"{name}.png"
        figure.write_image(png_path, format="png", scale=2)
        figure_paths[f"{name}_png"] = str(png_path)

    protocol = report.get("protocol", {})
    validation_only = bool(protocol.get("validation_only", False))
    split_notice = (
        "This is a validation-only report for development. It must not be described as a test-set result."
        if validation_only
        else "The official test data were excluded from training and validation but were inspected in earlier "
        "project runs; this is not a pristine confirmatory holdout."
    )
    cv_chart_note = (
        "Fold-score charts show each fold/seed run; error bars in aggregate charts use seed-level summaries. Folds are not counted as extra independent seeds.\n\n"
        if int(protocol.get("folds", 1)) > 1
        else ""
    )
    notes = (
        "Charts use Plotly Express. Hover a point or bar for its values and seed count; the "
        "legend can hide or show model/metric series. Points have no printed labels, keeping "
        f"close results readable. {_uncertainty_notice(report_aggregate)}\n\n"
        f"{split_notice}\n\n"
        "Each figure is available as an interactive HTML file and as a high-resolution PNG.\n\n"
        f"Validation design: {protocol.get('folds', 1)} folds; {protocol.get('validation_aggregation', 'historical per-seed aggregation')}\n\n"
        f"{cv_chart_note}"
        f"Evaluation split: {protocol.get('test', 'not specified')}\n\n"
        "Precision, recall, F1, and AUC are macro-averaged by class; AUC is macro one-vs-rest. "
        "FLOPs are a documented analytical estimate. Checkpoint size is decimal MB. Batch-1 "
        "latency excludes data loading and host-to-device transfer; batched FPS is model-only. "
        "Training peak memory is measured during optimization; peak allocated inference memory "
        "is measured during evaluation.\n"
    )
    (output_dir / "report_notes.md").write_text(notes, encoding="utf-8")

    fragments = []
    for index, (name, figure) in enumerate(figures.items()):
        fragments.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=True if index == 0 else False,
                config={"responsive": True, "displaylogo": False},
                div_id=f"plot-{name}",
            )
        )
    dashboard = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeSy-CVT benchmark dashboard</title>
<style>
body { font-family: Arial, sans-serif; margin: 0 auto; max-width: 1500px; padding: 20px; color: #17202a; }
h1 { margin-bottom: 4px; } .note { color: #4b5563; line-height: 1.5; }
section { margin: 26px 0 40px; } .plot { width: 100%; }
</style>
</head>
<body>
<h1>CIFAR-100 model comparison</h1>
<p class="note">Interactive Plotly Express figures. Hover for values; use the legend to filter models or metrics. """ + _uncertainty_notice(report_aggregate) + """</p>
<p class="note">""" + split_notice + """</p>
""" + "\n".join(
        f'<section class="plot">{fragment}</section>' for fragment in fragments
    ) + "\n</body>\n</html>\n"
    dashboard_path = output_dir / "benchmark_dashboard.html"
    dashboard_path.write_text(dashboard, encoding="utf-8")
    figure_paths["dashboard"] = str(dashboard_path)
    figure_paths["notes"] = str(output_dir / "report_notes.md")
    return figure_paths


def create_benchmark_report(input_path: str, output_dir: str) -> dict[str, str]:
    """Write comparative tables plus interactive Plotly Express HTML and static PNG charts."""
    try:
        import plotly.express  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Chart creation requires Plotly. Install it with `pip install -e .[reports]`."
        ) from exc

    report_file = Path(input_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    aggregate = _aggregate(report)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_only = bool(report.get("protocol", {}).get("validation_only", False))
    score_split = "validation" if validation_only else "test"
    _write_tables(output, aggregate, score_split=score_split, protocol=report.get("protocol", {}))
    batch_size = int(report.get("protocol", {}).get("batch_size", 16))
    quality_figure = _quality_figure(aggregate, score_split=score_split)
    efficiency_figures = _efficiency_figures(aggregate, batch_size)
    outputs = {
        "table_markdown": str(output / "comparison_table.md"),
        "table_csv": str(output / "comparison_table.csv"),
    }
    outputs.update(_write_plotly_files(output, quality_figure, efficiency_figures, report))
    return outputs


def create_before_after_report(
    before_path: str, after_path: str, output_dir: str
) -> dict[str, str]:
    """Create a same-test-set comparison of two versioned benchmark reports."""
    import plotly.express as px

    before_report = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after_report = json.loads(Path(after_path).read_text(encoding="utf-8"))
    before_protocol = before_report.get("protocol", {})
    after_protocol = after_report.get("protocol", {})
    before_hash = before_protocol.get("test_data_sha256")
    after_hash = after_protocol.get("test_data_sha256")
    if not before_hash or not after_hash or before_hash != after_hash:
        raise ValueError(
            "Before/after reports must include the same non-empty test_data_sha256."
        )
    if before_protocol.get("test_examples") != after_protocol.get("test_examples"):
        raise ValueError("Before/after reports must use the same number of test examples.")

    before = _aggregate(before_report)
    after = _aggregate(after_report)
    models = sorted(set(before) & set(after))
    if not models:
        raise ValueError("Before/after reports have no model names in common.")

    metrics = (
        ("top1", "Top-1 accuracy", "percent", 100.0),
        ("top5", "Top-5 accuracy", "percent", 100.0),
        ("precision_macro", "Macro precision", "percent", 100.0),
        ("recall_macro", "Macro recall", "percent", 100.0),
        ("macro_f1", "Macro F1", "percent", 100.0),
        ("auc_ovr_macro", "Macro OVR AUC", "percent", 100.0),
        ("ece_15", "ECE (15-bin)", "identity", 1.0),
        ("nll", "NLL", "identity", 1.0),
        ("coarse_top1", "Coarse top-1", "percent", 100.0),
        ("taxonomy_agreement", "Fine/coarse agreement", "percent", 100.0),
        ("hierarchy_consistency_mae", "Hierarchy consistency MAE", "identity", 1.0),
        ("num_parameters", "Parameters (M)", "millions", 1e-6),
        ("trainable_parameters", "Trainable parameters (M)", "millions", 1e-6),
        ("flops_g_per_image", "Estimated FLOPs (G/image)", "identity", 1.0),
        ("model_size_mb", "Checkpoint size (MB)", "identity", 1.0),
        ("single_image_latency_p50_ms", "Batch-1 latency p50 (ms)", "identity", 1.0),
        ("single_image_latency_p95_ms", "Batch-1 latency p95 (ms)", "identity", 1.0),
        ("inference_fps", "Batched model-only FPS", "identity", 1.0),
        ("pipeline_images_per_second", "Evaluation pipeline images/s", "identity", 1.0),
        ("peak_gpu_memory_mb", "Peak allocated GPU memory (MB)", "identity", 1.0),
        ("train_peak_gpu_memory_mb", "Training peak allocated GPU memory (MB)", "identity", 1.0),
        ("training_seconds_including_validation", "Training time (hours)", "hours", 1 / 3600),
    )
    long_rows = []
    delta_rows = []
    summary_rows = []
    for model in models:
        for key, label, unit, scale in metrics:
            old = before[model].get(key)
            new = after[model].get(key)
            if not old or not new:
                continue
            old_mean = float(old["mean"]) * scale
            new_mean = float(new["mean"]) * scale
            old_sd = float(old["sample_std"] or 0.0) * scale
            new_sd = float(new["sample_std"] or 0.0) * scale
            common = {
                "model": _label(model),
                "metric": label,
                "before_mean": old_mean,
                "before_sd": old_sd,
                "after_mean": new_mean,
                "after_sd": new_sd,
                "delta_after_minus_before": new_mean - old_mean,
                "before_seeds": int(old["n"]),
                "after_seeds": int(new["n"]),
                "unit": unit,
            }
            summary_rows.append(common)
            if key in {"top1", "precision_macro", "recall_macro", "macro_f1", "auc_ovr_macro"}:
                long_rows.extend(
                    [
                        {
                            "model": _label(model),
                            "metric": label,
                            "version": "Before · legacy recipe",
                            "mean": old_mean,
                            "sample_sd": old_sd,
                            "seeds": int(old["n"]),
                        },
                        {
                            "model": _label(model),
                            "metric": label,
                            "version": "After · new recipe",
                            "mean": new_mean,
                            "sample_sd": new_sd,
                            "seeds": int(new["n"]),
                        },
                    ]
                )
            if key == "top1":
                delta_rows.append(
                    {
                        "model": _label(model),
                        "delta_pp": (new_mean - old_mean),
                        "before_top1": old_mean,
                        "after_top1": new_mean,
                        "before_sd": old_sd,
                        "after_sd": new_sd,
                        "before_seeds": int(old["n"]),
                        "after_seeds": int(new["n"]),
                    }
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "before_after_metrics.csv"
    fieldnames = list(summary_rows[0]) if summary_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    figure_data = _columnar(long_rows)
    quality_figure = px.bar(
        figure_data,
        x="model",
        y="mean",
        color="version",
        facet_col="metric",
        facet_col_wrap=2,
        barmode="group",
        error_y="sample_sd",
        hover_data={"mean": ":.2f", "sample_sd": ":.2f", "seeds": True},
        labels={"model": "Model", "mean": "Score (%)", "version": "Recipe"},
        title="Exploratory CIFAR-100 results before and after the protocol update",
        template="plotly_white",
    )
    quality_figure.update_layout(
        margin={"l": 80, "r": 35, "t": 100, "b": 160},
        height=max(900, 35 * len(models) + 800),
        legend_orientation="h",
        legend_y=-0.18,
    )
    quality_figure.update_xaxes(tickangle=-35)

    delta_figure = px.bar(
        _columnar(delta_rows),
        x="model",
        y="delta_pp",
        color="delta_pp",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        hover_data={
            "delta_pp": ":+.2f",
            "before_top1": ":.2f",
            "after_top1": ":.2f",
            "before_seeds": True,
            "after_seeds": True,
        },
        labels={"model": "Model", "delta_pp": "Top-1 change (percentage points)"},
        title="Change in top-1 after the recipe update (descriptive only)",
        template="plotly_white",
    )
    delta_figure.update_layout(
        margin={"l": 80, "r": 35, "t": 80, "b": 150},
        height=600,
        xaxis_tickangle=-35,
        coloraxis_showscale=False,
    )

    figures = {"before_after_quality": quality_figure, "top1_change": delta_figure}
    import plotly.io as pio

    figure_paths: dict[str, str] = {"csv": str(csv_path)}
    for name, figure in figures.items():
        html_path = output / f"{name}.html"
        png_path = output / f"{name}.png"
        figure.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        figure.write_image(png_path, format="png", scale=2)
        figure_paths[name] = str(html_path)
        figure_paths[f"{name}_png"] = str(png_path)
    dashboard = "\n".join(
        pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs=True if index == 0 else False,
            config={"responsive": True, "displaylogo": False},
            div_id=f"before-after-{name}",
        )
        for index, (name, figure) in enumerate(figures.items())
    )
    dashboard_path = output / "before_after_dashboard.html"
    dashboard_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Before/after CIFAR-100</title></head>"
        "<body><h1>Before/after comparison</h1><p>Same official test-set fingerprint; existing test-set "
        "prior exposure means these results remain exploratory. Recipe and schedule differences mean "
        "the plot is descriptive and does not isolate a single change.</p>"
        + dashboard
        + "</body></html>\n",
        encoding="utf-8",
    )
    figure_paths["dashboard"] = str(dashboard_path)
    notes_path = output / "README.md"
    notes_path.write_text(
        "# Before/After Benchmark Report\n\n"
        "This report compares two result files only when their official test-set fingerprints "
        "match. It preserves all seed-level means and sample standard deviations. It is descriptive: "
        "the CIFAR-100 test set was previously inspected, and differences in training recipe mean "
        "the plot does not identify one causal component. Do not label it confirmatory SOTA evidence.\n",
        encoding="utf-8",
    )
    figure_paths["notes"] = str(notes_path)
    return figure_paths


def create_training_curve_report(
    before_runs_dir: str, after_runs_dir: str, output_dir: str
) -> dict[str, str]:
    """Plot paired validation learning curves while preserving each study as a separate facet."""
    import plotly.express as px

    studies = (("Before · legacy recipe", Path(before_runs_dir)), ("After · CIFAR v2", Path(after_runs_dir)))
    records = []
    for study, root in studies:
        if not root.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {root}")
        for progress_path in root.rglob("progress.json"):
            try:
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
                model = str(payload["model"])
                seed = int(payload["seed"])
                fold_index = int(payload.get("fold_index", 1))
                folds = int(payload.get("folds", 1))
                history = payload["epochs"]
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            for row in history:
                records.append(
                    {
                        "study": study,
                        "model": _label(model),
                        "seed": seed,
                        "fold_index": fold_index,
                        "folds": folds,
                        "epoch": int(row["epoch"]),
                        "val_top1_percent": float(row["val_top1"]) * 100,
                        "train_loss": float(row["train_loss"]),
                    }
                )
    if not records:
        raise ValueError("No complete progress.json histories were found in the two run folders.")

    per_seed_fold: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["study"], record["model"], record["seed"], record["epoch"])
        per_seed_fold.setdefault(key, []).append(record)
    per_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for (study, model, seed, epoch), fold_rows in per_seed_fold.items():
        per_seed.setdefault((study, model, epoch), []).append(
            {
                "seed": seed,
                "val_top1": statistics.mean(row["val_top1_percent"] for row in fold_rows),
                "train_loss": statistics.mean(row["train_loss"] for row in fold_rows),
                "folds": len(fold_rows),
            }
        )
    aggregate_rows = []
    for (study, model, epoch), rows in per_seed.items():
        top1s = [row["val_top1"] for row in rows]
        losses = [row["train_loss"] for row in rows]
        aggregate_rows.append(
            {
                "study": study,
                "model": model,
                "epoch": epoch,
                "val_top1_mean": statistics.mean(top1s),
                "val_top1_sample_sd": statistics.stdev(top1s) if len(top1s) > 1 else 0.0,
                "train_loss_mean": statistics.mean(losses),
                "train_loss_sample_sd": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "seeds": len(rows),
                "folds_per_seed": max(row["folds"] for row in rows),
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "training_curve_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(aggregate_rows, key=lambda row: (row["study"], row["model"], row["epoch"])))

    frame = _columnar(aggregate_rows)
    accuracy_figure = px.line(
        frame,
        x="epoch",
        y="val_top1_mean",
        color="model",
        facet_col="study",
        facet_col_spacing=0.05,
        error_y="val_top1_sample_sd",
        markers=True,
        hover_data={"seeds": True, "val_top1_mean": ":.2f", "val_top1_sample_sd": ":.2f"},
        labels={
            "epoch": "Epoch",
            "val_top1_mean": "Validation top-1 (%)",
            "val_top1_sample_sd": "Sample SD (percentage points)",
            "model": "Model",
            "study": "Study",
        },
        title="Before/after validation learning curves",
        template="plotly_white",
    )
    accuracy_figure.update_layout(
        legend_orientation="h",
        legend_y=-0.24,
        margin={"l": 80, "r": 30, "t": 90, "b": 150},
        height=650,
    )
    loss_figure = px.line(
        frame,
        x="epoch",
        y="train_loss_mean",
        color="model",
        facet_col="study",
        facet_col_spacing=0.05,
        error_y="train_loss_sample_sd",
        markers=True,
        hover_data={"seeds": True, "train_loss_mean": ":.3f", "train_loss_sample_sd": ":.3f"},
        labels={
            "epoch": "Epoch",
            "train_loss_mean": "Training loss",
            "train_loss_sample_sd": "Sample SD",
            "model": "Model",
            "study": "Study",
        },
        title="Before/after training-loss curves",
        template="plotly_white",
    )
    loss_figure.update_layout(
        legend_orientation="h",
        legend_y=-0.24,
        margin={"l": 80, "r": 30, "t": 90, "b": 150},
        height=650,
    )

    import plotly.io as pio

    paths = {"summary_csv": str(csv_path)}
    figures = {
        "validation_top1_before_after": accuracy_figure,
        "training_loss_before_after": loss_figure,
    }
    dashboard_fragments = []
    for index, (name, figure) in enumerate(figures.items()):
        html_path = output / f"{name}.html"
        png_path = output / f"{name}.png"
        figure.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        figure.write_image(png_path, format="png", scale=2)
        paths[name] = str(html_path)
        paths[f"{name}_png"] = str(png_path)
        dashboard_fragments.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=True if index == 0 else False,
                config={"responsive": True, "displaylogo": False},
                div_id=f"training-curves-{name}",
            )
        )
    dashboard_path = output / "training_curves_dashboard.html"
    dashboard_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>NeSy-CVT learning curves</title></head>"
        "<body><h1>Before/after training curves</h1><p>The facets show each protocol separately. "
        "A short post-change preflight is a pipeline diagnostic, not a final performance comparison.</p>"
        + "\n".join(dashboard_fragments)
        + "</body></html>\n",
        encoding="utf-8",
    )
    paths["dashboard"] = str(dashboard_path)
    return paths
