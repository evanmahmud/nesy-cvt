from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


TABLE_COLUMNS = (
    ("Accuracy (%)", "top1", "percent"),
    ("Top-5 (%)", "top5", "percent"),
    ("Precision macro (%)", "precision_macro", "percent"),
    ("Recall macro (%)", "recall_macro", "percent"),
    ("F1 macro (%)", "macro_f1", "percent"),
    ("AUC OVR macro (%)", "auc_ovr_macro", "percent"),
    ("Parameters (M)", "num_parameters", "millions"),
    ("Estimated FLOPs (G/image)", "flops_g_per_image", "gflops"),
    ("Checkpoint size (MB)", "model_size_mb", "identity"),
    ("Batch-1 latency p50 (ms)", "single_image_latency_p50_ms", "identity"),
    ("FPS (model-only, batched)", "inference_fps", "identity"),
    ("Peak GPU memory (MB)", "peak_gpu_memory_mb", "identity"),
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
)


def _aggregate(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate_mean_and_sample_std")
    if not isinstance(aggregate, dict) or not aggregate:
        raise ValueError("The report has no aggregate_mean_and_sample_std section")
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
        "resnet18_cifar": "ResNet-18",
        "mobilenet_v3_small": "MobileNetV3-Small",
        "densenet121_cifar": "DenseNet-121",
        "convnext_tiny_cifar": "ConvNeXt-Tiny",
        "swin_t_cifar": "Swin-T",
        "vit_tiny_patch4": "ViT-Tiny/4",
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


def _markdown_table(aggregate: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    headers = ["Model", "Seeds", *(column[0] for column in TABLE_COLUMNS)]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    csv_rows: list[dict[str, Any]] = []
    for model_name, metrics in sorted(aggregate.items()):
        top1 = metrics.get("top1", {})
        seed_count = int(top1.get("n", 0))
        row = {"Model": _label(model_name), "Seeds": seed_count}
        values = []
        for title, key, conversion in TABLE_COLUMNS:
            digits = 3 if conversion == "gflops" else 2
            cell = _summary_cell(metrics.get(key), conversion, digits)
            row[title] = cell
            values.append(cell)
        lines.append("| " + " | ".join([row["Model"], str(seed_count), *values]) + " |")
        csv_rows.append(row)
    return "\n".join(lines) + "\n", csv_rows


def _write_tables(output_dir: Path, aggregate: dict[str, Any]) -> None:
    markdown, rows = _markdown_table(aggregate)
    (output_dir / "comparison_table.md").write_text(markdown, encoding="utf-8")
    columns = ["Model", "Seeds", *(column[0] for column in TABLE_COLUMNS)]
    with (output_dir / "comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _quality_figure(aggregate: dict[str, Any]):
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
        title="CIFAR-100 quality metrics (mean ± sample SD across seeds)",
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
    for name, figure in figures.items():
        path = output_dir / f"{name}.html"
        figure.write_html(path, include_plotlyjs="cdn", full_html=True)
        figure_paths[name] = str(path)
        png_path = output_dir / f"{name}.png"
        figure.write_image(png_path, format="png", scale=2)
        figure_paths[f"{name}_png"] = str(png_path)

    protocol = report.get("protocol", {})
    notes = (
        "Charts use Plotly Express. Hover a point or bar for its values and seed count; the "
        "legend can hide or show model/metric series. Points have no printed labels, keeping "
        "close results readable. Error bars show sample standard deviation across seeds.\n\n"
        "The test data were excluded from training and validation but were inspected in earlier "
        "project runs; this is not a pristine confirmatory holdout.\n\n"
        "Each figure is available as an interactive HTML file and as a high-resolution PNG.\n\n"
        f"Test set: {protocol.get('test', 'not specified')}\n\n"
        "Precision, recall, F1, and AUC are macro-averaged by class; AUC is macro one-vs-rest. "
        "FLOPs are a documented analytical estimate. Checkpoint size is decimal MB. Batch-1 "
        "latency excludes data loading and host-to-device transfer; batched FPS is model-only.\n"
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
<p class="note">Interactive Plotly Express figures. Hover for values; use the legend to filter models or metrics. Error bars show sample standard deviation across seeds.</p>
<p class="note">The official test split was excluded from training and validation but was inspected in earlier project runs, so it is not a pristine confirmatory holdout.</p>
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
    _write_tables(output, aggregate)
    batch_size = int(report.get("protocol", {}).get("batch_size", 16))
    quality_figure = _quality_figure(aggregate)
    efficiency_figures = _efficiency_figures(aggregate, batch_size)
    outputs = {
        "table_markdown": str(output / "comparison_table.md"),
        "table_csv": str(output / "comparison_table.csv"),
    }
    outputs.update(_write_plotly_files(output, quality_figure, efficiency_figures, report))
    return outputs
