from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    dataset: str
    top1: float | None
    top5: float | None
    params_m: float | None
    flops_g: float | None
    latency_ms: float | None
    source: str
    reference_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_benchmark_tables(
    controlled: list[BenchmarkResult],
    literature_sota: list[BenchmarkResult],
) -> dict[str, list[dict[str, Any]]]:
    controlled_rows = [row.to_dict() for row in controlled]
    literature_rows = []
    for row in literature_sota:
        data = row.to_dict()
        data["not_rerun_warning"] = (
            "Reported literature result only. It was not rerun with this repository, "
            "hardware, data loader, or training recipe."
        )
        literature_rows.append(data)
    return {"controlled": controlled_rows, "literature_sota": literature_rows}


def save_metrics_json(
    path: str | Path,
    metrics: dict[str, Any],
    *,
    seed: int,
    hardware: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "reproducibility": {
            "seed": seed,
            "hardware": hardware,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    if extra:
        payload["extra"] = extra
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


