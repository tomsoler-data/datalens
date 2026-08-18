from __future__ import annotations

import json
from pathlib import Path

from app.evals.schemas import (
    AnalyticalEvalCase,
    EvalSplit,
)


def load_benchmark(
    path: str | Path,
    *,
    split: EvalSplit | None = None,
) -> list[AnalyticalEvalCase]:
    """
    Charge un benchmark DataLens au format JSONL.

    Chaque ligne non vide doit contenir exactement
    un AnalyticalEvalCase valide.
    """

    benchmark_path = Path(
        path,
    )

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark introuvable : {benchmark_path}"
        )

    cases: list[
        AnalyticalEvalCase
    ] = []

    seen_case_ids: set[
        str
    ] = set()

    with benchmark_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                payload = json.loads(
                    line,
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "JSON invalide dans "
                    f"{benchmark_path} "
                    f"à la ligne {line_number}: "
                    f"{exc.msg}"
                ) from exc

            try:
                case = (
                    AnalyticalEvalCase
                    .model_validate(
                        payload,
                    )
                )

            except Exception as exc:
                raise ValueError(
                    "Cas de benchmark invalide dans "
                    f"{benchmark_path} "
                    f"à la ligne {line_number}: "
                    f"{exc}"
                ) from exc

            if (
                case.case_id
                in seen_case_ids
            ):
                raise ValueError(
                    "case_id dupliqué : "
                    f"{case.case_id}"
                )

            seen_case_ids.add(
                case.case_id,
            )

            if (
                split is None
                or case.split == split
            ):
                cases.append(
                    case,
                )

    return cases