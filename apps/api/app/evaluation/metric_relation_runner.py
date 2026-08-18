from __future__ import annotations

import hashlib
import json

from collections import Counter
from pathlib import Path

from app.ai.provider import DEFAULT_MODEL

from app.evaluation.benchmarks.clinical_lab import (
    CLINICAL_LAB_DATASET_ID,
    CLINICAL_LAB_FILENAME,
    build_clinical_lab_benchmark_dataframe,
)

from app.evaluation.benchmarks.customer_support import (
    CUSTOMER_SUPPORT_DATASET_ID,
    CUSTOMER_SUPPORT_FILENAME,
    build_customer_support_benchmark_dataframe,
)

from app.semantics.family import build_quantity_family_reports
from app.semantics.pipeline import prepare_datasets_semantics

from app.semantics.relation_evidence import (
    build_metric_relation_evidence,
)

from app.semantics.relation_evidence_schemas import (
    METRIC_RELATION_EVIDENCE_VERSION,
)


GROUND_TRUTH_PATH = Path(
    "artifacts/evaluation/development/"
    "metric_relation_taxonomy_development_v0.1.json"
)

POSITIVE_CLASS = "same_metric_different_state"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def output_path_for_version() -> Path:
    version = METRIC_RELATION_EVIDENCE_VERSION.removeprefix(
        "metric_relation_evidence_"
    )

    return Path(
        "artifacts/evaluation/experiments/"
        f"metric_relation_evidence_development_{version}.json"
    )


def load_ground_truth() -> dict:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(GROUND_TRUTH_PATH)

    return json.loads(
        GROUND_TRUTH_PATH.read_text(
            encoding="utf-8-sig",
        )
    )


def build_datasets() -> list[dict]:
    return [
        {
            "dataset_id": CLINICAL_LAB_DATASET_ID,
            "filename": CLINICAL_LAB_FILENAME,
            "dataframe": build_clinical_lab_benchmark_dataframe(),
        },
        {
            "dataset_id": CUSTOMER_SUPPORT_DATASET_ID,
            "filename": CUSTOMER_SUPPORT_FILENAME,
            "dataframe": build_customer_support_benchmark_dataframe(),
        },
    ]


def main() -> None:
    ground_truth = load_ground_truth()

    case_sets = {
        "clinical_lab": ground_truth["clinical_lab"],
        "customer_support": ground_truth["customer_support"],
    }

    domain_to_dataset_id = {
        "clinical_lab": CLINICAL_LAB_DATASET_ID,
        "customer_support": CUSTOMER_SUPPORT_DATASET_ID,
    }

    cases = [
        case
        for domain_cases in case_sets.values()
        for case in domain_cases
    ]

    positive_count = sum(
        case["relation"] == POSITIVE_CLASS
        for case in cases
    )

    print()
    print("=" * 90)
    print("METRIC RELATION DEVELOPMENT EVALUATION")
    print("=" * 90)
    print("Component:", METRIC_RELATION_EVIDENCE_VERSION)
    print("Model:", DEFAULT_MODEL)
    print("Cases:", len(cases))
    print("Positive cases:", positive_count)

    print()
    print("Preparing semantic profiles...")

    preparation = prepare_datasets_semantics(
        datasets=build_datasets(),
        model=DEFAULT_MODEL,
    )

    profiles = list(preparation.profiles)

    print("Building quantity-family reports...")

    family_reports = build_quantity_family_reports(
        profiles=profiles,
        model=DEFAULT_MODEL,
    )

    columns_by_dataset = {
        dataset.dataset_id: {
            column.column: column
            for column in dataset.columns
        }
        for dataset in profiles
    }

    families_by_dataset = {
        report.dataset_id: report
        for report in family_reports
    }

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    prediction_counts = Counter()
    results: list[dict] = []

    for domain, domain_cases in case_sets.items():
        dataset_id = domain_to_dataset_id[domain]
        columns = columns_by_dataset[dataset_id]
        family_report = families_by_dataset[dataset_id]

        for case in domain_cases:
            left = columns.get(case["left"])
            right = columns.get(case["right"])

            if left is None:
                raise KeyError(
                    f"Missing profile: {dataset_id} / {case['left']}"
                )

            if right is None:
                raise KeyError(
                    f"Missing profile: {dataset_id} / {case['right']}"
                )

            evidence = build_metric_relation_evidence(
                left=left,
                right=right,
                family_report=family_report,
                embedding_pair=None,
            )

            predicted = evidence.interpretation.proposed_relation
            expected = case["relation"]

            expected_positive = expected == POSITIVE_CLASS
            predicted_positive = predicted == POSITIVE_CLASS

            prediction_counts[predicted] += 1

            if expected_positive and predicted_positive:
                outcome = "TP"
                tp += 1

            elif not expected_positive and predicted_positive:
                outcome = "FP"
                fp += 1

            elif not expected_positive and not predicted_positive:
                outcome = "TN"
                tn += 1

            else:
                outcome = "FN"
                fn += 1

            results.append(
                {
                    "domain": domain,
                    "dataset_id": dataset_id,
                    "case_id": case["case_id"],
                    "left": case["left"],
                    "right": case["right"],
                    "expected_relation": expected,
                    "predicted_relation": predicted,
                    "confidence": evidence.interpretation.confidence,
                    "binary_outcome": outcome,
                    "family_source": evidence.family.relation_source,
                    "same_quantity_family": (
                        evidence.family.same_quantity_family
                    ),
                    "same_known_dimension": (
                        evidence.quantity.same_known_quantity_dimension
                    ),
                    "same_concept_family": (
                        evidence.comparator.same_concept_family
                    ),
                    "distinct_known_states": (
                        evidence.family.distinct_known_states
                    ),
                    "dimension_conflict": (
                        evidence.quantity.dimension_conflict
                    ),
                    "full_evidence": evidence.model_dump(mode="json"),
                }
            )

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    accuracy = safe_ratio(tp + tn, tp + fp + tn + fn)

    safety_gate = fp == 0
    capability_gate = tp > 0
    development_gate = safety_gate and capability_gate

    print()
    print("=" * 90)
    print("CONFUSION MATRIX")
    print("=" * 90)
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)

    print()
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)

    print()
    print("Prediction counts:", dict(prediction_counts))

    print()
    print("=" * 90)
    print("POSITIVE PREDICTIONS")
    print("=" * 90)

    positives = [
        result
        for result in results
        if result["predicted_relation"] == POSITIVE_CLASS
    ]

    if not positives:
        print("NONE")

    for result in positives:
        print(
            f"[{result['binary_outcome']}] "
            f"{result['left']} VS {result['right']}"
        )

    print()
    print("=" * 90)
    print("FALSE POSITIVES")
    print("=" * 90)

    false_positives = [
        result
        for result in results
        if result["binary_outcome"] == "FP"
    ]

    if not false_positives:
        print("NONE")

    for result in false_positives:
        print(
            result["case_id"],
            "|",
            result["left"],
            "VS",
            result["right"],
        )

    print()
    print("=" * 90)
    print("FALSE NEGATIVES")
    print("=" * 90)

    false_negatives = [
        result
        for result in results
        if result["binary_outcome"] == "FN"
    ]

    if not false_negatives:
        print("NONE")

    for result in false_negatives:
        print(
            result["case_id"],
            "|",
            result["left"],
            "VS",
            result["right"],
        )

    print()
    print("=" * 90)
    print("GATES")
    print("=" * 90)
    print("Safety Gate (FP == 0):", safety_gate)
    print("Capability Gate (TP > 0):", capability_gate)
    print("Development Gate:", development_gate)

    output_path = output_path_for_version()

    if output_path.exists():
        raise FileExistsError(
            "Evaluation artifact already exists. "
            f"Refusing to overwrite: {output_path}"
        )

    artifact = {
        "component": "MetricRelationEvidence",
        "component_version": METRIC_RELATION_EVIDENCE_VERSION,
        "evaluation": "development_same_metric_detection",
        "model": DEFAULT_MODEL,
        "positive_class": POSITIVE_CLASS,
        "ground_truth": {
            "path": str(GROUND_TRUTH_PATH),
            "sha256": sha256_file(GROUND_TRUTH_PATH),
            "case_count": len(cases),
            "positive_count": positive_count,
        },
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
        },
        "gates": {
            "safety_fp_zero": safety_gate,
            "capability_tp_positive": capability_gate,
            "development_gate": development_gate,
        },
        "prediction_counts": dict(prediction_counts),
        "embedding_evidence_used": False,
        "analytical_authority": "none_semantic_evidence_only",
        "cases": results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("ARTIFACT")
    print("=" * 90)
    print("Saved:", output_path)
    print("SHA256:", sha256_file(output_path))


if __name__ == "__main__":
    main()
