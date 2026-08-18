from pathlib import Path

import pandas as pd

from app.preparation.data_quality import (
    QualityIssueKind,
    build_data_quality_report,
)


CSV_PATH = Path(
    "datalens_dirty_customers.csv"
)


if not CSV_PATH.exists():
    raise SystemExit(
        "Place datalens_dirty_customers.csv "
        "in the current directory before running this test."
    )


dataframe = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig",
)


report = build_data_quality_report(
    [
        {
            "dataset_id":
                "dataset:0001",

            "filename":
                CSV_PATH.name,

            "extension":
                ".csv",

            "dataframe":
                dataframe,
        }
    ]
)


kinds = {
    issue.kind
    for issue in report.issues
}


required_kinds = {
    QualityIssueKind.MISSING_VALUES,
    QualityIssueKind.MISSING_IDENTIFIER,
    QualityIssueKind.DUPLICATE_ROWS,
    QualityIssueKind.CONSTANT_COLUMN,
    QualityIssueKind.INVALID_NUMERIC_VALUES,
    QualityIssueKind.NUMERIC_OUTLIERS,
    QualityIssueKind.INVALID_DATES,
    QualityIssueKind.MIXED_DATE_FORMATS,
    QualityIssueKind.CATEGORY_FORMAT_VARIANTS,
    QualityIssueKind.POSSIBLE_SEMANTIC_ALIASES,
    QualityIssueKind.INVALID_EMAILS,
}


missing_required = (
    required_kinds -
    kinds
)


if missing_required:
    raise AssertionError(
        "Missing expected issue kinds: "
        +
        ", ".join(
            sorted(
                kind.value
                for kind in
                missing_required
            )
        )
    )


duplicate_summary = (
    report.datasets[
        0
    ].duplicate_row_count
)


if duplicate_summary != 3:
    raise AssertionError(
        f"Expected 3 exact duplicates, got {duplicate_summary}."
    )


age_issues = [
    issue
    for issue in report.issues
    if issue.column ==
    "age"
]


age_examples = {
    example
    for issue in age_issues
    for example in
    issue.evidence.examples
}


if "thirty" not in age_examples:
    raise AssertionError(
        "Expected 'thirty' to be detected in age."
    )


if "2024" not in age_examples:
    raise AssertionError(
        "Expected age=2024 to be detected as an outlier."
    )


salary_issues = [
    issue
    for issue in report.issues
    if issue.column ==
    "annual_salary"
]


salary_examples = {
    example
    for issue in salary_issues
    for example in
    issue.evidence.examples
}


if not any(
    "45000" in example
    for example in
    salary_examples
):
    raise AssertionError(
        "Expected decorated salary 45000€ to be detected."
    )


if "9999999" not in salary_examples:
    raise AssertionError(
        "Expected extreme salary to be detected."
    )


date_issues = [
    issue
    for issue in report.issues
    if issue.column ==
    "signup_date"
]


date_examples = {
    example
    for issue in date_issues
    for example in
    issue.evidence.examples
}


if not any(
    value in
    date_examples
    for value in (
        "2025-13-40",
        "not_a_date",
    )
):
    raise AssertionError(
        "Expected invalid signup dates to be detected."
    )


print(
    "Data quality engine v0.1 : OK"
)

print(
    "Rule version :",
    report.rule_version,
)

print(
    "Issues :",
    report.issue_count,
)

print(
    "Important :",
    report.important_count,
)

print(
    "Moderate :",
    report.moderate_count,
)

print(
    "Minor :",
    report.minor_count,
)

print(
    "Semantic review candidates :",
    report.semantic_review_count,
)

print(
    "Exact duplicates :",
    duplicate_summary,
)


for issue in report.issues:
    print(
        "-",
        issue.severity.value,
        "|",
        issue.kind.value,
        "|",
        issue.column
        or "<dataset>",
        "|",
        issue.evidence.examples,
    )
