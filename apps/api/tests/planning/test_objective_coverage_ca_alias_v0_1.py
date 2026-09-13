from __future__ import annotations


from app.planning.objective_coverage import (
    build_objective_coverage,
)

from tests.planning.test_objective_coverage_multi_intent_same_metric_v0_1 import (
    build_catalog,
)


def revenue_requirement(
    objective: str,
):
    report = build_objective_coverage(
        objective=objective,
        catalog=build_catalog(),
        contracts=[],
    )

    matches = [
        requirement
        for requirement
        in report.requirements
        if requirement.concept == "revenue_total"
    ]

    return report, matches


def test_ca_explicitly_requires_revenue_total() -> None:
    report, matches = revenue_requirement(
        "CA par catégorie"
    )

    assert report.status == "incomplete"

    assert len(matches) == 1

    requirement = matches[0]

    assert requirement.covered is False

    assert (
        "ca"
        in [
            value.casefold()
            for value
            in requirement.requested_phrases
        ]
    )


def test_ca_does_not_match_inside_another_word() -> None:
    _, matches = revenue_requirement(
        "Catalogue par catégorie"
    )

    assert matches == []


if __name__ == "__main__":
    test_ca_explicitly_requires_revenue_total()

    print(
        "[PASS] CA explicitly triggers revenue_total coverage"
    )

    test_ca_does_not_match_inside_another_word()

    print(
        "[PASS] CA token does not match inside another word"
    )
