from __future__ import annotations


from copy import deepcopy


from pydantic import (
    ValidationError,
)


import app.planning.analytical_contract as contract_module


from app.planning.analytical_contract import (
    ANALYTICAL_CONTRACT_RULE_VERSION,
    AnalyticalContract,
)


# ============================================================
# FUTURE CANONICAL VOCABULARY
# ============================================================


SHARE_FIELD = (
    "share_of_total"
)


SHARE_REFERENCE = (
    "sum_of_group_values"
)


# ============================================================
# GENERIC CONTRACT PAYLOADS
# ============================================================


def ranking_payload() -> dict:

    return {
        "contract_id":
            "contract:ranking:share:01",

        "origin":
            "ai_planner",

        "status":
            "validated",

        "title":
            "Category contribution ranking",

        "request_text":
            (
                "Which category contributes the most revenue "
                "and what share of total revenue does it represent?"
            ),

        "family":
            "ranking",

        "required_dataset_ids": [
            "dataset:category_revenue",
        ],

        "required_dataset_filenames": [
            "category_revenue.derived",
        ],

        "analytical_grain":
            "category",

        "bindings": [
            {
                "role":
                    "value",

                "column":
                    "sum_value",

                "dataset_id":
                    "dataset:category_revenue",

                "dataset_filename":
                    "category_revenue.derived",

                "semantic_concept":
                    "revenue",

                "analysis_kind":
                    "quantitative",
            },

            {
                "role":
                    "dimension",

                "column":
                    "category",

                "dataset_id":
                    "dataset:category_revenue",

                "dataset_filename":
                    "category_revenue.derived",

                "semantic_concept":
                    None,

                "analysis_kind":
                    "categorical",
            },
        ],

        "aggregation": {
            "function":
                "sum",

            "source_role":
                "value",

            "group_by_roles": [
                "dimension",
            ],

            "output_name":
                "planned_metric",
        },

        "ranking": {
            "order":
                "descending",

            "limit":
                1,
        },

        "benchmark":
            None,

        "window":
            None,

        "filters": [],
        "joins": [],
        "derived_variables": [],

        "required_operations": [
            "Execute deterministic ranking.",
        ],

        "reasons": [
            "Generic share-of-total contract test.",
        ],

        "blockers": [],

        "planner_confidence":
            1.0,
    }


def grouped_aggregation_payload() -> dict:

    payload = ranking_payload()

    payload[
        "contract_id"
    ] = (
        "contract:aggregation:share:01"
    )

    payload[
        "family"
    ] = (
        "aggregation"
    )

    payload[
        "ranking"
    ] = None


    return payload


def with_share(
    payload: dict,
    *,
    reference: str = SHARE_REFERENCE,
) -> dict:

    candidate = deepcopy(
        payload
    )


    candidate[
        SHARE_FIELD
    ] = {
        "reference":
            reference,
    }


    return candidate


# ============================================================
# ERROR HELPERS
# ============================================================


def validation_errors(
    payload: dict,
) -> list[
    dict
]:

    try:

        AnalyticalContract.model_validate(
            payload
        )

    except ValidationError as error:

        return error.errors()


    return []


def validation_messages(
    payload: dict,
) -> list[
    str
]:

    return [
        str(
            error.get(
                "msg",
                "",
            )
        )

        for error
        in validation_errors(
            payload
        )
    ]


# ============================================================
# EXISTING CONTRACT PRESERVATION
# ============================================================


def assert_existing_contract_preserved() -> None:

    contract = (
        AnalyticalContract.model_validate(
            ranking_payload()
        )
    )


    assert (
        contract.family
        ==
        "ranking"
    )


    assert (
        contract.ranking
        is not None
    )


    assert (
        contract.ranking.order
        ==
        "descending"
    )


    assert (
        contract.ranking.limit
        ==
        1
    )


# ============================================================
# FUTURE CONTRACT GUARDS
#
# These execute once the canonical field exists.
# Until then, the explicit RED gaps below remain the authority.
# ============================================================


def assert_future_guards() -> None:

    if (
        SHARE_FIELD
        not in
        AnalyticalContract.model_fields
    ):

        return


    # --------------------------------------------------------
    # Positive ranking contract
    # --------------------------------------------------------

    ranking = (
        AnalyticalContract.model_validate(
            with_share(
                ranking_payload()
            )
        )
    )


    share = getattr(
        ranking,
        SHARE_FIELD,
    )


    assert (
        share
        is not None
    )


    assert (
        share.reference
        ==
        SHARE_REFERENCE
    )


    # --------------------------------------------------------
    # Positive grouped aggregation contract
    # --------------------------------------------------------

    aggregation = (
        AnalyticalContract.model_validate(
            with_share(
                grouped_aggregation_payload()
            )
        )
    )


    assert (
        getattr(
            aggregation,
            SHARE_FIELD,
        )
        is not None
    )


    # --------------------------------------------------------
    # Negative: scalar/non-grouped aggregation
    # --------------------------------------------------------

    scalar = (
        grouped_aggregation_payload()
    )


    scalar[
        "analytical_grain"
    ] = (
        "overall"
    )


    scalar[
        "bindings"
    ] = [
        scalar[
            "bindings"
        ][
            0
        ]
    ]


    scalar[
        "aggregation"
    ][
        "group_by_roles"
    ] = []


    scalar_errors = (
        validation_errors(
            with_share(
                scalar
            )
        )
    )


    assert scalar_errors, (
        "share_of_total must reject an ungrouped/scalar "
        "aggregation contract."
    )


    # --------------------------------------------------------
    # Negative: non-additive aggregation
    # --------------------------------------------------------

    non_sum = (
        ranking_payload()
    )


    non_sum[
        "aggregation"
    ][
        "function"
    ] = (
        "mean"
    )


    non_sum_errors = (
        validation_errors(
            with_share(
                non_sum
            )
        )
    )


    assert non_sum_errors, (
        "share_of_total must reject non-SUM aggregation."
    )


    # --------------------------------------------------------
    # Negative: unsupported family
    # --------------------------------------------------------

    unsupported = (
        grouped_aggregation_payload()
    )


    unsupported[
        "family"
    ] = (
        "descriptive_metric"
    )


    unsupported_errors = (
        validation_errors(
            with_share(
                unsupported
            )
        )
    )


    assert unsupported_errors, (
        "share_of_total must remain restricted to grouped "
        "aggregation/ranking contracts."
    )


    # --------------------------------------------------------
    # Negative: unknown denominator reference
    # --------------------------------------------------------

    unknown_reference_errors = (
        validation_errors(
            with_share(
                ranking_payload(),
                reference=
                    "some_other_population",
            )
        )
    )


    assert unknown_reference_errors, (
        "Unknown share denominator references must fail closed."
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS CANONICAL SHARE-OF-TOTAL CONTRACT v0.1 ==="
    )

    print()


    assert_existing_contract_preserved()


    print(
        "[PASS] existing ranking contract without share preserved"
    )


    gaps: list[
        str
    ] = []


    # ========================================================
    # 1. CANONICAL SPEC CLASS
    # ========================================================

    spec_class = getattr(
        contract_module,
        "ShareOfTotalSpec",
        None,
    )


    print(
        "ShareOfTotalSpec class                  "
        f"{'PRESENT' if spec_class is not None else 'MISSING'}"
    )


    if (
        spec_class
        is None
    ):

        gaps.append(
            "share_of_total_spec_class_gap"
        )


    # ========================================================
    # 2. CONTRACT FIELD
    # ========================================================

    field_present = (
        SHARE_FIELD
        in
        AnalyticalContract.model_fields
    )


    print(
        "AnalyticalContract.share_of_total       "
        f"{'PRESENT' if field_present else 'MISSING'}"
    )


    if not field_present:

        gaps.append(
            "share_of_total_contract_field_gap"
        )


    # ========================================================
    # 3. POSITIVE CANONICAL CONTRACT
    # ========================================================

    positive_errors = (
        validation_errors(
            with_share(
                ranking_payload()
            )
        )
    )


    print(
        "Positive ranking share validation       "
        f"{(
            'PASS'
            if not positive_errors
            else 'FAIL'
        )}"
    )


    if (
        positive_errors
    ):

        print(
            "Positive validation errors              "
            f"{positive_errors}"
        )


        gaps.append(
            "share_of_total_positive_contract_gap"
        )


    # ========================================================
    # 4. FUTURE STRICT GUARDS
    # ========================================================

    if (
        spec_class
        is not None

        and

        field_present

        and

        not positive_errors
    ):

        assert_future_guards()


        print(
            "[PASS] grouped SUM ranking accepted"
        )

        print(
            "[PASS] grouped SUM aggregation accepted"
        )

        print(
            "[PASS] scalar share contract rejected"
        )

        print(
            "[PASS] non-SUM share contract rejected"
        )

        print(
            "[PASS] unsupported family rejected"
        )

        print(
            "[PASS] unknown denominator reference rejected"
        )


    else:

        print(
            "Future strict guards                    DEFERRED UNTIL SPEC EXISTS"
        )


    # ========================================================
    # 5. VERSION
    # ========================================================

    print()
    print(
        "Analytical Contract rule                "
        f"{ANALYTICAL_CONTRACT_RULE_VERSION}"
    )

    print(
        "Required reference                      "
        f"{SHARE_REFERENCE}"
    )

    print(
        "Observed gaps                           "
        f"{gaps}"
    )


    if gaps:

        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                ",".join(
                    gaps
                )
            )
        )


    print()
    print(
        "PASS - canonical share-of-total contract v0.1"
    )


if __name__ == "__main__":

    main()