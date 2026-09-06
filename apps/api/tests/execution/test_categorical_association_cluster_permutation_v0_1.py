from __future__ import annotations


import math


import numpy as np
import pandas as pd


from scipy.stats import (
    chi2_contingency,
)


from app.execution.executor import (
    execute_categorical_association,
)


from app.planning.schemas import (
    AnalysisCandidate,
    PlannedVariable,
)


# ============================================================
# FUTURE GENERIC CONTRACT
# ============================================================


EXPECTED_TEST_NAME = (
    "cluster_permutation_chi_square"
)


EXPECTED_PERMUTATION_COUNT = (
    9999
)


EXPECTED_PERMUTATION_SEED = (
    20260904
)


EXPECTED_INFERENCE_SCOPE = (
    "cluster_aware"
)


EXPECTED_GAPS = [
    "categorical_cluster_permutation_execution_gap",
    "categorical_cluster_permutation_binding_symmetry_gap",
    "categorical_cluster_permutation_row_order_determinism_gap",
]


# ============================================================
# CANDIDATE
# ============================================================


def candidate(
    *,
    dataset_id: str,
    x_column: str,
    y_column: str,
) -> AnalysisCandidate:

    return AnalysisCandidate(
        analysis_id=(
            "test:"
            f"{dataset_id}:"
            f"{x_column}:"
            f"{y_column}"
        ),
        dataset_id=dataset_id,
        dataset_filename=(
            f"{dataset_id}.csv"
        ),
        title=(
            "Synthetic categorical association"
        ),
        family="categorical_association",
        priority_score=100,
        readiness="executable_now",
        variables=[
            PlannedVariable(
                column=x_column,
                role="x",
                analysis_kind="categorical",
            ),
            PlannedVariable(
                column=y_column,
                role="y",
                analysis_kind="categorical",
            ),
        ],
        chart_type="heatmap",
        statistical_strategy=(
            "chi_square_or_fisher_decision_engine"
        ),
        reasons=[
            (
                "Synthetic deterministic "
                "categorical-association regression."
            )
        ],
        limitations=[],
    )


# ============================================================
# SYNTHETIC DATASETS
# ============================================================


def independent_frame() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "gender": [
                "f",
                "f",
                "f",
                "f",
                "m",
                "m",
                "m",
                "m",
            ],
            "category": [
                "a",
                "a",
                "b",
                "b",
                "a",
                "b",
                "b",
                "b",
            ],
        }
    )


def clustered_frame() -> pd.DataFrame:

    customer_patterns = {
        "c01": (
            "f",
            [
                "a",
                "a",
                "a",
                "b",
            ],
        ),
        "c02": (
            "f",
            [
                "a",
                "a",
                "a",
                "b",
            ],
        ),
        "c03": (
            "f",
            [
                "a",
                "a",
                "b",
                "b",
            ],
        ),
        "c04": (
            "f",
            [
                "a",
                "b",
                "b",
                "b",
            ],
        ),
        "c05": (
            "f",
            [
                "a",
                "a",
                "b",
                "b",
            ],
        ),
        "c06": (
            "m",
            [
                "a",
                "b",
                "b",
                "b",
            ],
        ),
        "c07": (
            "m",
            [
                "b",
                "b",
                "b",
                "b",
            ],
        ),
        "c08": (
            "m",
            [
                "a",
                "b",
                "b",
                "b",
            ],
        ),
        "c09": (
            "m",
            [
                "a",
                "a",
                "b",
                "b",
            ],
        ),
        "c10": (
            "m",
            [
                "b",
                "b",
                "b",
                "b",
            ],
        ),
    }


    rows: list[
        dict[
            str,
            object,
        ]
    ] = []


    base_date = pd.Timestamp(
        "2026-01-01"
    )


    for (
        customer_index,
        (
            customer_id,
            (
                gender,
                categories,
            ),
        ),
    ) in enumerate(
        customer_patterns.items()
    ):

        for (
            event_index,
            category,
        ) in enumerate(
            categories
        ):

            rows.append(
                {
                    "customer_id":
                        customer_id,

                    "event_time":
                        (
                            base_date
                            +
                            pd.Timedelta(
                                days=(
                                    customer_index
                                    *
                                    10
                                    +
                                    event_index
                                )
                            )
                        ),

                    "gender":
                        gender,

                    "category":
                        category,
                }
            )


    return pd.DataFrame(
        rows
    )


def unstable_within_cluster_frame() -> pd.DataFrame:

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []


    base_date = pd.Timestamp(
        "2026-02-01"
    )


    for customer_index in range(
        8
    ):

        customer_id = (
            f"u{customer_index:02d}"
        )


        for event_index in range(
            4
        ):

            rows.append(
                {
                    "customer_id":
                        customer_id,

                    "event_time":
                        (
                            base_date
                            +
                            pd.Timedelta(
                                days=(
                                    customer_index
                                    *
                                    10
                                    +
                                    event_index
                                )
                            )
                        ),

                    # Both analytical variables change
                    # inside the same customer cluster.
                    "gender":
                        (
                            "f"
                            if (
                                event_index
                                %
                                2
                                ==
                                0
                            )
                            else
                            "m"
                        ),

                    "category":
                        (
                            "a"
                            if (
                                event_index
                                in {
                                    0,
                                    3,
                                }
                            )
                            else
                            "b"
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# AUTHORITIES
# ============================================================


def observed_authority(
    dataframe: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
) -> dict[
    str,
    object,
]:

    working = (
        dataframe[
            [
                x_column,
                y_column,
            ]
        ]
        .dropna()
    )


    contingency = pd.crosstab(
        working[
            x_column
        ],
        working[
            y_column
        ],
    )


    (
        chi2,
        naive_p_value,
        dof,
        expected,
    ) = chi2_contingency(
        contingency
    )


    n = int(
        contingency
        .to_numpy()
        .sum()
    )


    rows, columns = (
        contingency.shape
    )


    denominator = min(
        rows
        -
        1,

        columns
        -
        1,
    )


    cramers_v = math.sqrt(
        (
            float(
                chi2
            )
            /
            n
        )
        /
        denominator
    )


    return {
        "chi2":
            float(
                chi2
            ),

        "naive_p_value":
            float(
                naive_p_value
            ),

        "degrees_of_freedom":
            int(
                dof
            ),

        "cramers_v":
            float(
                cramers_v
            ),

        "n":
            n,
    }


# ============================================================
# FUTURE CLUSTER-AWARE RESULT SHAPE
# ============================================================


def is_cluster_aware_complete(
    result,
    *,
    expected_cluster_label_column: str,
    expected_response_column: str,
    expected_authority: dict[
        str,
        object,
    ],
) -> bool:

    if (
        result.execution_status
        !=
        "complete"
    ):

        return False


    statistical_result = (
        result.statistical_result
    )


    if not isinstance(
        statistical_result,
        dict,
    ):

        return False


    required_keys = {
        "test",
        "chi2",
        "p_value",
        "degrees_of_freedom",
        "cramers_v",
        "n",
        "alpha",
        "statistically_significant",
        "inference_scope",
        "cluster_column",
        "cluster_count",
        "cluster_label_column",
        "response_column",
        "permutation_count",
        "permutation_seed",
        "null_exceedance_count",
        "monte_carlo_standard_error",
    }


    if not required_keys.issubset(
        statistical_result.keys()
    ):

        return False


    if (
        statistical_result[
            "test"
        ]
        !=
        EXPECTED_TEST_NAME
    ):

        return False


    if (
        statistical_result[
            "inference_scope"
        ]
        !=
        EXPECTED_INFERENCE_SCOPE
    ):

        return False


    if (
        statistical_result[
            "cluster_column"
        ]
        !=
        "customer_id"
    ):

        return False


    if (
        statistical_result[
            "cluster_label_column"
        ]
        !=
        expected_cluster_label_column
    ):

        return False


    if (
        statistical_result[
            "response_column"
        ]
        !=
        expected_response_column
    ):

        return False


    if (
        statistical_result[
            "permutation_count"
        ]
        !=
        EXPECTED_PERMUTATION_COUNT
    ):

        return False


    if (
        statistical_result[
            "permutation_seed"
        ]
        !=
        EXPECTED_PERMUTATION_SEED
    ):

        return False


    if (
        "naive_p_value"
        in
        statistical_result
    ):

        return False


    if not math.isclose(
        float(
            statistical_result[
                "chi2"
            ]
        ),
        float(
            expected_authority[
                "chi2"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):

        return False


    if not math.isclose(
        float(
            statistical_result[
                "cramers_v"
            ]
        ),
        float(
            expected_authority[
                "cramers_v"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):

        return False


    if (
        statistical_result[
            "degrees_of_freedom"
        ]
        !=
        expected_authority[
            "degrees_of_freedom"
        ]
    ):

        return False


    if (
        statistical_result[
            "n"
        ]
        !=
        expected_authority[
            "n"
        ]
    ):

        return False


    p_value = float(
        statistical_result[
            "p_value"
        ]
    )


    if not (
        0.0
        <
        p_value
        <=
        1.0
    ):

        return False


    monte_carlo_standard_error = float(
        statistical_result[
            "monte_carlo_standard_error"
        ]
    )


    if not (
        math.isfinite(
            monte_carlo_standard_error
        )
        and
        monte_carlo_standard_error
        >=
        0.0
    ):

        return False


    cluster_count = int(
        statistical_result[
            "cluster_count"
        ]
    )


    if (
        cluster_count
        <
        2
    ):

        return False


    exceedance_count = int(
        statistical_result[
            "null_exceedance_count"
        ]
    )


    if not (
        0
        <=
        exceedance_count
        <=
        EXPECTED_PERMUTATION_COUNT
    ):

        return False


    return True


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS CATEGORICAL ASSOCIATION "
        "CLUSTER PERMUTATION v0.1 ==="
    )

    print()


    # ========================================================
    # 1. INDEPENDENT BASELINE MUST REMAIN CLASSICAL
    # ========================================================

    independent = (
        independent_frame()
    )


    independent_result = (
        execute_categorical_association(
            candidate(
                dataset_id="dataset:independent",
                x_column="gender",
                y_column="category",
            ),
            independent,
        )
    )


    assert (
        independent_result.execution_status
        ==
        "complete"
    )


    assert isinstance(
        independent_result.statistical_result,
        dict,
    )


    assert (
        independent_result.statistical_result[
            "test"
        ]
        ==
        "chi_square_independence"
    )


    print(
        "[PASS] independent observations retain classical chi-square"
    )


    # ========================================================
    # 2. CURRENT REPEATED-MEASURE CASE
    # ========================================================

    clustered = (
        clustered_frame()
    )


    clustered_authority = (
        observed_authority(
            clustered,
            x_column="gender",
            y_column="category",
        )
    )


    clustered_result = (
        execute_categorical_association(
            candidate(
                dataset_id="dataset:clustered",
                x_column="gender",
                y_column="category",
            ),
            clustered,
        )
    )


    cluster_execution_ok = (
        is_cluster_aware_complete(
            clustered_result,
            expected_cluster_label_column=
                "gender",

            expected_response_column=
                "category",

            expected_authority=
                clustered_authority,
        )
    )


    print(
        "Cluster-aware repeated execution         "
        f"{cluster_execution_ok}"
    )


    print(
        "  observed status                        "
        f"{clustered_result.execution_status}"
    )


    print(
        "  observed statistical result            "
        f"{clustered_result.statistical_result}"
    )


    # ========================================================
    # 3. X / Y SYMMETRY
    #
    # `category` varies inside customer; `gender` is stable.
    # The executor must detect `gender` as cluster label even
    # when it is bound to y rather than x.
    # ========================================================

    swapped_authority = (
        observed_authority(
            clustered,
            x_column="category",
            y_column="gender",
        )
    )


    swapped_result = (
        execute_categorical_association(
            candidate(
                dataset_id="dataset:clustered-swapped",
                x_column="category",
                y_column="gender",
            ),
            clustered,
        )
    )


    swapped_ok = (
        is_cluster_aware_complete(
            swapped_result,
            expected_cluster_label_column=
                "gender",

            expected_response_column=
                "category",

            expected_authority=
                swapped_authority,
        )
    )


    symmetry_ok = False


    if (
        cluster_execution_ok
        and
        swapped_ok
    ):

        left = (
            clustered_result
            .statistical_result
        )


        right = (
            swapped_result
            .statistical_result
        )


        assert isinstance(
            left,
            dict,
        )


        assert isinstance(
            right,
            dict,
        )


        symmetry_ok = bool(
            math.isclose(
                float(
                    left[
                        "chi2"
                    ]
                ),
                float(
                    right[
                        "chi2"
                    ]
                ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and
            math.isclose(
                float(
                    left[
                        "cramers_v"
                    ]
                ),
                float(
                    right[
                        "cramers_v"
                    ]
                ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and
            math.isclose(
                float(
                    left[
                        "p_value"
                    ]
                ),
                float(
                    right[
                        "p_value"
                    ]
                ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and
            left[
                "null_exceedance_count"
            ]
            ==
            right[
                "null_exceedance_count"
            ]
        )


    print(
        "Binding-order symmetry                   "
        f"{symmetry_ok}"
    )


    # ========================================================
    # 4. FIXED-SEED + ROW-ORDER DETERMINISM
    # ========================================================

    shuffled = (
        clustered
        .sample(
            frac=1.0,
            random_state=12345,
        )
        .reset_index(
            drop=True
        )
    )


    shuffled_result = (
        execute_categorical_association(
            candidate(
                dataset_id="dataset:clustered-shuffled",
                x_column="gender",
                y_column="category",
            ),
            shuffled,
        )
    )


    shuffled_ok = (
        is_cluster_aware_complete(
            shuffled_result,
            expected_cluster_label_column=
                "gender",

            expected_response_column=
                "category",

            expected_authority=
                clustered_authority,
        )
    )


    row_order_determinism_ok = False


    if (
        cluster_execution_ok
        and
        shuffled_ok
    ):

        left = (
            clustered_result
            .statistical_result
        )


        right = (
            shuffled_result
            .statistical_result
        )


        assert isinstance(
            left,
            dict,
        )


        assert isinstance(
            right,
            dict,
        )


        row_order_determinism_ok = bool(
            left[
                "p_value"
            ]
            ==
            right[
                "p_value"
            ]
            and
            left[
                "null_exceedance_count"
            ]
            ==
            right[
                "null_exceedance_count"
            ]
            and
            left[
                "permutation_seed"
            ]
            ==
            right[
                "permutation_seed"
            ]
            and
            left[
                "permutation_count"
            ]
            ==
            right[
                "permutation_count"
            ]
        )


    print(
        "Fixed-seed row-order determinism         "
        f"{row_order_determinism_ok}"
    )


    # ========================================================
    # 5. BOTH VARIABLES VARY WITHIN CLUSTER
    #
    # No valid between-cluster label exists.
    # Existing safety behavior must remain.
    # ========================================================

    unstable = (
        unstable_within_cluster_frame()
    )


    unstable_result = (
        execute_categorical_association(
            candidate(
                dataset_id="dataset:unstable",
                x_column="gender",
                y_column="category",
            ),
            unstable,
        )
    )


    unstable_guard_ok = bool(
        unstable_result.execution_status
        ==
        "descriptive_only"
        and
        unstable_result.statistical_result
        is None
    )


    assert (
        unstable_guard_ok
    )


    print(
        "[PASS] no cluster-stable categorical label remains fail-closed"
    )


    # ========================================================
    # 6. NAIVE P-VALUE MUST NOT BECOME INFERENCE AUTHORITY
    # ========================================================

    if cluster_execution_ok:

        statistical_result = (
            clustered_result
            .statistical_result
        )


        assert isinstance(
            statistical_result,
            dict,
        )


        assert (
            "naive_p_value"
            not in
            statistical_result
        )


        print(
            "[PASS] naive classical p-value is not exposed as inference authority"
        )


    else:

        print(
            "[INFO] naive-p authority guard awaits cluster-aware implementation"
        )


    # ========================================================
    # 7. RED MATRIX
    # ========================================================

    gaps: list[
        str
    ] = []


    if not (
        cluster_execution_ok
    ):

        gaps.append(
            EXPECTED_GAPS[
                0
            ]
        )


    if not (
        symmetry_ok
    ):

        gaps.append(
            EXPECTED_GAPS[
                1
            ]
        )


    if not (
        row_order_determinism_ok
    ):

        gaps.append(
            EXPECTED_GAPS[
                2
            ]
        )


    print()
    print("=" * 80)
    print(
        "P4-R5 CLUSTER-AWARE INFERENCE MATRIX"
    )
    print("=" * 80)
    print()


    print(
        "Independent classical chi-square         True"
    )

    print(
        "Repeated cluster-aware execution         "
        f"{cluster_execution_ok}"
    )

    print(
        "Binding-order symmetry                   "
        f"{symmetry_ok}"
    )

    print(
        "Fixed-seed row-order determinism         "
        f"{row_order_determinism_ok}"
    )

    print(
        "No stable cluster label fail-closed      "
        f"{unstable_guard_ok}"
    )

    print()

    print(
        "Observed gaps                            "
        f"{gaps}"
    )


    # ========================================================
    # TEST IS BOTH RED HARNESS AND FUTURE GREEN REGRESSION.
    #
    # Before implementation:
    #   exact expected gaps -> RED_EXPECTED.
    #
    # After implementation:
    #   gaps == [] -> permanent PASS.
    # ========================================================

    if gaps:

        if (
            gaps
            !=
            EXPECTED_GAPS
        ):

            raise AssertionError(
                (
                    "Unexpected P4-R5 RED gap set: "
                    f"{gaps!r}"
                )
            )


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
        "PASS - categorical association "
        "cluster permutation v0.1"
    )

    print(
        "P4-R5 REGRESSION: GREEN"
    )


if __name__ == "__main__":

    main()