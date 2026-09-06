from __future__ import annotations


import numpy as np
import pandas as pd


from app.statistics.decision import (
    MATERIAL_OUTLIER_FRACTION,
    MINIMUM_MATERIAL_OUTLIER_COUNT,
    decide_correlation_test,
    has_material_outlier_signal,
)


EXPECTED_RED_GAP = (
    "general_association_material_outlier_"
    "monotonic_rank_recovery_gap"
)


EXPECTED_MATERIAL_OUTLIER_FRACTION = 0.05

EXPECTED_MINIMUM_MATERIAL_OUTLIER_COUNT = 2


# ============================================================
# SYNTHETIC TARGET
# ============================================================


def monotonic_material_outlier_frame() -> pd.DataFrame:
    """
    Strictly increasing non-linear relationship.

    y = x**4 produces:

    - a strong monotonic rank relation;
    - a weaker linear relation;
    - a material upper-tail IQR signal;
    - no case-specific data or constants.

    The outlier signal is a property of the synthetic
    distribution, not manually injected benchmark values.
    """

    x = np.linspace(
        0.0,
        1.0,
        2000,
    )


    y = (
        x
        ** 4
    )


    return pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    )


# ============================================================
# NEGATIVE CONTROL
# ============================================================


def no_clear_material_outlier_frame() -> pd.DataFrame:
    """
    Independent deterministic noise with extreme values.

    This proves the repair must NOT become:

        material outliers -> Spearman.

    If shape remains no_clear_pattern, the current
    fail-closed behavior must remain.
    """

    sample_size = 2000


    x = np.linspace(
        -1.0,
        1.0,
        sample_size,
    )


    rng = np.random.default_rng(
        20260904
    )


    y = rng.normal(
        loc=0.0,
        scale=1.0,
        size=sample_size,
    )


    extreme_indices = np.arange(
        0,
        sample_size,
        16,
    )


    extreme_signs = np.where(
        (
            np.arange(
                extreme_indices.size
            )
            % 2
        )
        ==
        0,
        1.0,
        -1.0,
    )


    y[
        extreme_indices
    ] = (
        y[
            extreme_indices
        ]
        +
        (
            extreme_signs
            *
            12.0
        )
    )


    return pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    )


# ============================================================
# DECISION HELPERS
# ============================================================


def decide(
    dataframe: pd.DataFrame,
    *,
    analysis_goal: str = "general_association",
    analysis_mode: str = "exploratory",
    observations_independent: bool | None = True,
):

    return decide_correlation_test(
        dataframe=
            dataframe,

        x_column=
            "x",

        y_column=
            "y",

        analysis_goal=
            analysis_goal,

        analysis_mode=
            analysis_mode,

        x_kind=
            "continuous",

        y_kind=
            "continuous",

        observations_independent=
            observations_independent,
    )


def material_signal(
    decision,
) -> bool:

    diagnostics = (
        decision.diagnostics
    )


    x_signal = (
        has_material_outlier_signal(
            outlier_count=
                diagnostics.x_outlier_count,

            outlier_fraction=
                diagnostics.x_outlier_fraction,
        )
    )


    y_signal = (
        has_material_outlier_signal(
            outlier_count=
                diagnostics.y_outlier_count,

            outlier_fraction=
                diagnostics.y_outlier_fraction,
        )
    )


    return bool(
        x_signal
        or
        y_signal
    )


def future_rank_recovery_satisfied(
    decision,
) -> bool:

    return bool(
        decision.status
        ==
        "selected"

        and

        decision.selected_test
        ==
        "spearman"

        and

        decision.inference_method
        ==
        "standard"

        and

        decision.selection_is_data_driven
        is True

        and

        decision.diagnostics.shape_signal
        ==
        "monotonic_non_linear_candidate"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS MATERIAL-OUTLIER "
        "MONOTONIC RANK RECOVERY v0.1 ==="
    )

    print()


    # ========================================================
    # 1. THRESHOLD AUTHORITY
    # ========================================================

    assert (
        MATERIAL_OUTLIER_FRACTION
        ==
        EXPECTED_MATERIAL_OUTLIER_FRACTION
    )


    assert (
        MINIMUM_MATERIAL_OUTLIER_COUNT
        ==
        EXPECTED_MINIMUM_MATERIAL_OUTLIER_COUNT
    )


    print(
        "[PASS] material-outlier fraction threshold remains 0.05"
    )

    print(
        "[PASS] minimum material-outlier count remains 2"
    )


    # ========================================================
    # 2. TARGET RED CONTROL
    # ========================================================

    target_frame = (
        monotonic_material_outlier_frame()
    )


    target = decide(
        target_frame
    )


    target_diagnostics = (
        target.diagnostics
    )


    target_material = (
        material_signal(
            target
        )
    )


    assert target_material is True


    assert (
        target_diagnostics.shape_signal
        ==
        "monotonic_non_linear_candidate"
    )


    assert (
        target_diagnostics.spearman_coefficient
        is not None
    )


    assert (
        target_diagnostics.pearson_coefficient
        is not None
    )


    assert (
        abs(
            target_diagnostics.spearman_coefficient
        )
        >
        abs(
            target_diagnostics.pearson_coefficient
        )
    )


    target_future_satisfied = (
        future_rank_recovery_satisfied(
            target
        )
    )


    print()

    print(
        "Material-outlier monotonic candidate"
    )

    print(
        "  n valid                                "
        f"{target_diagnostics.n_valid}"
    )

    print(
        "  x outlier count                        "
        f"{target_diagnostics.x_outlier_count}"
    )

    print(
        "  x outlier fraction                     "
        f"{target_diagnostics.x_outlier_fraction}"
    )

    print(
        "  y outlier count                        "
        f"{target_diagnostics.y_outlier_count}"
    )

    print(
        "  y outlier fraction                     "
        f"{target_diagnostics.y_outlier_fraction}"
    )

    print(
        "  material signal                        "
        f"{target_material}"
    )

    print(
        "  Pearson                                "
        f"{target_diagnostics.pearson_coefficient}"
    )

    print(
        "  Spearman                               "
        f"{target_diagnostics.spearman_coefficient}"
    )

    print(
        "  coefficient gap                        "
        f"{target_diagnostics.coefficient_gap}"
    )

    print(
        "  shape signal                           "
        f"{target_diagnostics.shape_signal}"
    )

    print(
        "  status                                 "
        f"{target.status}"
    )

    print(
        "  selected test                          "
        f"{target.selected_test}"
    )

    print(
        "  inference method                       "
        f"{target.inference_method}"
    )

    print(
        "  selection is data-driven               "
        f"{target.selection_is_data_driven}"
    )

    print(
        "  future rank recovery satisfied          "
        f"{target_future_satisfied}"
    )


    # ========================================================
    # 3. NEGATIVE CONTROL:
    # NO-CLEAR + OUTLIERS MUST REMAIN FAIL-CLOSED
    # ========================================================

    no_clear_frame = (
        no_clear_material_outlier_frame()
    )


    no_clear = decide(
        no_clear_frame
    )


    no_clear_diagnostics = (
        no_clear.diagnostics
    )


    no_clear_material = (
        material_signal(
            no_clear
        )
    )


    assert no_clear_material is True


    assert (
        no_clear_diagnostics.shape_signal
        ==
        "no_clear_pattern"
    )


    assert (
        no_clear.status
        ==
        "needs_information"
    )


    assert (
        no_clear.selected_test
        is None
    )


    assert (
        no_clear.inference_method
        is None
    )


    print()

    print(
        "No-clear-pattern material-outlier control"
    )

    print(
        "  material signal                        "
        f"{no_clear_material}"
    )

    print(
        "  Pearson                                "
        f"{no_clear_diagnostics.pearson_coefficient}"
    )

    print(
        "  Spearman                               "
        f"{no_clear_diagnostics.spearman_coefficient}"
    )

    print(
        "  shape signal                           "
        f"{no_clear_diagnostics.shape_signal}"
    )

    print(
        "  status                                 "
        f"{no_clear.status}"
    )

    print(
        "  selected test                          "
        f"{no_clear.selected_test}"
    )


    print(
        "[PASS] outliers alone do not automatically select Spearman"
    )


    # ========================================================
    # 4. EXPLICIT LINEAR PRESERVED
    # ========================================================

    explicit_linear = decide(
        target_frame,
        analysis_goal=
            "linear_association",
    )


    assert (
        explicit_linear.status
        ==
        "selected"
    )


    assert (
        explicit_linear.selected_test
        ==
        "pearson"
    )


    assert (
        explicit_linear.inference_method
        ==
        "permutation_recommended"
    )


    assert (
        explicit_linear.selection_is_data_driven
        is False
    )


    print(
        "[PASS] explicit linear goal remains Pearson"
    )


    # ========================================================
    # 5. EXPLICIT MONOTONIC PRESERVED
    # ========================================================

    explicit_monotonic = decide(
        target_frame,
        analysis_goal=
            "monotonic_association",
    )


    assert (
        explicit_monotonic.status
        ==
        "selected"
    )


    assert (
        explicit_monotonic.selected_test
        ==
        "spearman"
    )


    assert (
        explicit_monotonic.inference_method
        ==
        "standard"
    )


    assert (
        explicit_monotonic.selection_is_data_driven
        is False
    )


    print(
        "[PASS] explicit monotonic goal remains Spearman"
    )


    # ========================================================
    # 6. CONFIRMATORY GENERAL PRESERVED
    # ========================================================

    confirmatory_general = decide(
        target_frame,
        analysis_goal=
            "general_association",

        analysis_mode=
            "confirmatory",
    )


    assert (
        confirmatory_general.status
        ==
        "needs_information"
    )


    assert (
        confirmatory_general.selected_test
        is None
    )


    print(
        "[PASS] confirmatory general association remains fail-closed"
    )


    # ========================================================
    # 7. UNKNOWN INDEPENDENCE PRESERVED
    # ========================================================

    unknown_independence = decide(
        target_frame,
        observations_independent=
            None,
    )


    assert (
        unknown_independence.status
        ==
        "needs_information"
    )


    assert (
        unknown_independence.selected_test
        is None
    )


    print(
        "[PASS] unknown observation independence remains fail-closed"
    )


    # ========================================================
    # 8. REPEATED OBSERVATIONS PRESERVED
    # ========================================================

    repeated_observations = decide(
        target_frame,
        observations_independent=
            False,
    )


    assert (
        repeated_observations.status
        ==
        "not_applicable"
    )


    assert (
        repeated_observations.selected_test
        is None
    )


    print(
        "[PASS] repeated observations remain outside simple correlation"
    )


    # ========================================================
    # 9. RED MATRIX
    # ========================================================

    gaps = []


    if not target_future_satisfied:

        gaps.append(
            EXPECTED_RED_GAP
        )


    print()

    print(
        "=" * 80
    )

    print(
        "P6-R1 MATERIAL-OUTLIER MONOTONIC RECOVERY MATRIX"
    )

    print(
        "=" * 80
    )

    print()


    print(
        "Material threshold unchanged             "
        f"{MATERIAL_OUTLIER_FRACTION == 0.05}"
    )

    print(
        "Material monotonic diagnostic confirmed  "
        f"{target_diagnostics.shape_signal == 'monotonic_non_linear_candidate'}"
    )

    print(
        "Material signal confirmed                "
        f"{target_material}"
    )

    print(
        "Future rank recovery                     "
        f"{target_future_satisfied}"
    )

    print(
        "No-clear outlier remains fail-closed     "
        f"{no_clear.status == 'needs_information'}"
    )

    print(
        "Explicit linear preserved                "
        f"{explicit_linear.selected_test == 'pearson'}"
    )

    print(
        "Explicit monotonic preserved             "
        f"{explicit_monotonic.selected_test == 'spearman'}"
    )

    print(
        "Confirmatory general fail-closed         "
        f"{confirmatory_general.status == 'needs_information'}"
    )

    print(
        "Unknown independence fail-closed         "
        f"{unknown_independence.status == 'needs_information'}"
    )

    print(
        "Repeated observations fail-closed        "
        f"{repeated_observations.status == 'not_applicable'}"
    )

    print()

    print(
        f"Observed gaps                            {gaps}"
    )




    if gaps:

        assert (
            gaps
            ==
            [
                EXPECTED_RED_GAP
            ]
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
        "PASS - material-outlier monotonic "
        "rank recovery v0.1"
    )


if __name__ == "__main__":

    main()