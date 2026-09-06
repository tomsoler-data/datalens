from __future__ import annotations


import math


import numpy as np
import pandas as pd


from scipy.stats import (
    spearmanr,
)


from app.statistics.decision import (
    decide_correlation_test,
)


from app.statistics.diagnostics import (
    MINIMUM_ASSOCIATION_SIGNAL,
    determine_shape_signal,
)


EXPECTED_THRESHOLD = 0.20


EXPECTED_RED_GAP = "general_association_no_clear_pattern_fallback_gap"


RANDOM_SEED = 20260904


# ============================================================
# SYNTHETIC CONTROLS
# ============================================================


def weak_detectable_frame() -> tuple[
    pd.DataFrame,
    float,
    float,
]:
    """
    Find a deterministic weak association where:

    - abs(Spearman rho) is below the current diagnostic threshold;
    - p < .05;
    - current exploratory shape = no_clear_pattern.

    No application-specific data or evaluator values are used.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    n = 12000


    x = np.linspace(
        -1.0,
        1.0,
        n,
        dtype=float,
    )


    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )


    for slope in (
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
        0.22,
        0.24,
    ):

        y = (
            slope
            *
            x
            +
            noise
        )


        frame = pd.DataFrame(
            {
                "x": x,
                "y": y,
            }
        )


        result = spearmanr(
            frame["x"],
            frame["y"],
            alternative="two-sided",
        )


        rho = float(
            result.statistic
        )


        p_value = float(
            result.pvalue
        )


        decision = decide_correlation_test(
            dataframe=frame,
            x_column="x",
            y_column="y",
            analysis_goal="general_association",
            analysis_mode="exploratory",
            x_kind="continuous",
            y_kind="continuous",
            observations_independent=True,
        )


        if (
            abs(rho) < MINIMUM_ASSOCIATION_SIGNAL
            and p_value < 0.05
            and decision.diagnostics.shape_signal == "no_clear_pattern"
        ):

            return (
                frame,
                rho,
                p_value,
            )


    raise RuntimeError(
        "Unable to build deterministic weak-detectable control."
    )


def null_like_frame() -> pd.DataFrame:

    rng = np.random.default_rng(
        314159
    )


    return pd.DataFrame(
        {
            "x": rng.normal(
                size=5000
            ),
            "y": rng.normal(
                size=5000
            ),
        }
    )


def strong_monotonic_frame() -> pd.DataFrame:
    """
    Strong monotonic/non-linear control without a material
    IQR-outlier signal.

    x + 1.01 is strictly positive over [-1, 1], therefore
    squaring preserves strict monotonic increase.

    The quadratic curvature is intentional: this exercises the
    existing monotonic_non_linear_candidate path without letting
    the separate material-outlier guard intercept the case.
    """

    x = np.linspace(
        -1.0,
        1.0,
        1000,
    )


    y = (
        x
        +
        1.01
    ) ** 2


    return pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    )


# ============================================================
# FUTURE CONTRACT
# ============================================================


def is_no_clear_pattern_spearman_fallback(
    decision,
) -> bool:
    """
    Future generic behavior.

    For:
      - exploratory
      - general association
      - continuous x/y
      - independent rows
      - no_clear_pattern

    DataLens should still select a test.

    Spearman is used as the conservative general-association
    fallback, without claiming that the observed data proved a
    monotonic relationship.

    Because this fallback is reached only after observed
    relationship diagnostics classify the shape as
    `no_clear_pattern`, this remains an exploratory
    data-driven routing decision.

    Therefore selection_is_data_driven must remain True.

    The fallback POLICY itself is deterministic:
    no-clear-pattern -> Spearman.

    This does not mean that the observed data established a
    monotonic functional form.
    """

    return bool(
        decision.status == "selected"
        and decision.selected_test == "spearman"
        and decision.diagnostics.shape_signal == "no_clear_pattern"
        and decision.selection_is_data_driven is True
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS GENERAL ASSOCIATION "
        "NO-CLEAR-PATTERN FALLBACK v0.1 ==="
    )

    print()


    # ========================================================
    # 1. DIAGNOSTIC THRESHOLD MUST NOT CHANGE
    # ========================================================

    assert math.isclose(
        MINIMUM_ASSOCIATION_SIGNAL,
        EXPECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    )


    below = determine_shape_signal(
        n_valid=10000,
        pearson_coefficient=0.05,
        spearman_coefficient=0.19,
        linear_r_squared=0.0025,
        quadratic_gain=0.0,
    )


    above = determine_shape_signal(
        n_valid=10000,
        pearson_coefficient=0.05,
        spearman_coefficient=0.21,
        linear_r_squared=0.0025,
        quadratic_gain=0.0,
    )


    assert (
        below
        ==
        "no_clear_pattern"
    )


    assert (
        above
        ==
        "monotonic_non_linear_candidate"
    )


    print(
        "[PASS] diagnostic threshold remains 0.20"
    )

    print(
        "[PASS] shape-signal diagnostic boundary preserved"
    )


    # ========================================================
    # 2. WEAK BUT DETECTABLE ASSOCIATION
    # ========================================================

    (
        weak_frame,
        weak_rho,
        weak_p,
    ) = weak_detectable_frame()


    weak_decision = decide_correlation_test(
        dataframe=weak_frame,
        x_column="x",
        y_column="y",
        analysis_goal="general_association",
        analysis_mode="exploratory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=True,
    )


    weak_fallback = (
        is_no_clear_pattern_spearman_fallback(
            weak_decision
        )
    )


    print()
    print(
        "Weak detectable association"
    )

    print(
        "  rho                                    "
        f"{weak_rho}"
    )

    print(
        "  p-value                                "
        f"{weak_p}"
    )

    print(
        "  below 0.20                             "
        f"{abs(weak_rho) < MINIMUM_ASSOCIATION_SIGNAL}"
    )

    print(
        "  shape signal                           "
        f"{weak_decision.diagnostics.shape_signal}"
    )

    print(
        "  status                                 "
        f"{weak_decision.status}"
    )

    print(
        "  selected test                          "
        f"{weak_decision.selected_test}"
    )

    print(
        "  selection is data-driven               "
        f"{weak_decision.selection_is_data_driven}"
    )

    print(
        "  future fallback satisfied              "
        f"{weak_fallback}"
    )


    # ========================================================
    # 3. NULL-LIKE EFFECT
    #
    # Anti-circularity control:
    # the ability to run an association test cannot require
    # observing a sufficiently large association first.
    # ========================================================

    null_frame = null_like_frame()


    null_rho_result = spearmanr(
        null_frame["x"],
        null_frame["y"],
        alternative="two-sided",
    )


    null_rho = float(
        null_rho_result.statistic
    )


    null_decision = decide_correlation_test(
        dataframe=null_frame,
        x_column="x",
        y_column="y",
        analysis_goal="general_association",
        analysis_mode="exploratory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=True,
    )


    assert (
        null_decision.diagnostics.shape_signal
        ==
        "no_clear_pattern"
    )


    null_fallback = (
        is_no_clear_pattern_spearman_fallback(
            null_decision
        )
    )


    print()
    print(
        "Null-like observed association"
    )

    print(
        "  Spearman rho                          "
        f"{null_rho}"
    )

    print(
        "  shape signal                          "
        f"{null_decision.diagnostics.shape_signal}"
    )

    print(
        "  status                                "
        f"{null_decision.status}"
    )

    print(
        "  selected test                         "
        f"{null_decision.selected_test}"
    )

    print(
        "  selection is data-driven              "
        f"{null_decision.selection_is_data_driven}"
    )

    print(
        "  future fallback satisfied             "
        f"{null_fallback}"
    )


    # ========================================================
    # 4. EXPLICIT LINEAR GOAL
    # ========================================================

    explicit_linear = decide_correlation_test(
        dataframe=weak_frame,
        x_column="x",
        y_column="y",
        analysis_goal="linear_association",
        analysis_mode="exploratory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=True,
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
        explicit_linear.selection_is_data_driven
        is False
    )


    print()
    print(
        "[PASS] explicit linear goal remains Pearson"
    )


    # ========================================================
    # 5. EXPLICIT MONOTONIC GOAL
    # ========================================================

    explicit_monotonic = decide_correlation_test(
        dataframe=weak_frame,
        x_column="x",
        y_column="y",
        analysis_goal="monotonic_association",
        analysis_mode="exploratory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=True,
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
        explicit_monotonic.selection_is_data_driven
        is False
    )


    print(
        "[PASS] explicit monotonic goal remains Spearman"
    )


    # ========================================================
    # 6. CONFIRMATORY GENERAL GOAL
    # ========================================================

    confirmatory_general = decide_correlation_test(
        dataframe=weak_frame,
        x_column="x",
        y_column="y",
        analysis_goal="general_association",
        analysis_mode="confirmatory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=True,
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
    # 7. UNKNOWN OBSERVATION INDEPENDENCE
    # ========================================================

    unknown_independence = decide_correlation_test(
        dataframe=weak_frame,
        x_column="x",
        y_column="y",
        analysis_goal="general_association",
        analysis_mode="exploratory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=None,
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
    # 8. REPEATED OBSERVATIONS
    # ========================================================

    repeated = decide_correlation_test(
        dataframe=weak_frame,
        x_column="x",
        y_column="y",
        analysis_goal="general_association",
        analysis_mode="exploratory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=False,
    )


    assert (
        repeated.status
        ==
        "not_applicable"
    )


    assert (
        repeated.selected_test
        is None
    )


    print(
        "[PASS] repeated-observation simple correlation remains fail-closed"
    )


    # ========================================================
    # 9. EXISTING STRONG MONOTONIC PATH
    # ========================================================

    strong_frame = strong_monotonic_frame()


    strong = decide_correlation_test(
        dataframe=strong_frame,
        x_column="x",
        y_column="y",
        analysis_goal="general_association",
        analysis_mode="exploratory",
        x_kind="continuous",
        y_kind="continuous",
        observations_independent=True,
    )


    assert (
        strong.status
        ==
        "selected"
    )


    assert (
        strong.selected_test
        ==
        "spearman"
    )


    assert (
        strong.selection_is_data_driven
        is True
    )


    assert (
        strong.diagnostics.shape_signal
        ==
        "monotonic_non_linear_candidate"
    )


    print(
        "[PASS] diagnosed monotonic non-linear branch remains data-driven"
    )


    # ========================================================
    # 10. RED MATRIX
    # ========================================================

    gaps: list[str] = []


    if not (
        weak_fallback
        and
        null_fallback
    ):

        gaps.append(
            EXPECTED_RED_GAP
        )


    print()
    print("=" * 80)
    print(
        "P5-R3 GENERAL ASSOCIATION FALLBACK MATRIX"
    )
    print("=" * 80)
    print()


    print(
        "Diagnostic threshold unchanged           True"
    )

    print(
        "Weak no-clear fallback                  "
        f"{weak_fallback}"
    )

    print(
        "Null-like no-clear fallback             "
        f"{null_fallback}"
    )

    print(
        "Explicit linear preserved               True"
    )

    print(
        "Explicit monotonic preserved            True"
    )

    print(
        "Confirmatory general fail-closed        True"
    )

    print(
        "Unknown independence fail-closed        True"
    )

    print(
        "Repeated observations fail-closed       True"
    )

    print(
        "Strong monotonic data-driven path       True"
    )

    print()

    print(
        "Observed gaps                            "
        f"{gaps}"
    )


    # ========================================================
    # RED NOW / GREEN AFTER IMPLEMENTATION
    # ========================================================

    if gaps:

        if gaps != [
            EXPECTED_RED_GAP
        ]:

            raise AssertionError(
                (
                    "Unexpected P5-R3 gap set: "
                    f"{gaps!r}"
                )
            )


        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                EXPECTED_RED_GAP
            )
        )


    print()

    print(
        "PASS - general association "
        "no-clear-pattern fallback v0.1"
    )

    print(
        "P5-R3 REGRESSION: GREEN"
    )


if __name__ == "__main__":

    main()