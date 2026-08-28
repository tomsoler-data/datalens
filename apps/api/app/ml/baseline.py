from __future__ import annotations


import math


from dataclasses import (
    dataclass,
)


from typing import (
    Literal,
)


import numpy as np
import pandas as pd


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.ml.contracts import (
    MLProblemType,
)


# ============================================================
# VERSION
# ============================================================


ML_BASELINE_RULE_VERSION = (
    "ml_baseline_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLBaselineStrategy = Literal[
    "mean_train_target",
    "majority_train_class",
]


MLBaselinePrimaryMetric = Literal[
    "rmse",
    "f1_macro",
]


# ============================================================
# ERRORS
# ============================================================


class MLBaselineError(
    RuntimeError
):
    pass


# ============================================================
# INTERNAL PREDICTION BUNDLE
# ============================================================


@dataclass(
    frozen=True,
)
class MLBaselinePredictionBundle:
    """
    Runtime-only baseline predictions.

    Predictions are deliberately not part of the public
    persisted result contract.
    """

    strategy: MLBaselineStrategy

    predictions: np.ndarray


# ============================================================
# PUBLIC BASELINE EVALUATION
# ============================================================


class MLBaselineEvaluationResult(
    BaseModel
):
    """
    Privacy-minimal baseline evaluation.

    The learned baseline constant / majority class and raw
    predictions are deliberately not exposed.

    Only strategy, metrics and row counts cross the boundary.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    problem_type: MLProblemType


    strategy: MLBaselineStrategy


    primary_metric: MLBaselinePrimaryMetric


    train_rows: int = Field(
        gt=0,
    )


    test_rows: int = Field(
        gt=0,
    )


    metrics: dict[
        str,
        float,
    ]


    rule_version: Literal[
        "ml_baseline_v0.1"
    ] = ML_BASELINE_RULE_VERSION


    @model_validator(
        mode="after"
    )
    def validate_consistency(
        self,
    ) -> "MLBaselineEvaluationResult":

        if (
            self.problem_type
            ==
            "regression"
        ):
            expected_strategy = (
                "mean_train_target"
            )

            expected_primary_metric = (
                "rmse"
            )

            expected_metric_names = {
                "mae",
                "rmse",
                "r2",
            }

        else:
            expected_strategy = (
                "majority_train_class"
            )

            expected_primary_metric = (
                "f1_macro"
            )

            expected_metric_names = {
                "accuracy",
                "f1_macro",
            }


        if (
            self.strategy
            !=
            expected_strategy
        ):
            raise ValueError(
                (
                    "Baseline strategy does not match "
                    "problem type."
                )
            )


        if (
            self.primary_metric
            !=
            expected_primary_metric
        ):
            raise ValueError(
                (
                    "Baseline primary metric does not "
                    "match problem type."
                )
            )


        if (
            set(
                self.metrics
            )
            !=
            expected_metric_names
        ):
            raise ValueError(
                (
                    "Baseline metric set does not match "
                    "problem type."
                )
            )


        for (
            metric_name,
            raw_value,
        ) in self.metrics.items():

            value = float(
                raw_value
            )


            if not (
                math.isfinite(
                    value
                )
            ):
                raise ValueError(
                    (
                        "Baseline metric must be finite. "
                        f"metric={metric_name}"
                    )
                )


        return self


# ============================================================
# MODEL VS BASELINE
# ============================================================


class MLBaselineComparisonResult(
    BaseModel
):
    """
    Positive absolute_improvement always means the real model
    performs better than the baseline.

    Regression:
        baseline RMSE - model RMSE

    Classification:
        model F1 macro - baseline F1 macro

    relative_improvement_pct is omitted when the baseline
    primary metric is zero, because a finite relative percentage
    is not mathematically defined in that case.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    problem_type: MLProblemType


    primary_metric: MLBaselinePrimaryMetric


    model_primary_metric_value: float


    baseline_primary_metric_value: float


    absolute_improvement: float


    relative_improvement_pct: (
        float
        |
        None
    )


    beats_baseline: bool


    rule_version: Literal[
        "ml_baseline_v0.1"
    ] = ML_BASELINE_RULE_VERSION


    @model_validator(
        mode="after"
    )
    def validate_consistency(
        self,
    ) -> "MLBaselineComparisonResult":

        if (
            self.problem_type
            ==
            "regression"
        ):
            expected_primary_metric = (
                "rmse"
            )

            expected_improvement = (
                self.baseline_primary_metric_value
                -
                self.model_primary_metric_value
            )

        else:
            expected_primary_metric = (
                "f1_macro"
            )

            expected_improvement = (
                self.model_primary_metric_value
                -
                self.baseline_primary_metric_value
            )


        if (
            self.primary_metric
            !=
            expected_primary_metric
        ):
            raise ValueError(
                (
                    "Baseline comparison primary metric "
                    "does not match problem type."
                )
            )


        finite_values = [
            self.model_primary_metric_value,
            self.baseline_primary_metric_value,
            self.absolute_improvement,
        ]


        if (
            self.relative_improvement_pct
            is not None
        ):
            finite_values.append(
                self.relative_improvement_pct
            )


        if not all(
            math.isfinite(
                float(
                    value
                )
            )

            for value
            in finite_values
        ):
            raise ValueError(
                (
                    "Baseline comparison values must "
                    "be finite."
                )
            )


        if not math.isclose(
            self.absolute_improvement,
            expected_improvement,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                (
                    "Baseline absolute improvement is "
                    "inconsistent with metric values."
                )
            )


        expected_beats_baseline = (
            expected_improvement
            >
            0.0
        )


        if (
            self.beats_baseline
            !=
            expected_beats_baseline
        ):
            raise ValueError(
                (
                    "beats_baseline is inconsistent "
                    "with primary metric values."
                )
            )


        denominator = abs(
            self.baseline_primary_metric_value
        )


        if (
            denominator
            <=
            1e-15
        ):
            if (
                self.relative_improvement_pct
                is not None
            ):
                raise ValueError(
                    (
                        "Relative improvement must be "
                        "None when baseline primary "
                        "metric is zero."
                    )
                )

        else:
            expected_relative = (
                (
                    expected_improvement
                    /
                    denominator
                )
                *
                100.0
            )


            if (
                self.relative_improvement_pct
                is None
                or
                not math.isclose(
                    self.relative_improvement_pct,
                    expected_relative,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    (
                        "Relative baseline improvement "
                        "is inconsistent with metric "
                        "values."
                    )
                )


        return self


# ============================================================
# POLICY
# ============================================================


def baseline_strategy_for_problem(
    problem_type: MLProblemType,
) -> MLBaselineStrategy:

    if (
        problem_type
        ==
        "regression"
    ):
        return (
            "mean_train_target"
        )


    if (
        problem_type
        ==
        "classification"
    ):
        return (
            "majority_train_class"
        )


    raise MLBaselineError(
        (
            "Unsupported ML problem type for baseline. "
            f"problem_type={problem_type}"
        )
    )


def baseline_primary_metric_for_problem(
    problem_type: MLProblemType,
) -> MLBaselinePrimaryMetric:

    if (
        problem_type
        ==
        "regression"
    ):
        return (
            "rmse"
        )


    if (
        problem_type
        ==
        "classification"
    ):
        return (
            "f1_macro"
        )


    raise MLBaselineError(
        (
            "Unsupported ML problem type for baseline. "
            f"problem_type={problem_type}"
        )
    )


# ============================================================
# BASELINE PREDICTIONS
# ============================================================


def build_ml_baseline_predictions(
    *,
    problem_type: MLProblemType,
    y_train: pd.Series,
    test_rows: int,
) -> MLBaselinePredictionBundle:
    """
    Learn a deterministic baseline exclusively from y_train.

    y_test is intentionally NOT accepted by this function.

    This makes target leakage through baseline construction
    structurally impossible.

    Regression:
        predict mean(y_train)

    Classification:
        predict majority class from y_train

    Majority-class ties are resolved by first appearance in the
    deterministic training split.
    """

    if not isinstance(
        y_train,
        pd.Series,
    ):
        raise MLBaselineError(
            (
                "Baseline y_train must be a pandas "
                "Series."
            )
        )


    if (
        len(
            y_train
        )
        <
        1
    ):
        raise MLBaselineError(
            (
                "Baseline training target cannot "
                "be empty."
            )
        )


    if (
        int(
            test_rows
        )
        <
        1
    ):
        raise MLBaselineError(
            (
                "Baseline test_rows must be positive."
            )
        )


    if bool(
        y_train
        .isna()
        .any()
    ):
        raise MLBaselineError(
            (
                "Baseline training target contains "
                "missing values."
            )
        )


    # ========================================================
    # REGRESSION
    # ========================================================

    if (
        problem_type
        ==
        "regression"
    ):

        if (
            pd.api.types
            .is_bool_dtype(
                y_train.dtype
            )
            or
            not pd.api.types
            .is_numeric_dtype(
                y_train.dtype
            )
        ):
            raise MLBaselineError(
                (
                    "Regression baseline requires a "
                    "numeric non-boolean training target."
                )
            )


        try:
            numeric_target = (
                y_train
                .to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

        except Exception as error:
            raise (
                MLBaselineError(
                    (
                        "Regression baseline target "
                        "could not be converted to "
                        "float64."
                    )
                )
            ) from error


        if not (
            np.isfinite(
                numeric_target
            )
            .all()
        ):
            raise MLBaselineError(
                (
                    "Regression baseline training "
                    "target contains non-finite values."
                )
            )


        mean_target = float(
            np.mean(
                numeric_target
            )
        )


        if not math.isfinite(
            mean_target
        ):
            raise MLBaselineError(
                (
                    "Regression baseline mean is "
                    "non-finite."
                )
            )


        predictions = np.full(
            shape=(
                int(
                    test_rows
                ),
            ),
            fill_value=
                mean_target,
            dtype=np.float64,
        )


        return (
            MLBaselinePredictionBundle(
                strategy=
                    "mean_train_target",

                predictions=
                    predictions,
            )
        )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if (
        problem_type
        ==
        "classification"
    ):

        try:
            (
                codes,
                unique_values,
            ) = (
                pd.factorize(
                    y_train,
                    sort=False,
                )
            )

        except Exception as error:
            raise (
                MLBaselineError(
                    (
                        "Classification baseline could "
                        "not factorize training labels."
                    )
                )
            ) from error


        if (
            len(
                unique_values
            )
            <
            1
            or
            bool(
                (
                    codes
                    <
                    0
                )
                .any()
            )
        ):
            raise MLBaselineError(
                (
                    "Classification baseline requires "
                    "valid non-missing training labels."
                )
            )


        counts = (
            np.bincount(
                codes
            )
        )


        max_count = int(
            counts.max()
        )


        tied_codes = (
            np.flatnonzero(
                counts
                ==
                max_count
            )
        )


        majority_code = int(
            tied_codes[
                0
            ]
        )


        majority_value = (
            unique_values[
                majority_code
            ]
        )


        predictions = np.repeat(
            np.asarray(
                [
                    majority_value
                ]
            ),
            int(
                test_rows
            ),
        )


        return (
            MLBaselinePredictionBundle(
                strategy=
                    "majority_train_class",

                predictions=
                    predictions,
            )
        )


    raise MLBaselineError(
        (
            "Unsupported ML problem type for baseline. "
            f"problem_type={problem_type}"
        )
    )


# ============================================================
# PUBLIC RESULT BUILDERS
# ============================================================


def build_ml_baseline_evaluation(
    *,
    problem_type: MLProblemType,
    strategy: MLBaselineStrategy,
    metrics: dict[
        str,
        float,
    ],
    train_rows: int,
    test_rows: int,
) -> MLBaselineEvaluationResult:

    try:
        return (
            MLBaselineEvaluationResult(
                problem_type=
                    problem_type,

                strategy=
                    strategy,

                primary_metric=(
                    baseline_primary_metric_for_problem(
                        problem_type
                    )
                ),

                train_rows=
                    int(
                        train_rows
                    ),

                test_rows=
                    int(
                        test_rows
                    ),

                metrics={
                    str(
                        metric_name
                    ):
                        float(
                            metric_value
                        )

                    for (
                        metric_name,
                        metric_value,
                    )
                    in metrics.items()
                },

                rule_version=
                    ML_BASELINE_RULE_VERSION,
            )
        )

    except Exception as error:
        raise (
            MLBaselineError(
                (
                    "Baseline evaluation result could "
                    "not be constructed."
                )
            )
        ) from error


def compare_model_to_baseline(
    *,
    problem_type: MLProblemType,
    model_metrics: dict[
        str,
        float,
    ],
    baseline_metrics: dict[
        str,
        float,
    ],
) -> MLBaselineComparisonResult:

    primary_metric = (
        baseline_primary_metric_for_problem(
            problem_type
        )
    )


    try:
        model_value = float(
            model_metrics[
                primary_metric
            ]
        )

        baseline_value = float(
            baseline_metrics[
                primary_metric
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise (
            MLBaselineError(
                (
                    "Model/baseline primary metric "
                    "is missing or invalid."
                )
            )
        ) from error


    if not (
        math.isfinite(
            model_value
        )
        and
        math.isfinite(
            baseline_value
        )
    ):
        raise MLBaselineError(
            (
                "Model/baseline primary metric "
                "must be finite."
            )
        )


    if (
        problem_type
        ==
        "regression"
    ):
        absolute_improvement = (
            baseline_value
            -
            model_value
        )

    else:
        absolute_improvement = (
            model_value
            -
            baseline_value
        )


    denominator = abs(
        baseline_value
    )


    relative_improvement_pct = (
        None

        if (
            denominator
            <=
            1e-15
        )

        else (
            (
                absolute_improvement
                /
                denominator
            )
            *
            100.0
        )
    )


    try:
        return (
            MLBaselineComparisonResult(
                problem_type=
                    problem_type,

                primary_metric=
                    primary_metric,

                model_primary_metric_value=
                    model_value,

                baseline_primary_metric_value=
                    baseline_value,

                absolute_improvement=
                    float(
                        absolute_improvement
                    ),

                relative_improvement_pct=(
                    None

                    if (
                        relative_improvement_pct
                        is None
                    )

                    else float(
                        relative_improvement_pct
                    )
                ),

                beats_baseline=(
                    absolute_improvement
                    >
                    0.0
                ),

                rule_version=
                    ML_BASELINE_RULE_VERSION,
            )
        )

    except Exception as error:
        raise (
            MLBaselineError(
                (
                    "Model-to-baseline comparison "
                    "could not be constructed."
                )
            )
        ) from error
