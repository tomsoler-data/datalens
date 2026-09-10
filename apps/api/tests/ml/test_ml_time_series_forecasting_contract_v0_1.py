from __future__ import annotations


import ast
from pathlib import Path


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLSplitContract,
    MLTimeHoldoutSplitContract,
)


from app.ml.time_series_contracts import (
    ML_TIME_SERIES_FORECASTING_CONTRACT_RULE_VERSION,
    MLTimeSeriesForecastingContract,
)


# ============================================================
# TEST HELPERS
# ============================================================


def require_validation_error(
    callback,
    *,
    label: str,
) -> None:

    try:
        callback()

    except ValidationError:
        return


    raise AssertionError(
        (
            "Expected contract validation failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. VALID UNIVARIATE FORECAST
# ============================================================


contract = MLTimeSeriesForecastingContract(
    workflow_id=
        "workflow:time-series-contract",
    dataset_id=
        "dataset:monthly-revenue",
    target_column=
        "revenue",
    lookback=
        12,
    split=
        MLTimeHoldoutSplitContract(
            time_column=
                "order_month",
            test_size=
                0.20,
        ),
)


assert (
    contract.task_family
    ==
    "time_series_forecasting"
)


assert (
    contract.problem_type
    ==
    "regression"
)


assert (
    contract.target_column
    ==
    "revenue"
)


assert (
    contract.time_column
    ==
    "order_month"
)


assert (
    contract.lookback
    ==
    12
)


assert (
    contract.forecast_horizon
    ==
    1
)


assert (
    contract.split.strategy
    ==
    "time_holdout"
)


assert (
    contract.split.shuffle
    is False
)


assert (
    contract.split.stratify
    is False
)


assert (
    contract.rule_version
    ==
    ML_TIME_SERIES_FORECASTING_CONTRACT_RULE_VERSION
)


print(
    "[PASS] valid univariate one-step forecasting contract"
)


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================


normalized = MLTimeSeriesForecastingContract(
    workflow_id=
        "  workflow:normalized  ",
    dataset_id=
        "  dataset:normalized  ",
    target_column=
        "  revenue  ",
    lookback=
        6,
    split=
        MLTimeHoldoutSplitContract(
            time_column=
                "order_date",
        ),
)


assert (
    normalized.workflow_id
    ==
    "workflow:normalized"
)


assert (
    normalized.dataset_id
    ==
    "dataset:normalized"
)


assert (
    normalized.target_column
    ==
    "revenue"
)


print(
    "[PASS] required text fields normalize deterministically"
)


# ============================================================
# 3. LOOKBACK MUST BE EXPLICIT AND CONTROLLED
# ============================================================


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:missing-lookback",
            dataset_id=
                "dataset:missing-lookback",
            target_column=
                "revenue",
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "missing lookback",
)


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:zero-lookback",
            dataset_id=
                "dataset:zero-lookback",
            target_column=
                "revenue",
            lookback=
                0,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "lookback=0",
)


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:boolean-lookback",
            dataset_id=
                "dataset:boolean-lookback",
            target_column=
                "revenue",
            lookback=
                True,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "boolean lookback",
)


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:huge-lookback",
            dataset_id=
                "dataset:huge-lookback",
            target_column=
                "revenue",
            lookback=
                4097,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "lookback above server limit",
)


print(
    "[PASS] lookback is explicit bounded strict integer"
)


# ============================================================
# 4. V0.1 IS ONE-STEP AHEAD ONLY
# ============================================================


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:multi-horizon",
            dataset_id=
                "dataset:multi-horizon",
            target_column=
                "revenue",
            lookback=
                12,
            forecast_horizon=
                2,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "forecast_horizon=2",
)


print(
    "[PASS] forecast horizon is frozen to one step"
)


# ============================================================
# 5. ONLY CHRONOLOGICAL HOLDOUT IS ACCEPTED
# ============================================================


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:random-holdout",
            dataset_id=
                "dataset:random-holdout",
            target_column=
                "revenue",
            lookback=
                12,
            split=
                MLSplitContract(
                    test_size=
                        0.20,
                    shuffle=
                        True,
                ),
        ),
    label=
        "random row holdout",
)


print(
    "[PASS] forecasting requires chronological time holdout"
)


# ============================================================
# 6. TARGET AND TIME AXIS MUST BE DISTINCT
# ============================================================


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:role-collision",
            dataset_id=
                "dataset:role-collision",
            target_column=
                "order_date",
            lookback=
                12,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "target/time role collision",
)


print(
    "[PASS] target and temporal axis roles are distinct"
)


# ============================================================
# 7. TASK / PROBLEM IDENTITY IS FROZEN
# ============================================================


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:wrong-task",
            dataset_id=
                "dataset:wrong-task",
            task_family=
                "tabular_regression",
            target_column=
                "revenue",
            lookback=
                12,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "wrong task family",
)


require_validation_error(
    lambda:
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:wrong-problem",
            dataset_id=
                "dataset:wrong-problem",
            problem_type=
                "classification",
            target_column=
                "revenue",
            lookback=
                12,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                ),
        ),
    label=
        "non-regression problem type",
)


print(
    "[PASS] forecasting task and regression identity are frozen"
)


# ============================================================
# 8. USER CANNOT INJECT EXECUTION STATE
# ============================================================


for forbidden_field in (
    "seed",
    "random_seed",
    "device",
    "optimizer",
    "loss",
    "raw_rows",
    "predictions",
    "model_state",
):

    payload = {
        "workflow_id":
            "workflow:forbidden-field",

        "dataset_id":
            "dataset:forbidden-field",

        "target_column":
            "revenue",

        "lookback":
            12,

        "split":
            {
                "strategy":
                    "time_holdout",

                "time_column":
                    "order_date",

                "test_size":
                    0.20,

                "random_seed":
                    42,

                "shuffle":
                    False,

                "stratify":
                    False,
            },

        forbidden_field:
            "forbidden",
    }


    require_validation_error(
        lambda payload=payload:
            MLTimeSeriesForecastingContract
            .model_validate(
                payload
            ),
        label=
            (
                "forbidden field "
                f"{forbidden_field}"
            ),
    )


print(
    "[PASS] execution and learned state cannot enter task contract"
)


# ============================================================
# 9. SERIALIZED SURFACE IS PRIVACY-MINIMAL
# ============================================================


serialized = (
    contract.model_dump(
        mode="json"
    )
)


assert (
    set(
        serialized
    )
    ==
    {
        "workflow_id",
        "dataset_id",
        "task_family",
        "problem_type",
        "target_column",
        "lookback",
        "forecast_horizon",
        "split",
        "rule_version",
    }
)


assert (
    "feature_columns"
    not in serialized
)


assert (
    "categorical_feature_columns"
    not in serialized
)


assert (
    "estimator_key"
    not in serialized
)


print(
    "[PASS] v0.1 task contract remains univariate and model-neutral"
)


# ============================================================
# 10. STATIC TORCH-FREE CONTRACT SURFACE
# ============================================================


source_path = Path(
    "app/ml/time_series_contracts.py"
)


source = source_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source
)


imports = []


for node in ast.walk(
    tree
):

    if isinstance(
        node,
        ast.Import,
    ):

        imports.extend(
            alias.name
            for alias in node.names
        )


    elif isinstance(
        node,
        ast.ImportFrom,
    ):

        if node.module:

            imports.append(
                node.module
            )


assert not any(
    name
    ==
    "torch"
    or
    name.startswith(
        "torch."
    )
    for name in imports
)


assert (
    "app.deep_learning"
    not in imports
)


print(
    "[PASS] forecasting task contract is torch-free"
)


print()
print("=" * 80)
print("DL-5-A1-V1 FINAL VERDICT")
print("=" * 80)
print()

print(
    "Univariate forecasting task contract      PASS"
)

print(
    "Regression identity                       PASS"
)

print(
    "Chronological split authority             PASS"
)

print(
    "Explicit lookback                         PASS"
)

print(
    "One-step horizon                          PASS"
)

print(
    "Target/time role isolation                PASS"
)

print(
    "No execution-state injection              PASS"
)

print(
    "Model-neutral contract                    PASS"
)

print(
    "Torch-free boundary                       PASS"
)

print()
print(
    "DL-5-A1-V1 - FORECASTING TASK CONTRACT: PASS"
)
