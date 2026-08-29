from __future__ import annotations


import pandas as pd


from app.api.model_lab_service import (
    ModelLabServiceError,
    get_model_lab_model_detail,
)


from app.api.model_training_contracts import (
    ModelTrainingColumn,
    ModelTrainingContextResponse,
    ModelTrainingDataset,
    ModelTrainingRequest,
)


from app.ml.classical_executor import (
    ClassicalMLExecutorError,
    ClassicalMLInputError,
    ClassicalMLEstimatorError,
    execute_classical_ml,
)


from app.preparation.analysis_input_handoff import (
    AnalysisInputHandoffError,
    load_validated_analysis_input,
)


from app.preparation.analysis_readiness_gate import (
    AnalysisReadinessError,
)


# ============================================================
# VERSION
# ============================================================


MODEL_TRAINING_SERVICE_RULE_VERSION = (
    "model_training_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class ModelTrainingServiceError(
    RuntimeError
):
    pass


class ModelTrainingContextError(
    ModelTrainingServiceError
):
    pass


class ModelTrainingInputError(
    ModelTrainingServiceError
):
    pass


class ModelTrainingEstimatorError(
    ModelTrainingServiceError
):
    pass


class ModelTrainingExecutionError(
    ModelTrainingServiceError
):
    pass


# ============================================================
# IDENTIFIER
# ============================================================


def _required_identifier(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()

    if not normalized:
        raise ModelTrainingServiceError(
            f"{field_name} cannot be empty."
        )

    return normalized


# ============================================================
# COLUMN KIND
# ============================================================


def _column_kind(
    series: pd.Series,
) -> str:

    dtype = (
        series.dtype
    )

    if (
        pd.api.types
        .is_bool_dtype(
            dtype
        )
    ):
        return "boolean"

    if (
        pd.api.types
        .is_numeric_dtype(
            dtype
        )
    ):
        return "numeric"

    if (
        pd.api.types
        .is_datetime64_any_dtype(
            dtype
        )
    ):
        return "datetime"

    if (
        pd.api.types
        .is_categorical_dtype(
            dtype
        )
        or
        pd.api.types
        .is_string_dtype(
            dtype
        )
        or
        pd.api.types
        .is_object_dtype(
            dtype
        )
    ):
        return "categorical"

    return "other"


# ============================================================
# TRAINING CONTEXT
# ============================================================


def get_model_training_context(
    *,
    workflow_id: str,
) -> ModelTrainingContextResponse:

    normalized_workflow_id = (
        _required_identifier(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )

    try:
        handoff = (
            load_validated_analysis_input(
                workflow_id=
                    normalized_workflow_id
            )
        )

    except (
        AnalysisInputHandoffError,
        AnalysisReadinessError,
        ValueError,
    ) as error:
        raise ModelTrainingContextError(
            (
                "Model Training requires a valid "
                "READY Preparation analysis handoff."
            )
        ) from error

    datasets: list[
        ModelTrainingDataset
    ] = []

    for record in (
        handoff.dataset_records
    ):

        if not isinstance(
            record,
            dict,
        ):
            raise ModelTrainingContextError(
                (
                    "Validated analysis handoff "
                    "contains an invalid dataset record."
                )
            )

        dataset_id = (
            _required_identifier(
                record.get(
                    "dataset_id"
                ),
                field_name=
                    "dataset_id",
            )
        )

        filename = (
            _required_identifier(
                record.get(
                    "filename"
                ),
                field_name=
                    "filename",
            )
        )

        dataframe = (
            record.get(
                "dataframe"
            )
        )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise ModelTrainingContextError(
                (
                    "Validated analysis handoff "
                    "does not contain a DataFrame."
                )
            )

        raw_columns = list(
            dataframe.columns
        )

        if not raw_columns:
            raise ModelTrainingContextError(
                (
                    "Validated ML dataset "
                    "contains no columns."
                )
            )

        if not all(
            isinstance(
                column,
                str,
            )
            for column
            in raw_columns
        ):
            raise ModelTrainingContextError(
                (
                    "Model Training v0.1 requires "
                    "string column names."
                )
            )

        column_names = [
            column.strip()
            for column
            in raw_columns
        ]

        if any(
            not name
            for name
            in column_names
        ):
            raise ModelTrainingContextError(
                (
                    "Model Training v0.1 does not "
                    "support empty column names."
                )
            )

        if (
            len(
                column_names
            )
            !=
            len(
                set(
                    column_names
                )
            )
        ):
            raise ModelTrainingContextError(
                (
                    "Model Training v0.1 does not "
                    "support duplicate column names."
                )
            )

        columns: list[
            ModelTrainingColumn
        ] = []

        for (
            raw_column,
            column_name,
        ) in zip(
            raw_columns,
            column_names,
        ):

            series = (
                dataframe[
                    raw_column
                ]
            )

            columns.append(
                ModelTrainingColumn(
                    name=
                        column_name,

                    kind=
                        _column_kind(
                            series
                        ),

                    nullable=
                        bool(
                            series
                            .isna()
                            .any()
                        ),
                )
            )

        datasets.append(
            ModelTrainingDataset(
                dataset_id=
                    dataset_id,

                filename=
                    filename,

                row_count=
                    int(
                        len(
                            dataframe
                        )
                    ),

                column_count=
                    len(
                        columns
                    ),

                columns=
                    columns,
            )
        )

    if not datasets:
        raise ModelTrainingContextError(
            (
                "Validated Preparation handoff "
                "contains no trainable datasets."
            )
        )

    return (
        ModelTrainingContextResponse(
            workflow_id=
                handoff.workflow_id,

            preparation_session_revision=
                int(
                    handoff
                    .session_revision
                ),

            dataset_count=
                len(
                    datasets
                ),

            datasets=
                datasets,
        )
    )


# ============================================================
# TRAIN
# ============================================================


def train_model(
    request: ModelTrainingRequest,
):
    try:
        config = (
            ModelTrainingRequest
            .model_validate(
                request
            )
        )

    except Exception as error:
        raise ModelTrainingInputError(
            (
                "Model Training request "
                "is invalid."
            )
        ) from error

    try:
        result = (
            execute_classical_ml(
                training_contract=
                    config.training,

                expected_preparation_session_revision=(
                    config
                    .expected_preparation_session_revision
                ),
            )
        )

    except ClassicalMLInputError as error:
        raise ModelTrainingInputError(
            (
                "Model Training input is not "
                "compatible with the validated "
                "Preparation output."
            )
        ) from error

    except ClassicalMLEstimatorError as error:
        raise ModelTrainingEstimatorError(
            (
                "Model Training estimator "
                "configuration is invalid."
            )
        ) from error

    except ClassicalMLExecutorError as error:
        raise ModelTrainingExecutionError(
            (
                "Classical ML training "
                "could not be completed."
            )
        ) from error

    try:
        return (
            get_model_lab_model_detail(
                workflow_id=
                    config
                    .training
                    .workflow_id,

                model_id=
                    result
                    .model_artifact
                    .model_id,
            )
        )

    except ModelLabServiceError as error:
        raise ModelTrainingExecutionError(
            (
                "Training completed but the "
                "persisted Model Artifact could "
                "not be restored safely."
            )
        ) from error
