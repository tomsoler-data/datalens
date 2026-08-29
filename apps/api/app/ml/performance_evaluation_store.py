from __future__ import annotations


import json
import math


from typing import (
    Any,
)


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    ml_training_contract_sha256,
)


from app.ml.model_artifact_index import (
    ml_model_artifact_store_scope,
)


from app.ml.model_artifact_store import (
    resolve_ml_model_artifact_store_path,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# VERSION
# ============================================================


ML_PERFORMANCE_EVALUATION_STORE_RULE_VERSION = (
    "ml_performance_evaluation_store_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLPerformanceEvaluationStoreError(
    RuntimeError
):
    pass


class MLPerformanceEvaluationNotFoundError(
    MLPerformanceEvaluationStoreError
):
    pass


class MLPerformanceEvaluationAlreadyExistsError(
    MLPerformanceEvaluationStoreError
):
    pass


class MLPerformanceEvaluationAuthorityError(
    MLPerformanceEvaluationStoreError
):
    pass


class MLPerformanceEvaluationWorkflowMismatchError(
    MLPerformanceEvaluationStoreError
):
    pass


# ============================================================
# TEXT
# ============================================================


def _required_text(
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
        raise MLPerformanceEvaluationStoreError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# JSON
# ============================================================


def _canonical_json(
    value: object,
) -> str:

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

    except Exception as error:
        raise MLPerformanceEvaluationStoreError(
            (
                "ML Performance Evaluation "
                "cannot be serialized."
            )
        ) from error


def _decode_json_object(
    raw: object,
    *,
    field_name: str,
) -> dict[
    str,
    Any,
]:

    try:
        value = json.loads(
            str(
                raw
            )
        )

    except Exception as error:
        raise MLPerformanceEvaluationStoreError(
            (
                f"{field_name} contains "
                "invalid JSON."
            )
        ) from error


    if not isinstance(
        value,
        dict,
    ):
        raise MLPerformanceEvaluationStoreError(
            (
                f"{field_name} must decode "
                "to an object."
            )
        )


    return value


# ============================================================
# STORE SCOPE
# ============================================================


def _store_root(
) -> str:

    return (
        ml_model_artifact_store_scope(
            resolve_ml_model_artifact_store_path()
        )
    )


# ============================================================
# RECORD VALIDATION
# ============================================================


def _validated_evaluation(
    value: object,
) -> MLPerformanceEvaluationRecord:

    try:
        return (
            MLPerformanceEvaluationRecord
            .model_validate(
                value
            )
        )

    except ValidationError as error:
        raise MLPerformanceEvaluationStoreError(
            (
                "Invalid ML Performance "
                "Evaluation record."
            )
        ) from error


# ============================================================
# METRICS
# ============================================================


def _normalized_metrics(
    value: object,
) -> dict[
    str,
    float,
]:

    if not isinstance(
        value,
        dict,
    ):
        raise MLPerformanceEvaluationAuthorityError(
            (
                "Persisted Model Artifact "
                "metrics are invalid."
            )
        )


    if not value:
        raise MLPerformanceEvaluationAuthorityError(
            (
                "Persisted Model Artifact "
                "contains no metrics."
            )
        )


    normalized = {}


    for (
        raw_name,
        raw_value,
    ) in value.items():

        name = str(
            raw_name
        ).strip()


        if not name:
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Persisted Model Artifact "
                    "contains an empty metric name."
                )
            )


        if (
            isinstance(
                raw_value,
                bool,
            )
        ):
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Persisted Model Artifact "
                    "metric cannot be boolean."
                )
            )


        try:
            metric_value = float(
                raw_value
            )

        except Exception as error:
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Persisted Model Artifact "
                    "contains a non-numeric metric."
                )
            ) from error


        if not math.isfinite(
            metric_value
        ):
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Persisted Model Artifact "
                    "contains a non-finite metric."
                )
            )


        if name in normalized:
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Persisted Model Artifact "
                    "contains duplicate normalized "
                    "metric names."
                )
            )


        normalized[
            name
        ] = metric_value


    return normalized


# ============================================================
# DATABASE ROW
# ============================================================


def _row_to_record(
    row,
) -> MLPerformanceEvaluationRecord:

    payload = (
        _decode_json_object(
            row[
                "payload_json"
            ],
            field_name=
                "payload_json",
        )
    )


    record = (
        _validated_evaluation(
            payload
        )
    )


    expected = {
        "performance_evaluation_id":
            str(
                row[
                    "performance_evaluation_id"
                ]
            ),

        "model_id":
            str(
                row[
                    "model_id"
                ]
            ),

        "workflow_id":
            str(
                row[
                    "workflow_id"
                ]
            ),

        "reference_dataset_id":
            str(
                row[
                    "reference_dataset_id"
                ]
            ),

        "observed_dataset_id":
            str(
                row[
                    "observed_dataset_id"
                ]
            ),

        "experiment_id":
            str(
                row[
                    "experiment_id"
                ]
            ),

        "preparation_session_revision":
            int(
                row[
                    "preparation_session_revision"
                ]
            ),

        "observed_preparation_session_revision":
            int(
                row[
                    "observed_preparation_session_revision"
                ]
            ),

        "training_contract_sha256":
            str(
                row[
                    "training_contract_sha256"
                ]
            ),

        "problem_type":
            str(
                row[
                    "problem_type"
                ]
            ),

        "target_column":
            str(
                row[
                    "target_column"
                ]
            ),

        "reference_evaluation_row_count":
            int(
                row[
                    "reference_evaluation_row_count"
                ]
            ),

        "observed_row_count":
            int(
                row[
                    "observed_row_count"
                ]
            ),

        "evaluated_at_utc":
            str(
                row[
                    "evaluated_at_utc"
                ]
            ),

        "primary_metric":
            str(
                row[
                    "primary_metric"
                ]
            ),

        "primary_metric_degradation_amount":
            float(
                row[
                    "primary_metric_degradation_amount"
                ]
            ),

        "primary_metric_degradation_ratio":
            (
                float(
                    row[
                        "primary_metric_degradation_ratio"
                    ]
                )

                if (
                    row[
                        "primary_metric_degradation_ratio"
                    ]
                    is not None
                )

                else None
            ),

        "degradation_basis":
            str(
                row[
                    "degradation_basis"
                ]
            ),

        "performance_status":
            str(
                row[
                    "performance_status"
                ]
            ),

        "privacy_scope":
            str(
                row[
                    "privacy_scope"
                ]
            ),

        "rule_version":
            str(
                row[
                    "rule_version"
                ]
            ),
    }


    actual = {
        "performance_evaluation_id":
            record.performance_evaluation_id,

        "model_id":
            record.model_id,

        "workflow_id":
            record.workflow_id,

        "reference_dataset_id":
            record.reference_dataset_id,

        "observed_dataset_id":
            record.observed_dataset_id,

        "experiment_id":
            record.experiment_id,

        "preparation_session_revision":
            record.preparation_session_revision,

        "observed_preparation_session_revision":
            record.observed_preparation_session_revision,

        "training_contract_sha256":
            record.training_contract_sha256,

        "problem_type":
            record.problem_type,

        "target_column":
            record.target_column,

        "reference_evaluation_row_count":
            record.reference_evaluation_row_count,

        "observed_row_count":
            record.observed_row_count,

        "evaluated_at_utc":
            record.evaluated_at_utc,

        "primary_metric":
            record.primary_metric,

        "primary_metric_degradation_amount":
            record.primary_metric_degradation_amount,

        "primary_metric_degradation_ratio":
            record.primary_metric_degradation_ratio,

        "degradation_basis":
            record.degradation_basis,

        "performance_status":
            record.performance_status,

        "privacy_scope":
            record.privacy_scope,

        "rule_version":
            record.rule_version,
    }


    if actual != expected:
        raise MLPerformanceEvaluationStoreError(
            (
                "Stored ML Performance Evaluation "
                "index fields do not match "
                "payload_json."
            )
        )


    return record


# ============================================================
# MODEL ARTIFACT AUTHORITY
# ============================================================


def _assert_model_artifact_authority(
    *,
    connection,
    store_root: str,
    evaluation: MLPerformanceEvaluationRecord,
) -> None:
    """
    Bind one Performance Evaluation to the exact persisted
    Model Artifact + Experiment Provenance + holdout metrics.

    All checks occur inside the registration transaction.
    """

    row = (
        connection.execute(
            """
            SELECT *
            FROM ml_model_artifacts

            WHERE
                store_root = ?
                AND
                model_id = ?

            LIMIT 1
            """,
            (
                store_root,
                evaluation.model_id,
            ),
        )
        .fetchone()
    )


    if row is None:
        raise MLPerformanceEvaluationAuthorityError(
            (
                "ML Performance Evaluation "
                "references a Model Artifact "
                "that is not server-owned."
            )
        )


    # --------------------------------------------------------
    # TRAINING CONTRACT
    # --------------------------------------------------------


    training_contract_payload = (
        _decode_json_object(
            row[
                "training_contract_json"
            ],
            field_name=
                "model_artifact.training_contract_json",
        )
    )


    try:
        training_contract = (
            MLTrainingContract
            .model_validate(
                training_contract_payload
            )
        )

    except ValidationError as error:
        raise MLPerformanceEvaluationAuthorityError(
            (
                "Persisted Model Artifact "
                "Training Contract is invalid."
            )
        ) from error


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------


    metrics_payload = (
        _decode_json_object(
            row[
                "metrics_json"
            ],
            field_name=
                "model_artifact.metrics_json",
        )
    )


    metrics = (
        _normalized_metrics(
            metrics_payload
        )
    )


    # --------------------------------------------------------
    # EXPERIMENT PROVENANCE
    # --------------------------------------------------------


    raw_provenance = (
        row[
            "experiment_provenance_json"
        ]
    )


    if raw_provenance is None:
        raise MLPerformanceEvaluationAuthorityError(
            (
                "ML Performance Evaluation "
                "requires persisted Experiment "
                "Provenance."
            )
        )


    provenance_payload = (
        _decode_json_object(
            raw_provenance,
            field_name=(
                "model_artifact."
                "experiment_provenance_json"
            ),
        )
    )


    try:
        provenance = (
            MLExperimentProvenanceRecord
            .model_validate(
                provenance_payload
            )
        )

    except ValidationError as error:
        raise MLPerformanceEvaluationAuthorityError(
            (
                "Persisted Model Artifact "
                "Experiment Provenance is invalid."
            )
        ) from error


    # --------------------------------------------------------
    # INDEX ? CONTRACT / PROVENANCE
    # --------------------------------------------------------


    indexed_experiment_id = (
        str(
            row[
                "experiment_id"
            ]
        )
        if (
            row[
                "experiment_id"
            ]
            is not None
        )
        else None
    )


    indexed_bindings = [
        (
            "workflow_id",
            str(
                row[
                    "workflow_id"
                ]
            ),
            training_contract.workflow_id,
        ),
        (
            "dataset_id",
            str(
                row[
                    "dataset_id"
                ]
            ),
            training_contract.dataset_id,
        ),
        (
            "problem_type",
            str(
                row[
                    "problem_type"
                ]
            ),
            training_contract.problem_type,
        ),
        (
            "target_column",
            str(
                row[
                    "target_column"
                ]
            ),
            training_contract.target_column,
        ),
        (
            "experiment_id",
            indexed_experiment_id,
            provenance.experiment_id,
        ),
        (
            "model_id",
            str(
                row[
                    "model_id"
                ]
            ),
            provenance.model_id,
        ),
        (
            "provenance_workflow_id",
            str(
                row[
                    "workflow_id"
                ]
            ),
            provenance.workflow_id,
        ),
        (
            "provenance_dataset_id",
            str(
                row[
                    "dataset_id"
                ]
            ),
            provenance.dataset_id,
        ),
        (
            "train_rows",
            int(
                row[
                    "train_rows"
                ]
            ),
            provenance.train_rows,
        ),
        (
            "test_rows",
            int(
                row[
                    "test_rows"
                ]
            ),
            provenance.test_rows,
        ),
    ]


    for (
        field_name,
        indexed_value,
        authority_value,
    ) in indexed_bindings:

        if (
            indexed_value
            !=
            authority_value
        ):
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Persisted Model Artifact "
                    "index does not match its "
                    "authority payload. "
                    f"field={field_name}"
                )
            )


    expected_training_sha = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    if (
        provenance.training_contract_sha256
        !=
        expected_training_sha
    ):
        raise MLPerformanceEvaluationAuthorityError(
            (
                "Persisted Experiment Provenance "
                "Training Contract fingerprint "
                "is invalid."
            )
        )


    if (
        provenance.metrics
        !=
        metrics
    ):
        raise MLPerformanceEvaluationAuthorityError(
            (
                "Persisted Experiment Provenance "
                "metrics do not match Model "
                "Artifact metrics."
            )
        )


    # --------------------------------------------------------
    # EVALUATION ? MODEL AUTHORITY
    # --------------------------------------------------------


    bindings = [
        (
            "model_id",
            evaluation.model_id,
            str(
                row[
                    "model_id"
                ]
            ),
        ),
        (
            "workflow_id",
            evaluation.workflow_id,
            str(
                row[
                    "workflow_id"
                ]
            ),
        ),
        (
            "reference_dataset_id",
            evaluation.reference_dataset_id,
            str(
                row[
                    "dataset_id"
                ]
            ),
        ),
        (
            "experiment_id",
            evaluation.experiment_id,
            provenance.experiment_id,
        ),
        (
            "preparation_session_revision",
            evaluation.preparation_session_revision,
            provenance.preparation_session_revision,
        ),
        (
            "training_contract_sha256",
            evaluation.training_contract_sha256,
            expected_training_sha,
        ),
        (
            "problem_type",
            evaluation.problem_type,
            training_contract.problem_type,
        ),
        (
            "target_column",
            evaluation.target_column,
            training_contract.target_column,
        ),
        (
            "reference_evaluation_row_count",
            evaluation.reference_evaluation_row_count,
            int(
                row[
                    "test_rows"
                ]
            ),
        ),
    ]


    for (
        field_name,
        evaluation_value,
        authority_value,
    ) in bindings:

        if (
            evaluation_value
            !=
            authority_value
        ):
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "ML Performance Evaluation "
                    "does not match persisted "
                    "Model Artifact authority. "
                    f"field={field_name}"
                )
            )


    # --------------------------------------------------------
    # REFERENCE METRIC AUTHORITY
    # --------------------------------------------------------


    result_names = [
        result.metric_name

        for result
        in evaluation.metric_results
    ]


    if (
        set(
            result_names
        )
        !=
        set(
            metrics
        )
    ):
        raise MLPerformanceEvaluationAuthorityError(
            (
                "Performance reference metric "
                "surface does not match persisted "
                "Model Artifact metrics."
            )
        )


    for result in (
        evaluation.metric_results
    ):

        persisted_value = (
            metrics[
                result.metric_name
            ]
        )


        if not math.isclose(
            result.reference_value,
            persisted_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Performance reference metric "
                    "does not match persisted "
                    "Model Artifact evidence. "
                    f"metric_name="
                    f"{result.metric_name}"
                )
            )


# ============================================================
# REGISTER
# ============================================================


def register_ml_performance_evaluation(
    *,
    evaluation: MLPerformanceEvaluationRecord,
) -> MLPerformanceEvaluationRecord:
    """
    Persist one immutable aggregate-only Performance Evaluation.

    Registration is the durable commit boundary.

    The observed Preparation revision is revalidated inside the
    same SQLite write transaction as the INSERT.
    """

    validated = (
        _validated_evaluation(
            evaluation
        )
    )


    store_root = (
        _store_root()
    )


    payload_json = (
        _canonical_json(
            validated.model_dump(
                mode="json"
            )
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:

        _assert_model_artifact_authority(
            connection=
                connection,

            store_root=
                store_root,

            evaluation=
                validated,
        )


        # ====================================================
        # OBSERVED PREPARATION SNAPSHOT GUARD
        # ====================================================


        observed_session = (
            connection.execute(
                """
                SELECT revision
                FROM preparation_sessions

                WHERE
                    workflow_id = ?

                LIMIT 1
                """,
                (
                    validated.workflow_id,
                ),
            )
            .fetchone()
        )


        if observed_session is None:
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Observed Preparation workflow "
                    "is no longer server-owned."
                )
            )


        current_observed_revision = int(
            observed_session[
                "revision"
            ]
        )


        if (
            current_observed_revision
            !=
            validated
            .observed_preparation_session_revision
        ):
            raise MLPerformanceEvaluationAuthorityError(
                (
                    "Observed Preparation revision "
                    "changed before Performance "
                    "Evaluation persistence. "
                    "expected_revision="
                    f"{validated.observed_preparation_session_revision}, "
                    "current_revision="
                    f"{current_observed_revision}"
                )
            )


        # ====================================================
        # DUPLICATE IDENTITY
        # ====================================================


        existing = (
            connection.execute(
                """
                SELECT performance_evaluation_id
                FROM ml_performance_evaluations

                WHERE
                    store_root = ?
                    AND
                    performance_evaluation_id = ?

                LIMIT 1
                """,
                (
                    store_root,
                    validated
                    .performance_evaluation_id,
                ),
            )
            .fetchone()
        )


        if existing is not None:
            raise (
                MLPerformanceEvaluationAlreadyExistsError(
                    (
                        "ML Performance Evaluation "
                        "identity already exists."
                    )
                )
            )


        # ====================================================
        # INSERT
        # ====================================================


        connection.execute(
            """
            INSERT INTO ml_performance_evaluations (
                store_root,
                performance_evaluation_id,
                model_id,
                workflow_id,
                reference_dataset_id,
                observed_dataset_id,
                experiment_id,
                preparation_session_revision,
                observed_preparation_session_revision,
                training_contract_sha256,
                problem_type,
                target_column,
                reference_evaluation_row_count,
                observed_row_count,
                evaluated_at_utc,
                primary_metric,
                primary_metric_degradation_amount,
                primary_metric_degradation_ratio,
                degradation_basis,
                performance_status,
                privacy_scope,
                rule_version,
                payload_json
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                store_root,

                validated
                .performance_evaluation_id,

                validated.model_id,

                validated.workflow_id,

                validated
                .reference_dataset_id,

                validated
                .observed_dataset_id,

                validated.experiment_id,

                validated
                .preparation_session_revision,

                validated
                .observed_preparation_session_revision,

                validated
                .training_contract_sha256,

                validated.problem_type,

                validated.target_column,

                validated
                .reference_evaluation_row_count,

                validated.observed_row_count,

                validated.evaluated_at_utc,

                validated.primary_metric,

                validated
                .primary_metric_degradation_amount,

                validated
                .primary_metric_degradation_ratio,

                validated.degradation_basis,

                validated.performance_status,

                validated.privacy_scope,

                validated.rule_version,

                payload_json,
            ),
        )


    return validated


# ============================================================
# GET
# ============================================================


def get_ml_performance_evaluation(
    *,
    performance_evaluation_id: str,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> MLPerformanceEvaluationRecord:

    normalized_evaluation_id = (
        _required_text(
            performance_evaluation_id,
            field_name=
                "performance_evaluation_id",
        )
    )


    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )

        if (
            workflow_id
            is not None
        )

        else None
    )


    store_root = (
        _store_root()
    )


    with sqlite_connection(
        write=False
    ) as connection:

        row = (
            connection.execute(
                """
                SELECT *
                FROM ml_performance_evaluations

                WHERE
                    store_root = ?
                    AND
                    performance_evaluation_id = ?

                LIMIT 1
                """,
                (
                    store_root,
                    normalized_evaluation_id,
                ),
            )
            .fetchone()
        )


    if row is None:
        raise MLPerformanceEvaluationNotFoundError(
            (
                "ML Performance Evaluation "
                "was not found."
            )
        )


    record = (
        _row_to_record(
            row
        )
    )


    if (
        normalized_workflow_id
        is not None
        and
        record.workflow_id
        !=
        normalized_workflow_id
    ):
        raise (
            MLPerformanceEvaluationWorkflowMismatchError(
                (
                    "ML Performance Evaluation "
                    "does not belong to the "
                    "requested workflow."
                )
            )
        )


    return record


# ============================================================
# LIST MODEL
# ============================================================


def list_ml_performance_evaluations_for_model(
    *,
    model_id: str,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> list[
    MLPerformanceEvaluationRecord
]:

    normalized_model_id = (
        _required_text(
            model_id,
            field_name=
                "model_id",
        )
    )


    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )

        if (
            workflow_id
            is not None
        )

        else None
    )


    store_root = (
        _store_root()
    )


    query = """
        SELECT *
        FROM ml_performance_evaluations

        WHERE
            store_root = ?
            AND
            model_id = ?
    """


    parameters: list[
        object
    ] = [
        store_root,
        normalized_model_id,
    ]


    if (
        normalized_workflow_id
        is not None
    ):
        query += """
            AND
            workflow_id = ?
        """

        parameters.append(
            normalized_workflow_id
        )


    query += """
        ORDER BY
            evaluated_at_utc ASC,
            performance_evaluation_id ASC
    """


    with sqlite_connection(
        write=False
    ) as connection:

        rows = (
            connection.execute(
                query,
                tuple(
                    parameters
                ),
            )
            .fetchall()
        )


    return [
        _row_to_record(
            row
        )

        for row
        in rows
    ]


# ============================================================
# LIST WORKFLOW
# ============================================================


def list_ml_performance_evaluations_for_workflow(
    *,
    workflow_id: str,
) -> list[
    MLPerformanceEvaluationRecord
]:

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    store_root = (
        _store_root()
    )


    with sqlite_connection(
        write=False
    ) as connection:

        rows = (
            connection.execute(
                """
                SELECT *
                FROM ml_performance_evaluations

                WHERE
                    store_root = ?
                    AND
                    workflow_id = ?

                ORDER BY
                    evaluated_at_utc ASC,
                    performance_evaluation_id ASC
                """,
                (
                    store_root,
                    normalized_workflow_id,
                ),
            )
            .fetchall()
        )


    return [
        _row_to_record(
            row
        )

        for row
        in rows
    ]
