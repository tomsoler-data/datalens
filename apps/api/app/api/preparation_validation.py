from __future__ import annotations

from typing import (
    Any,
    Dict,
)

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.preparation.final_validation_v0_2 import (
    FinalPreparationValidationV02BlockedError,
    require_final_preparation_validation_v0_2,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    PreparationSessionRevisionConflictError,
    PreparationSessionView,
    get_preparation_session,
    record_validation_stage_signal,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_VALIDATION_API_VERSION = (
    "preparation_validation_api_v0.2"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix="/preparation",
    tags=[
        "preparation",
    ],
)


# ============================================================
# STRICT REQUEST
# ============================================================


class PreparationValidationRequest(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str = Field(
        min_length=1
    )


# ============================================================
# ERROR DETAILS
# ============================================================


def _revision_conflict_detail(
    *,
    error: Exception,
    workflow_id: str,
) -> Dict[
    str,
    Any,
]:
    return {
        "error": (
            "preparation_session_revision_conflict"
        ),

        "message": str(
            error
        ),

        "workflow_id": (
            workflow_id
        ),

        "retryable": (
            True
        ),

        "api_version": (
            PREPARATION_VALIDATION_API_VERSION
        ),
    }


def _invalid_request_detail(
    *,
    error: Exception,
    workflow_id: str,
) -> Dict[
    str,
    Any,
]:
    return {
        "error": (
            "invalid_preparation_validation_request"
        ),

        "message": str(
            error
        ),

        "workflow_id": (
            workflow_id
        ),

        "api_version": (
            PREPARATION_VALIDATION_API_VERSION
        ),
    }


# ============================================================
# VALIDATE
# ============================================================


@router.post(
    "/validate",
    response_model=
        PreparationSessionView,
)
def validate_preparation_session(
    request: PreparationValidationRequest,
) -> PreparationSessionView:
    """
    Run Final Preparation Validation v0.2.

    The client sends only workflow_id.

    The client cannot provide:

        passed = True
        status = "passed"
        ready_for_analysis = True
        dataset_ids = [...]
        analysis_output_dataset_ids = [...]

    Trust sequence:

        server-owned PreparationSession
                    ↓
        Final Preparation Validation v0.2
                    ↓
        Artifact Store / lineage revalidation
                    ↓
        optimistic session revision guard
                    ↓
        VALIDATE stage commit

    A successful VALIDATE stage contains the final
    analysis_output_dataset_ids, not the immutable
    Preparation-root dataset scope.
    """

    try:
        # ====================================================
        # SERVER-OWNED SESSION
        # ====================================================

        session = (
            get_preparation_session(
                request.workflow_id
            )
        )


        # ====================================================
        # FINAL VALIDATION v0.2
        # ====================================================

        try:
            report = (
                require_final_preparation_validation_v0_2(
                    session
                )
            )


        except FinalPreparationValidationV02BlockedError as error:
            # =================================================
            # RECORD FAILED VALIDATION ATTEMPT
            #
            # This write is also revision-guarded.
            #
            # If the session changed after evaluation, the
            # failed report itself is stale and must not be
            # committed.
            # =================================================

            record_validation_stage_signal(
                workflow_id=
                    request.workflow_id,

                completed=
                    True,

                passed=
                    False,

                dataset_ids=[],

                evidence_refs=[
                    (
                        "final_validation:"
                        f"{error.report.rule_version}"
                    ),

                    (
                        "final_validation_failed_checks:"
                        f"{error.report.failed_check_count}"
                    ),
                ],

                blocking_reasons=
                    error.report
                    .blocking_reasons,

                expected_revision=
                    error.report
                    .session_revision,
            )


            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),

                detail={
                    "error": (
                        "final_preparation_validation_failed"
                    ),

                    "message": (
                        "Preparation is not ready "
                        "for final validation."
                    ),

                    "workflow_id": (
                        request.workflow_id
                    ),

                    "validation": (
                        error.report.model_dump(
                            mode="json"
                        )
                    ),

                    "api_version": (
                        PREPARATION_VALIDATION_API_VERSION
                    ),
                },
            ) from error


        # ====================================================
        # PASSED
        #
        # VALIDATE now certifies final analytical outputs,
        # not Preparation roots.
        # ====================================================

        updated = (
            record_validation_stage_signal(
                workflow_id=
                    request.workflow_id,

                completed=
                    True,

                passed=
                    True,

                dataset_ids=
                    list(
                        report
                        .analysis_output_dataset_ids
                    ),

                evidence_refs=[
                    (
                        "final_validation:"
                        f"{report.rule_version}"
                    ),

                    (
                        "final_validation_passed_checks:"
                        f"{report.passed_check_count}"
                    ),

                    (
                        "analysis_output_scope:"
                        + ",".join(
                            report
                            .analysis_output_dataset_ids
                        )
                    ),
                ],

                blocking_reasons=[],

                expected_revision=
                    report.session_revision,
            )
        )


        return (
            updated
        )


    # ========================================================
    # SESSION NOT FOUND
    # ========================================================

    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),

            detail={
                "error": (
                    "preparation_session_not_found"
                ),

                "message": str(
                    error
                ),

                "workflow_id": (
                    request.workflow_id
                ),

                "api_version": (
                    PREPARATION_VALIDATION_API_VERSION
                ),
            },
        ) from error


    # ========================================================
    # STALE VALIDATION RESULT
    # ========================================================

    except PreparationSessionRevisionConflictError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),

            detail=(
                _revision_conflict_detail(
                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error


    # ========================================================
    # NORMALIZATION / INVALID SERVER REQUEST
    # ========================================================

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),

            detail=(
                _invalid_request_detail(
                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error


    except HTTPException:
        raise