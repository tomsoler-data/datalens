from __future__ import annotations

import json

from time import perf_counter
from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.ai.dataset_dependency_prompt_v1 import (
    DATASET_DEPENDENCY_MODEL,
    DATASET_DEPENDENCY_PROMPT_VERSION,
    DATASET_DEPENDENCY_SYSTEM_PROMPT,
)

from app.ai.provider import (
    client,
)

from app.planning.analytical_v1.dependency import (
    DATASET_DEPENDENCY_CONTRACT_VERSION,
    DatasetDependencyCandidate,
    validate_dependency_candidate,
)

from app.planning.analytical_v1.relationships import (
    RoutingRelationshipContext,
)


# ============================================================
# VERSION
# ============================================================

DATASET_DEPENDENCY_EXTRACTOR_VERSION = (
    "dataset_dependency_extractor_v1.0"
)


# ============================================================
# LOCKED MODEL CONFIGURATION
# ============================================================

DATASET_DEPENDENCY_TEMPERATURE = 0


DATASET_DEPENDENCY_THINKING = False


# ============================================================
# STATUS
# ============================================================

DatasetDependencyExtractionStatus = Literal[
    "valid",
    "invalid_candidate",
    "generation_error",
]


# ============================================================
# RESULT
# ============================================================

class DatasetDependencyExtractionResult(
    BaseModel
):
    """
    Production result of semantic dataset dependency
    extraction.

    valid
        Structured output was produced and every dataset_id
        is authorized by the trusted structural context.

    invalid_candidate
        Structured generation succeeded, but Python rejected
        one or more semantic dependency references.

    generation_error
        Ollama failed or the returned content did not satisfy
        DatasetDependencyCandidate.

    IMPORTANT:
        status="valid" does NOT mean that the analytical
        request is structurally executable.

        Cross-dataset feasibility is evaluated by the next
        deterministic Python layer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    extractor_version: str

    prompt_version: str

    contract_version: str

    model: str

    temperature: int

    thinking: bool

    status: DatasetDependencyExtractionStatus

    valid_for_feasibility_gate: bool

    inference_ms: float

    raw_content: (
        str
        | None
    )

    candidate: (
        DatasetDependencyCandidate
        | None
    )

    error: (
        str
        | None
    )


# ============================================================
# DATASET SERIALIZATION
# ============================================================

def _serialize_dataset(
    dataset: Any,
) -> dict[str, Any]:
    """
    Serialize exactly the semantic dataset information used
    by the historical dependency extractor.

    Relationship metadata and available tools are
    deliberately excluded.
    """

    return {
        "dataset_id":
            dataset.dataset_id,

        "filename":
            dataset.filename,

        "grain":
            dataset.grain,

        "entity_columns":
            dataset.entity_columns,

        "columns": [
            column.model_dump(
                mode="json",
            )

            for column
            in dataset.columns
        ],
    }


# ============================================================
# MODEL-VISIBLE CONTEXT
# ============================================================

def build_dependency_visible_context(
    *,
    user_request: str,
    context: RoutingRelationshipContext,
) -> dict[str, Any]:
    """
    Build the only information that Qwen may observe.

    Deliberately absent:
    - available_tools;
    - validated relationships;
    - relationship keys;
    - feasibility information;
    - routing decisions;
    - benchmark expectations.
    """

    normalized_request = (
        user_request.strip()
    )


    if not (
        normalized_request
    ):

        raise ValueError(
            "Dataset dependency extraction requires a "
            "non-empty user_request."
        )


    return {
        "user_request":
            normalized_request,

        "datasets": [
            _serialize_dataset(
                dataset
            )

            for dataset
            in context.datasets
        ],
    }


# ============================================================
# EXACT USER PROMPT
# ============================================================

def build_dependency_user_prompt(
    *,
    user_request: str,
    context: RoutingRelationshipContext,
) -> str:
    """
    Preserve the model-visible prompt format used by the
    historical v0.8 dependency extractor.
    """

    visible_context = (
        build_dependency_visible_context(
            user_request=(
                user_request
            ),

            context=(
                context
            ),
        )
    )


    return (
        "CONTEXTE:\n\n"
        + json.dumps(
            visible_context,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n"
        + (
            "Identifie uniquement les groupes de datasets "
            "nécessaires aux résultats analytiques demandés."
        )
    )


# ============================================================
# OLLAMA RESPONSE
# ============================================================

def _extract_response_content(
    response: Any,
) -> str:
    """
    Extract response.message.content defensively.
    """

    message = getattr(
        response,
        "message",
        None,
    )


    if (
        message
        is None
    ):

        raise ValueError(
            "Dataset Dependency Extractor response "
            "contains no message."
        )


    content = getattr(
        message,
        "content",
        None,
    )


    if (
        content
        is None
    ):

        raise ValueError(
            "Dataset Dependency Extractor response "
            "contains no message content."
        )


    content = (
        str(
            content
        )
    )


    if not (
        content.strip()
    ):

        raise ValueError(
            "Dataset Dependency Extractor returned "
            "empty structured content."
        )


    return (
        content
    )


# ============================================================
# RESULT HELPERS
# ============================================================

def _build_result(
    *,
    status: DatasetDependencyExtractionStatus,
    valid_for_feasibility_gate: bool,
    inference_ms: float,
    raw_content: (
        str
        | None
    ),
    candidate: (
        DatasetDependencyCandidate
        | None
    ),
    error: (
        str
        | None
    ),
    model: str,
) -> DatasetDependencyExtractionResult:

    return (
        DatasetDependencyExtractionResult(
            extractor_version=(
                DATASET_DEPENDENCY_EXTRACTOR_VERSION
            ),

            prompt_version=(
                DATASET_DEPENDENCY_PROMPT_VERSION
            ),

            contract_version=(
                DATASET_DEPENDENCY_CONTRACT_VERSION
            ),

            model=(
                model
            ),

            temperature=(
                DATASET_DEPENDENCY_TEMPERATURE
            ),

            thinking=(
                DATASET_DEPENDENCY_THINKING
            ),

            status=(
                status
            ),

            valid_for_feasibility_gate=(
                valid_for_feasibility_gate
            ),

            inference_ms=(
                inference_ms
            ),

            raw_content=(
                raw_content
            ),

            candidate=(
                candidate
            ),

            error=(
                error
            ),
        )
    )


# ============================================================
# PUBLIC EXTRACTION
# ============================================================

def extract_dataset_dependencies(
    *,
    user_request: str,
    context: RoutingRelationshipContext,
    model: str = (
        DATASET_DEPENDENCY_MODEL
    ),
    chat_client: (
        Any
        | None
    ) = None,
) -> DatasetDependencyExtractionResult:
    """
    Perform semantic dependency extraction.

    Pipeline:

        user request
        + dataset schemas
            ↓
        Qwen structured generation
            ↓
        DatasetDependencyCandidate
            ↓
        deterministic dataset reference validation
            ↓
        VALID / INVALID_CANDIDATE

    No structural feasibility decision is made here.

    chat_client is injectable so tests can execute without
    contacting Ollama.
    """

    active_client = (
        client

        if (
            chat_client
            is None
        )

        else (
            chat_client
        )
    )


    raw_content: (
        str
        | None
    ) = None


    started_at = (
        perf_counter()
    )


    # ========================================================
    # MODEL GENERATION
    # ========================================================

    try:

        user_prompt = (
            build_dependency_user_prompt(
                user_request=(
                    user_request
                ),

                context=(
                    context
                ),
            )
        )


        response = (
            active_client.chat(
                model=(
                    model
                ),

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            DATASET_DEPENDENCY_SYSTEM_PROMPT,
                    },

                    {
                        "role":
                            "user",

                        "content":
                            user_prompt,
                    },
                ],

                format=(
                    DatasetDependencyCandidate
                    .model_json_schema()
                ),

                options={
                    "temperature":
                        DATASET_DEPENDENCY_TEMPERATURE,
                },

                think=(
                    DATASET_DEPENDENCY_THINKING
                ),
            )
        )


        raw_content = (
            _extract_response_content(
                response
            )
        )


        candidate = (
            DatasetDependencyCandidate
            .model_validate_json(
                raw_content
            )
        )


    except Exception as error:

        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        return (
            _build_result(
                status=(
                    "generation_error"
                ),

                valid_for_feasibility_gate=False,

                inference_ms=(
                    inference_ms
                ),

                raw_content=(
                    raw_content
                ),

                candidate=None,

                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),

                model=(
                    model
                ),
            )
        )


    inference_ms = (
        (
            perf_counter()
            - started_at
        )
        * 1000.0
    )


    # ========================================================
    # DETERMINISTIC REFERENCE VALIDATION
    #
    # This checks whether the model used only dataset IDs
    # supplied by DataLens.
    #
    # It deliberately does NOT evaluate whether those
    # datasets can be combined.
    # ========================================================

    try:

        validate_dependency_candidate(
            candidate=(
                candidate
            ),

            context=(
                context
            ),
        )


    except ValueError as error:

        return (
            _build_result(
                status=(
                    "invalid_candidate"
                ),

                valid_for_feasibility_gate=False,

                inference_ms=(
                    inference_ms
                ),

                raw_content=(
                    raw_content
                ),

                candidate=(
                    candidate
                ),

                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),

                model=(
                    model
                ),
            )
        )


    # ========================================================
    # VALID SEMANTIC CANDIDATE
    #
    # This is permission to enter the feasibility gate,
    # NOT permission to execute an analysis.
    # ========================================================

    return (
        _build_result(
            status=(
                "valid"
            ),

            valid_for_feasibility_gate=True,

            inference_ms=(
                inference_ms
            ),

            raw_content=(
                raw_content
            ),

            candidate=(
                candidate
            ),

            error=None,

            model=(
                model
            ),
        )
    )


# ============================================================
# REQUIRE VALID CANDIDATE
# ============================================================

def require_dataset_dependencies(
    *,
    user_request: str,
    context: RoutingRelationshipContext,
    model: str = (
        DATASET_DEPENDENCY_MODEL
    ),
    chat_client: (
        Any
        | None
    ) = None,
) -> DatasetDependencyCandidate:
    """
    Return a dependency candidate only when its dataset
    references have passed deterministic validation.

    The returned candidate must STILL pass the structural
    feasibility gate before analytical planning.
    """

    result = (
        extract_dataset_dependencies(
            user_request=(
                user_request
            ),

            context=(
                context
            ),

            model=(
                model
            ),

            chat_client=(
                chat_client
            ),
        )
    )


    if (
        result.status
        == "generation_error"
    ):

        raise RuntimeError(
            "Dataset Dependency Extractor generation "
            f"failed. {result.error}"
        )


    if (
        result.status
        == "invalid_candidate"
    ):

        raise ValueError(
            "Dataset Dependency Extractor produced an "
            "invalid semantic candidate. "
            f"{result.error}"
        )


    if (
        not result.valid_for_feasibility_gate
        or result.candidate
        is None
    ):

        raise RuntimeError(
            "Dataset Dependency Extractor reached an "
            "inconsistent valid state."
        )


    return (
        result.candidate
    )


# ============================================================
# METADATA
# ============================================================

def dataset_dependency_extractor_metadata() -> dict[
    str,
    Any,
]:
    """
    Observability metadata for the production extractor.
    """

    return {
        "extractor_version":
            DATASET_DEPENDENCY_EXTRACTOR_VERSION,

        "prompt_version":
            DATASET_DEPENDENCY_PROMPT_VERSION,

        "contract_version":
            DATASET_DEPENDENCY_CONTRACT_VERSION,

        "model":
            DATASET_DEPENDENCY_MODEL,

        "temperature":
            DATASET_DEPENDENCY_TEMPERATURE,

        "thinking":
            DATASET_DEPENDENCY_THINKING,
    }