from __future__ import annotations

import json
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request

from app.security.llm_egress import (
    open_local_llm_request,
)

from app.security.llm_payload import (
    LLMPayloadClass,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.preparation.preparation_artifact_store import (
    PreparationDatasetArtifactInfo,
    list_preparation_artifacts,
)

from app.preparation.preparation_session import (
    get_preparation_session,
)


ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION = (
    "analysis_output_explanation_v0.1"
)

DEFAULT_ANALYSIS_OUTPUT_EXPLANATION_MODEL = "gemma3:4b"

DEFAULT_OLLAMA_CHAT_URL = (
    "http://127.0.0.1:11434/api/chat"
)


AnalysisOutputRecommendationStatus = Literal[
    "recommended_terminal",
    "superseded_intermediate",
]


class AnalysisOutputRecommendationFacts(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    workflow_id: str
    dataset_id: str
    dataset_filename: str
    stage: str
    rows: int
    columns: int
    recommendation_status: AnalysisOutputRecommendationStatus
    is_terminal: bool
    direct_parent_dataset_ids: list[str]
    ancestor_dataset_ids: list[str]
    ancestor_dataset_filenames: list[str]
    root_dataset_ids: list[str]
    root_dataset_filenames: list[str]
    lineage_depth: int
    replaced_dataset_ids: list[str]
    evidence_refs: list[str]
    deterministic_reasons: list[str]
    rule_version: str = ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION


class RawAnalysisOutputExplanation(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    title: str = Field(
        min_length=1,
        max_length=140,
    )

    explanation: str = Field(
        min_length=1,
        max_length=1200,
    )

    user_message: str = Field(
        min_length=1,
        max_length=800,
    )

    referenced_dataset_ids: list[str] = Field(
        default_factory=list,
    )

    cautions: list[str] = Field(
        default_factory=list,
        max_length=6,
    )


class AnalysisOutputExplanation(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    workflow_id: str
    dataset_id: str
    dataset_filename: str
    recommendation_status: AnalysisOutputRecommendationStatus
    confidence: float
    title: str
    explanation: str
    user_message: str
    referenced_dataset_ids: list[str]
    cautions: list[str]
    python_validated: bool = True
    executable: bool = False
    model: str
    rule_version: str = ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION


def _artifact_map(
    artifacts: list[PreparationDatasetArtifactInfo],
) -> dict[str, PreparationDatasetArtifactInfo]:
    return {
        artifact.dataset_id: artifact
        for artifact in artifacts
    }


def _effective_parent_ids(
    artifact: PreparationDatasetArtifactInfo,
) -> tuple[str, ...]:
    """
    Ignore in-place lineage self-parent references.

    Example:
        dataset_id = orders
        parent_dataset_ids = [orders]
    """

    return tuple(
        parent_id
        for parent_id in artifact.parent_dataset_ids
        if parent_id != artifact.dataset_id
    )


def terminal_dataset_ids(
    artifacts: list[PreparationDatasetArtifactInfo],
) -> tuple[str, ...]:
    used_as_parent: set[str] = set()

    known_ids = {
        artifact.dataset_id
        for artifact in artifacts
    }

    for artifact in artifacts:
        for parent_id in _effective_parent_ids(
            artifact
        ):
            if parent_id in known_ids:
                used_as_parent.add(
                    parent_id
                )

    return tuple(
        artifact.dataset_id
        for artifact in artifacts
        if artifact.dataset_id not in used_as_parent
    )


def _ancestor_ids(
    *,
    dataset_id: str,
    artifacts: list[PreparationDatasetArtifactInfo],
) -> tuple[str, ...]:
    by_id = _artifact_map(
        artifacts
    )

    ancestors: list[str] = []
    visited: set[str] = set()

    def visit(
        current_id: str,
    ) -> None:
        artifact = by_id.get(
            current_id
        )

        if artifact is None:
            return

        for parent_id in _effective_parent_ids(
            artifact
        ):
            if parent_id in visited:
                continue

            visited.add(
                parent_id
            )

            ancestors.append(
                parent_id
            )

            visit(
                parent_id
            )

    visit(
        dataset_id
    )

    return tuple(
        ancestors
    )


def _lineage_depth(
    *,
    dataset_id: str,
    artifacts: list[PreparationDatasetArtifactInfo],
) -> int:
    by_id = _artifact_map(
        artifacts
    )

    visiting: set[str] = set()
    cache: dict[str, int] = {}

    def depth(
        current_id: str,
    ) -> int:
        if current_id in cache:
            return cache[
                current_id
            ]

        if current_id in visiting:
            return 0

        visiting.add(
            current_id
        )

        artifact = by_id.get(
            current_id
        )

        if artifact is None:
            result = 0

        else:
            parent_depths = [
                depth(
                    parent_id
                )
                for parent_id in _effective_parent_ids(
                    artifact
                )
                if parent_id in by_id
            ]

            result = (
                0
                if not parent_depths
                else 1 + max(
                    parent_depths
                )
            )

        visiting.discard(
            current_id
        )

        cache[
            current_id
        ] = result

        return result

    return depth(
        dataset_id
    )


def build_analysis_output_recommendation_facts(
    *,
    workflow_id: str,
    dataset_id: str,
) -> AnalysisOutputRecommendationFacts:
    session = get_preparation_session(
        workflow_id
    )

    artifacts = list_preparation_artifacts(
        workflow_id=workflow_id
    )

    if not artifacts:
        raise ValueError(
            (
                "No materialized Preparation artifacts "
                "are available for output explanation."
            )
        )

    by_id = _artifact_map(
        artifacts
    )

    candidate = by_id.get(
        dataset_id
    )

    if candidate is None:
        raise ValueError(
            (
                "Preparation output candidate was not found: "
                f"{dataset_id}"
            )
        )

    terminal_ids = set(
        terminal_dataset_ids(
            artifacts
        )
    )

    is_terminal = (
        dataset_id in terminal_ids
    )

    ancestors = _ancestor_ids(
        dataset_id=dataset_id,
        artifacts=artifacts,
    )

    root_ids = [
        root_id
        for root_id in session.selected_analysis_dataset_ids
        if (
            root_id == dataset_id
            or
            root_id in ancestors
        )
    ]

    ancestor_filenames = [
        (
            by_id[
                ancestor_id
            ].dataset_filename
            if ancestor_id in by_id
            else ancestor_id
        )
        for ancestor_id in ancestors
    ]

    root_filenames = [
        (
            by_id[
                root_id
            ].dataset_filename
            if root_id in by_id
            else root_id
        )
        for root_id in root_ids
    ]

    recommendation_status: (
        AnalysisOutputRecommendationStatus
    ) = (
        "recommended_terminal"
        if is_terminal
        else "superseded_intermediate"
    )

    reasons: list[str] = []

    if is_terminal:
        reasons.append(
            (
                "The artifact is terminal in the current "
                "Preparation lineage: no materialized descendant "
                "currently replaces it."
            )
        )

        if ancestors:
            reasons.append(
                (
                    f"It already incorporates {len(ancestors)} "
                    "ancestor artifact(s), so selecting those "
                    "ancestors as well would duplicate lineage "
                    "coverage."
                )
            )

        if len(root_ids) > 1:
            reasons.append(
                (
                    f"It consolidates {len(root_ids)} imported "
                    "root datasets into one analytical artifact."
                )
            )

        if candidate.stage == "combine":
            reasons.append(
                (
                    "The artifact is a materialized COMBINE output "
                    "produced after controlled join execution."
                )
            )

        elif candidate.stage == "transform":
            reasons.append(
                (
                    "The artifact is the latest materialized "
                    "TRANSFORM output on its lineage branch."
                )
            )

        elif candidate.stage == "clean":
            reasons.append(
                (
                    "The artifact is the latest materialized "
                    "CLEAN output on its lineage branch."
                )
            )

        else:
            reasons.append(
                (
                    "The root artifact remains terminal because "
                    "no derived artifact currently replaces it."
                )
            )

    else:
        reasons.append(
            (
                "The artifact is not terminal in the current "
                "Preparation lineage because another materialized "
                "artifact descends from it."
            )
        )

        reasons.append(
            (
                "Selecting both this artifact and its descendant "
                "can cause redundant analytical coverage."
            )
        )

    return AnalysisOutputRecommendationFacts(
        workflow_id=workflow_id,
        dataset_id=candidate.dataset_id,
        dataset_filename=candidate.dataset_filename,
        stage=candidate.stage,
        rows=candidate.rows,
        columns=candidate.columns,
        recommendation_status=recommendation_status,
        is_terminal=is_terminal,
        direct_parent_dataset_ids=list(
            _effective_parent_ids(
                candidate
            )
        ),
        ancestor_dataset_ids=list(
            ancestors
        ),
        ancestor_dataset_filenames=ancestor_filenames,
        root_dataset_ids=root_ids,
        root_dataset_filenames=root_filenames,
        lineage_depth=_lineage_depth(
            dataset_id=dataset_id,
            artifacts=artifacts,
        ),
        replaced_dataset_ids=list(
            ancestors
        ),
        evidence_refs=list(
            candidate.evidence_refs
        ),
        deterministic_reasons=reasons,
        rule_version=ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION,
    )


def build_output_explanation_llm_facts(
    facts: AnalysisOutputRecommendationFacts,
) -> dict[str, object]:
    """
    Only Preparation metadata is sent to the local model.

    No DataFrame, sample row, raw value or statistical result
    is included.
    """

    return {
        "dataset_id":
            facts.dataset_id,

        "dataset_filename":
            facts.dataset_filename,

        "stage":
            facts.stage,

        "rows":
            facts.rows,

        "columns":
            facts.columns,

        "recommendation_status":
            facts.recommendation_status,

        "is_terminal":
            facts.is_terminal,

        "direct_parent_dataset_ids":
            list(
                facts.direct_parent_dataset_ids
            ),

        "ancestor_dataset_ids":
            list(
                facts.ancestor_dataset_ids
            ),

        "ancestor_dataset_filenames":
            list(
                facts.ancestor_dataset_filenames
            ),

        "root_dataset_ids":
            list(
                facts.root_dataset_ids
            ),

        "root_dataset_filenames":
            list(
                facts.root_dataset_filenames
            ),

        "lineage_depth":
            facts.lineage_depth,

        "deterministic_reasons":
            list(
                facts.deterministic_reasons
            ),

        "rule_version":
            facts.rule_version,
    }


def _system_prompt() -> str:
    return (
        "Tu es la couche d'explication locale de DataLens pour "
        "la sélection de la sortie analytique finale. "
        "Python possède l'autorité factuelle et détermine si "
        "l'artefact est terminal ou intermédiaire. "
        "Tu ne dois jamais modifier cette recommandation. "
        "Tu expliques uniquement les faits structurés fournis. "
        "N'invente jamais de colonne, de métrique, de contenu métier, "
        "de relation, de transformation, de nombre de lignes ou de "
        "dataset absent des faits. "
        "Si recommendation_status=recommended_terminal, explique "
        "pourquoi cette sortie est préférable à ses ancêtres pour "
        "l'analyse finale. "
        "Si recommendation_status=superseded_intermediate, explique "
        "pourquoi elle n'est pas la sortie recommandée. "
        "referenced_dataset_ids doit contenir uniquement des IDs "
        "présents dans les faits fournis. "
        "Le texte doit être court, concret et compréhensible par "
        "un analyste de données. "
        "Retourne uniquement le JSON conforme au schéma."
    )


def build_output_explanation_prompt(
    facts: AnalysisOutputRecommendationFacts,
) -> str:
    payload = build_output_explanation_llm_facts(
        facts
    )

    return (
        "Explique la recommandation de sortie analytique suivante.\n\n"
        "FAITS VALIDÉS PAR PYTHON\n"
        "========================\n"
        +
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        +
        "\n\n"
        "RAPPEL\n"
        "======\n"
        "La lineage et le statut de recommandation sont imposés "
        "par Python. Ne les réinterprète pas."
    )


def _allowed_dataset_ids(
    facts: AnalysisOutputRecommendationFacts,
) -> set[str]:
    return {
        facts.dataset_id,
        *facts.direct_parent_dataset_ids,
        *facts.ancestor_dataset_ids,
        *facts.root_dataset_ids,
    }


def validate_analysis_output_explanation(
    *,
    facts: AnalysisOutputRecommendationFacts,
    raw: RawAnalysisOutputExplanation,
    model: str = DEFAULT_ANALYSIS_OUTPUT_EXPLANATION_MODEL,
) -> AnalysisOutputExplanation:
    invented_ids = (
        set(
            raw.referenced_dataset_ids
        )
        -
        _allowed_dataset_ids(
            facts
        )
    )

    if invented_ids:
        raise ValueError(
            (
                "Analysis output explanation referenced "
                "unknown or unauthorized dataset id(s): "
                +
                ", ".join(
                    sorted(
                        invented_ids
                    )
                )
                +
                "."
            )
        )

    return AnalysisOutputExplanation(
        workflow_id=facts.workflow_id,
        dataset_id=facts.dataset_id,
        dataset_filename=facts.dataset_filename,
        recommendation_status=facts.recommendation_status,
        confidence=raw.confidence,
        title=raw.title.strip(),
        explanation=raw.explanation.strip(),
        user_message=raw.user_message.strip(),
        referenced_dataset_ids=list(
            dict.fromkeys(
                raw.referenced_dataset_ids
            )
        ),
        cautions=[
            caution.strip()
            for caution in raw.cautions
            if caution.strip()
        ],
        python_validated=True,
        executable=False,
        model=model,
        rule_version=ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION,
    )


def _ollama_output_explanation(
    *,
    facts: AnalysisOutputRecommendationFacts,
    model: str,
    ollama_chat_url: str,
    timeout_seconds: float,
) -> RawAnalysisOutputExplanation:
    request_payload = {
        "model":
            model,

        "messages": [
            {
                "role":
                    "system",
                "content":
                    _system_prompt(),
            },
            {
                "role":
                    "user",
                "content":
                    build_output_explanation_prompt(
                        facts
                    ),
            },
        ],

        "stream":
            False,

        "format": (
            RawAnalysisOutputExplanation
            .model_json_schema()
        ),

        "options": {
            "temperature":
                0.0,
        },
    }

    request = Request(
        ollama_chat_url,
        data=(
            json.dumps(
                request_payload,
                ensure_ascii=False,
            )
            .encode(
                "utf-8"
            )
        ),
        headers={
            "Content-Type":
                "application/json",
        },
        method="POST",
    )

    try:
        with open_local_llm_request(
            request,
            payload_class=(
                LLMPayloadClass
                .DETERMINISTIC_EVIDENCE
            ),
            timeout=timeout_seconds,
        ) as response:
            payload = json.loads(
                response
                .read()
                .decode(
                    "utf-8"
                )
            )

    except HTTPError as error:
        raise RuntimeError(
            (
                "Local model analysis-output explanation "
                "request failed."
            )
        ) from error

    except URLError as error:
        raise RuntimeError(
            (
                "Ollama analysis-output explanation is unavailable. "
                "Verify that the local Ollama service is running."
            )
        ) from error

    except TimeoutError as error:
        raise RuntimeError(
            (
                "Ollama analysis-output explanation timed out."
            )
        ) from error

    content = (
        payload
        .get(
            "message",
            {},
        )
        .get(
            "content"
        )
    )

    if not isinstance(
        content,
        str,
    ):
        raise RuntimeError(
            (
                "Ollama analysis-output explanation returned "
                "no textual JSON content."
            )
        )

    try:
        parsed = json.loads(
            content
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            (
                "Ollama analysis-output explanation returned "
                "invalid JSON content."
            )
        ) from error

    try:
        return RawAnalysisOutputExplanation.model_validate(
            parsed
        )

    except ValidationError as error:
        raise RuntimeError(
            (
                "Ollama analysis-output explanation did not "
                "respect the structured response schema."
            )
        ) from error


def explain_analysis_output_with_ai(
    *,
    facts: AnalysisOutputRecommendationFacts,
    model: str = DEFAULT_ANALYSIS_OUTPUT_EXPLANATION_MODEL,
    ollama_chat_url: str = DEFAULT_OLLAMA_CHAT_URL,
    timeout_seconds: float = 30.0,
) -> AnalysisOutputExplanation:
    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater than zero."
        )

    raw = _ollama_output_explanation(
        facts=facts,
        model=model,
        ollama_chat_url=ollama_chat_url,
        timeout_seconds=timeout_seconds,
    )

    return validate_analysis_output_explanation(
        facts=facts,
        raw=raw,
        model=model,
    )
