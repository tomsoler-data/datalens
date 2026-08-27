from __future__ import annotations

import json
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request

from app.security.llm_egress import (
    open_local_llm_request,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from app.preparation.dataset_identity import (
    DatasetIdentityReport,
)


# ============================================================
# VERSION
# ============================================================


DATASET_IDENTITY_EXPLANATION_RULE_VERSION = (
    "dataset_identity_explanation_v0.1"
)

DEFAULT_DATASET_IDENTITY_MODEL = "gemma3:4b"

DEFAULT_OLLAMA_CHAT_URL = (
    "http://127.0.0.1:11434/api/chat"
)


# ============================================================
# TYPES
# ============================================================


IdentityExplanationAction = Literal[
    "keep_detected_key",
    "create_surrogate_key",
    "review_identity",
]


# ============================================================
# RAW LLM PROTOCOL
# ============================================================


class RawDatasetIdentityExplanation(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    action: IdentityExplanationAction

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    title: str = Field(
        min_length=1,
        max_length=120,
    )

    explanation: str = Field(
        min_length=1,
        max_length=900,
    )

    user_message: str = Field(
        min_length=1,
        max_length=650,
    )

    referenced_columns: list[str] = Field(
        default_factory=list,
    )

    surrogate_column: str | None

    cautions: list[str] = Field(
        default_factory=list,
        max_length=6,
    )


# ============================================================
# VALIDATED OUTPUT
# ============================================================


class DatasetIdentityExplanation(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    dataset_id: str
    dataset_filename: str
    deterministic_status: str
    action: IdentityExplanationAction
    confidence: float
    title: str
    explanation: str
    user_message: str
    referenced_columns: list[str]
    surrogate_column: str | None
    cautions: list[str]
    python_validated: bool = True
    requires_user_confirmation: bool
    executable: bool = False
    model: str
    rule_version: str = (
        DATASET_IDENTITY_EXPLANATION_RULE_VERSION
    )


# ============================================================
# PRIVACY-PRESERVING FACTS
# ============================================================


def build_identity_explanation_facts(
    report: DatasetIdentityReport,
) -> dict[str, object]:
    """
    Build the only facts sent to the local model.

    No DataFrame and no raw row samples are included.
    """

    preferred = report.preferred_candidate

    return {
        "dataset_id":
            report.dataset_id,

        "dataset_filename":
            report.dataset_filename,

        "row_count":
            report.row_count,

        "column_count":
            report.column_count,

        "status":
            report.status,

        "preferred_candidate": (
            {
                "columns":
                    list(
                        preferred.columns
                    ),

                "kind":
                    preferred.kind,

                "unique":
                    preferred.unique,

                "complete":
                    preferred.complete,

                "deterministic_score":
                    preferred.deterministic_score,
            }
            if preferred is not None
            else None
        ),

        "identifier_like_columns":
            list(
                report.identifier_like_columns
            ),

        "mechanically_unique_columns":
            list(
                report.mechanically_unique_columns
            ),

        "surrogate_key_recommended":
            report.surrogate_key_recommended,

        "suggested_surrogate_column":
            report.suggested_surrogate_column,

        "deterministic_reasons":
            list(
                report.reasons
            ),

        "identity_rule_version":
            report.rule_version,
    }


# ============================================================
# PROMPTS
# ============================================================


def _system_prompt() -> str:
    return (
        "Tu es la couche d'explication locale de DataLens pour "
        "l'identité des lignes d'un dataset. "
        "Python possède l'autorité factuelle. "
        "Tu ne mesures rien et tu ne modifies aucune donnée. "
        "Tu dois uniquement expliquer les faits structurés fournis. "
        "N'invente jamais de colonne, de clé, de valeur, de relation "
        "entre datasets ou de règle métier. "
        "Si status=single_key ou composite_key, tu peux seulement "
        "recommander keep_detected_key ou review_identity. "
        "Si status=surrogate_recommended, tu peux seulement recommander "
        "create_surrogate_key ou review_identity. "
        "Si tu proposes create_surrogate_key, surrogate_column doit être "
        "exactement suggested_surrogate_column. "
        "Une clé technique sert uniquement à identifier une ligne et "
        "ne doit jamais être présentée comme une clé de jointure. "
        "referenced_columns doit contenir uniquement des colonnes "
        "présentes dans les faits fournis. "
        "Le texte doit être clair pour un analyste de données. "
        "Retourne uniquement le JSON conforme au schéma."
    )


def build_identity_explanation_prompt(
    report: DatasetIdentityReport,
) -> str:
    facts = build_identity_explanation_facts(
        report
    )

    return (
        "Explique la situation d'identité des lignes ci-dessous "
        "et formule une recommandation prudente.\n\n"
        "FAITS VALIDÉS PAR PYTHON\n"
        "========================\n"
        +
        json.dumps(
            facts,
            ensure_ascii=False,
            indent=2,
        )
        +
        "\n\n"
        "RAPPEL\n"
        "======\n"
        "Ne complète pas les faits avec des suppositions. "
        "La recommandation reste non exécutable et nécessitera "
        "une confirmation utilisateur."
    )


# ============================================================
# DETERMINISTIC VALIDATION
# ============================================================


def _allowed_column_names(
    report: DatasetIdentityReport,
) -> set[str]:
    names = set(
        report.identifier_like_columns
    )

    names.update(
        report.mechanically_unique_columns
    )

    if report.preferred_candidate is not None:
        names.update(
            report.preferred_candidate.columns
        )

    return names


def validate_identity_explanation(
    *,
    report: DatasetIdentityReport,
    raw: RawDatasetIdentityExplanation,
    model: str = DEFAULT_DATASET_IDENTITY_MODEL,
) -> DatasetIdentityExplanation:
    """
    Validate the LLM narrative against Python-owned facts.

    The model cannot override deterministic identity status.
    """

    allowed_columns = _allowed_column_names(
        report
    )

    invented_columns = (
        set(
            raw.referenced_columns
        )
        -
        allowed_columns
    )

    if invented_columns:
        raise ValueError(
            (
                "Identity explanation referenced unknown or "
                "unauthorized column(s): "
                +
                ", ".join(
                    sorted(
                        invented_columns
                    )
                )
                +
                "."
            )
        )

    if report.status in {
        "single_key",
        "composite_key",
    }:
        if raw.action not in {
            "keep_detected_key",
            "review_identity",
        }:
            raise ValueError(
                (
                    "The model cannot recommend creating a "
                    "surrogate key when Python already detected "
                    "a reliable identity candidate."
                )
            )

        if raw.surrogate_column is not None:
            raise ValueError(
                (
                    "surrogate_column must be null when a reliable "
                    "identity candidate already exists."
                )
            )

    elif report.status == "surrogate_recommended":
        if raw.action not in {
            "create_surrogate_key",
            "review_identity",
        }:
            raise ValueError(
                (
                    "The model cannot claim that a reliable key "
                    "was detected when Python recommends a "
                    "surrogate identity."
                )
            )

        if raw.action == "create_surrogate_key":
            expected_column = (
                report.suggested_surrogate_column
            )

            if expected_column is None:
                raise ValueError(
                    (
                        "Python did not provide an authorized "
                        "surrogate column name."
                    )
                )

            if raw.surrogate_column != expected_column:
                raise ValueError(
                    (
                        "The model attempted to invent or change "
                        "the surrogate column name. Expected: "
                        f"{expected_column}."
                    )
                )

        elif raw.surrogate_column is not None:
            raise ValueError(
                (
                    "surrogate_column must be null when the model "
                    "asks for identity review instead of creating "
                    "the authorized surrogate key."
                )
            )

    requires_user_confirmation = (
        raw.action
        in {
            "create_surrogate_key",
            "review_identity",
        }
    )

    return DatasetIdentityExplanation(
        dataset_id=
            report.dataset_id,

        dataset_filename=
            report.dataset_filename,

        deterministic_status=
            report.status,

        action=
            raw.action,

        confidence=
            raw.confidence,

        title=
            raw.title.strip(),

        explanation=
            raw.explanation.strip(),

        user_message=
            raw.user_message.strip(),

        referenced_columns=
            list(
                dict.fromkeys(
                    raw.referenced_columns
                )
            ),

        surrogate_column=
            raw.surrogate_column,

        cautions=[
            item.strip()
            for item in raw.cautions
            if item.strip()
        ],

        python_validated=
            True,

        requires_user_confirmation=
            requires_user_confirmation,

        executable=
            False,

        model=
            model,

        rule_version=
            DATASET_IDENTITY_EXPLANATION_RULE_VERSION,
    )


# ============================================================
# OLLAMA CALL
# ============================================================


def _ollama_identity_explanation(
    *,
    report: DatasetIdentityReport,
    model: str,
    ollama_chat_url: str,
    timeout_seconds: float,
) -> RawDatasetIdentityExplanation:
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
                    build_identity_explanation_prompt(
                        report
                    ),
            },
        ],

        "stream":
            False,

        "format": (
            RawDatasetIdentityExplanation
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

        method=
            "POST",
    )

    try:
        with open_local_llm_request(
            request,
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
        body = ""

        try:
            body = (
                error
                .read()
                .decode(
                    "utf-8",
                    errors=
                        "replace",
                )
            )

        except Exception:
            pass

        raise RuntimeError(
            (
                "Ollama dataset-identity explanation failed "
                f"with HTTP {error.code}: {body}"
            )
        ) from error

    except URLError as error:
        raise RuntimeError(
            (
                "Ollama dataset-identity explanation is "
                "unavailable. Verify that the local Ollama "
                "service is running."
            )
        ) from error

    except TimeoutError as error:
        raise RuntimeError(
            (
                "Ollama dataset-identity explanation timed out."
            )
        ) from error

    message = payload.get(
        "message",
        {},
    )

    content = message.get(
        "content"
    )

    if not isinstance(
        content,
        str,
    ):
        raise RuntimeError(
            (
                "Ollama dataset-identity explanation returned "
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
                "Ollama dataset-identity explanation returned "
                "invalid JSON content."
            )
        ) from error

    try:
        return (
            RawDatasetIdentityExplanation
            .model_validate(
                parsed
            )
        )

    except ValidationError as error:
        raise RuntimeError(
            (
                "Ollama dataset-identity explanation did not "
                "respect the structured response schema."
            )
        ) from error


# ============================================================
# PUBLIC FUNCTION
# ============================================================


def explain_dataset_identity_with_ai(
    report: DatasetIdentityReport,
    *,
    model: str = DEFAULT_DATASET_IDENTITY_MODEL,
    ollama_chat_url: str = DEFAULT_OLLAMA_CHAT_URL,
    timeout_seconds: float = 30.0,
) -> DatasetIdentityExplanation:
    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater than zero."
        )

    raw = _ollama_identity_explanation(
        report=report,
        model=model,
        ollama_chat_url=ollama_chat_url,
        timeout_seconds=timeout_seconds,
    )

    return validate_identity_explanation(
        report=report,
        raw=raw,
        model=model,
    )
