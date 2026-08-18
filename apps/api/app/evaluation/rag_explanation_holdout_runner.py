from __future__ import annotations


import hashlib
import json

from pathlib import Path

from typing import (
    Any,
)


from app.rag_explanation import (
    DEFAULT_EXPLANATION_MODEL,
    RAG_EXPLANATION_RULE_VERSION,
    SYSTEM_PROMPT,
    generate_grounded_explanation,
)

from app.rag_retrieval import (
    RagSearchHit,
)


# ============================================================
# VERSION
# ============================================================

HOLDOUT_VERSION = (
    "rag_explanation_holdout_v0.1"
)


# ============================================================
# PATHS
# ============================================================

FREEZE_PATH = Path(
    "artifacts/evaluation/holdouts/"
    "rag_explanation_holdout_v0.1_freeze.json"
)


FIRST_RUN_PATH = Path(
    "artifacts/evaluation/experiments/"
    "rag_explanation_holdout_v0.1_first_run.json"
)


# ============================================================
# ACCEPTANCE GATES
# ============================================================

MIN_GENERATION_SUCCESS_RATE = 0.80

MIN_DETERMINISTIC_ABSTENTION_RATE = 1.00

MAX_INVALID_CITATIONS = 0

MAX_INVALID_EVIDENCE_QUOTES = 0

MAX_NUMERIC_VIOLATIONS = 0


# ============================================================
# INDEPENDENT HOLDOUT CASES
#
# These cases are deliberately different from the development
# benchmark used to build rag_explanation_v0.2.
# ============================================================

CASES = [
    # --------------------------------------------------------
    # POSITIVE — INVENTORY
    # --------------------------------------------------------

    {
        "case_id":
            "inventory_reorder",

        "finding": (
            "Certains produits semblent être "
            "réapprovisionnés trop tard."
        ),

        "sources": [
            {
                "filename":
                    "inventory_policy.txt",

                "text": (
                    "Le point de commande correspond "
                    "au niveau de stock à partir duquel "
                    "un réapprovisionnement doit être "
                    "déclenché afin de limiter le risque "
                    "de rupture pendant le délai "
                    "d'approvisionnement."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # POSITIVE — CASH FLOW
    # --------------------------------------------------------

    {
        "case_id":
            "cashflow_liquidity",

        "finding": (
            "La capacité de l'entreprise à régler "
            "ses obligations à court terme doit "
            "être examinée."
        ),

        "sources": [
            {
                "filename":
                    "liquidity.txt",

                "text": (
                    "La liquidité décrit la capacité "
                    "d'une organisation à disposer "
                    "de ressources immédiatement "
                    "mobilisables pour couvrir ses "
                    "obligations à court terme."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # POSITIVE — COHORT
    # --------------------------------------------------------

    {
        "case_id":
            "cohort_retention",

        "finding": (
            "La rétention est comparée entre "
            "plusieurs groupes d'utilisateurs "
            "ayant commencé à des périodes différentes."
        ),

        "sources": [
            {
                "filename":
                    "cohort_analysis.txt",

                "text": (
                    "L'analyse par cohorte regroupe "
                    "les utilisateurs selon une période "
                    "commune d'inscription ou d'acquisition, "
                    "puis suit leur comportement au cours "
                    "des périodes suivantes."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # POSITIVE — DATA LEAKAGE
    # --------------------------------------------------------

    {
        "case_id":
            "model_leakage",

        "finding": (
            "Les performances mesurées avant "
            "déploiement semblent beaucoup plus "
            "favorables que les performances réelles."
        ),

        "sources": [
            {
                "filename":
                    "model_validation.txt",

                "text": (
                    "Une fuite de données peut produire "
                    "des performances artificiellement "
                    "optimistes lorsqu'une information "
                    "indisponible au moment réel de la "
                    "prédiction est utilisée pendant "
                    "l'entraînement ou la validation."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # POSITIVE — SEASONALITY
    # --------------------------------------------------------

    {
        "case_id":
            "seasonal_activity",

        "finding": (
            "Des variations d'activité semblent "
            "réapparaître selon un rythme régulier."
        ),

        "sources": [
            {
                "filename":
                    "seasonality.txt",

                "text": (
                    "La saisonnalité correspond à "
                    "des motifs qui se répètent à "
                    "des intervalles réguliers dans "
                    "une série temporelle. Elle doit "
                    "être distinguée d'une tendance "
                    "de long terme."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # POSITIVE — ENTITY RESOLUTION
    # --------------------------------------------------------

    {
        "case_id":
            "duplicate_entities",

        "finding": (
            "Plusieurs lignes pourraient représenter "
            "le même client malgré des différences "
            "dans les informations d'identification."
        ),

        "sources": [
            {
                "filename":
                    "entity_resolution.txt",

                "text": (
                    "La résolution d'entités cherche "
                    "à déterminer si plusieurs "
                    "enregistrements présentant des "
                    "variations de nom, d'adresse ou "
                    "d'identifiant représentent en "
                    "réalité la même entité."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # POSITIVE — MULTI-SOURCE
    #
    # This case verifies that the model can select one or more
    # valid chunks without inventing a third source.
    # --------------------------------------------------------

    {
        "case_id":
            "customer_waiting_time",

        "finding": (
            "L'augmentation du temps d'attente "
            "pourrait être liée à un déséquilibre "
            "entre la demande et la capacité."
        ),

        "sources": [
            {
                "filename":
                    "queue_demand.txt",

                "text": (
                    "Une file d'attente augmente "
                    "lorsque les demandes arrivent "
                    "plus rapidement qu'elles ne "
                    "peuvent être traitées."
                ),
            },

            {
                "filename":
                    "queue_capacity.txt",

                "text": (
                    "La capacité opérationnelle "
                    "dépend notamment du nombre "
                    "de ressources disponibles pour "
                    "traiter les demandes entrantes."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # POSITIVE — ACCESS CONTROL
    # --------------------------------------------------------

    {
        "case_id":
            "permission_scope",

        "finding": (
            "Des autorisations inutiles sont "
            "attribuées à certains comptes."
        ),

        "sources": [
            {
                "filename":
                    "least_privilege.txt",

                "text": (
                    "Le principe du moindre privilège "
                    "consiste à limiter les permissions "
                    "aux seules autorisations nécessaires "
                    "à l'exécution des fonctions prévues."
                ),
            },
        ],

        "expect_abstention":
            False,
    },

    # --------------------------------------------------------
    # DETERMINISTIC ABSTENTION
    #
    # No accepted source exists. Python must abstain without
    # asking Gemma to invent documentary context.
    # --------------------------------------------------------

    {
        "case_id":
            "abstain_without_document",

        "finding": (
            "Une relation semble exister entre "
            "le prix d'un produit et la météo."
        ),

        "sources": [],

        "expect_abstention":
            True,
    },

    # --------------------------------------------------------
    # SECOND DETERMINISTIC ABSTENTION
    # --------------------------------------------------------

    {
        "case_id":
            "abstain_without_policy_source",

        "finding": (
            "Une recommandation réglementaire "
            "devrait être formulée."
        ),

        "sources": [],

        "expect_abstention":
            True,
    },
]


# ============================================================
# HASH HELPERS
# ============================================================

def canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


def sha256_json(
    value: Any,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            value
        )
    ).hexdigest()


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def file_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ============================================================
# HIT BUILDER
# ============================================================

def build_case_hits(
    *,
    case_id: str,
    sources: list[
        dict[
            str,
            str,
        ]
    ],
) -> list[
    RagSearchHit
]:
    hits: list[
        RagSearchHit
    ] = []


    for (
        index,
        source,
    ) in enumerate(
        sources,
        start=1,
    ):
        text = (
            source[
                "text"
            ]
        )


        hits.append(
            RagSearchHit(
                rank=
                    index,

                score=
                    1.0,

                chunk_id=
                    (
                        "holdout:"
                        f"{case_id}:"
                        f"chunk:{index:04d}"
                    ),

                document_id=
                    (
                        "holdout:"
                        f"{case_id}:"
                        f"document:{index:04d}"
                    ),

                filename=
                    source[
                        "filename"
                    ],

                extension=
                    ".txt",

                chunk_index=
                    index,

                page_number=
                    None,

                source_locator=
                    "document",

                text=
                    text,

                character_count=
                    len(
                        text
                    ),
            )
        )


    return hits


# ============================================================
# FREEZE PAYLOAD
# ============================================================

def build_freeze_payload() -> dict[
    str,
    Any,
]:
    return {
        "holdout_version":
            HOLDOUT_VERSION,

        "explanation_rule_version":
            RAG_EXPLANATION_RULE_VERSION,

        "model":
            DEFAULT_EXPLANATION_MODEL,

        "case_count":
            len(
                CASES
            ),

        "cases_sha256":
            sha256_json(
                CASES
            ),

        "system_prompt_sha256":
            sha256_text(
                SYSTEM_PROMPT
            ),

        "preregistered_gates": {
            "minimum_generation_success_rate":
                MIN_GENERATION_SUCCESS_RATE,

            "minimum_deterministic_abstention_rate":
                MIN_DETERMINISTIC_ABSTENTION_RATE,

            "maximum_invalid_citations":
                MAX_INVALID_CITATIONS,

            "maximum_invalid_evidence_quotes":
                MAX_INVALID_EVIDENCE_QUOTES,

            "maximum_numeric_violations":
                MAX_NUMERIC_VIOLATIONS,
        },

        "policy": {
            "freeze_before_first_model_execution":
                True,

            "first_run_artifact_is_immutable":
                True,

            "tuning_on_this_holdout_requires_new_holdout":
                True,
        },
    }


# ============================================================
# FREEZE CREATION
# ============================================================

def create_freeze() -> None:
    FREEZE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = (
        build_freeze_payload()
    )


    FREEZE_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "=" * 100
    )

    print(
        "RAG EXPLANATION HOLDOUT — FREEZE CREATED"
    )

    print(
        "=" * 100
    )

    print(
        "Holdout:",
        HOLDOUT_VERSION,
    )

    print(
        "Rule:",
        RAG_EXPLANATION_RULE_VERSION,
    )

    print(
        "Model:",
        DEFAULT_EXPLANATION_MODEL,
    )

    print(
        "Cases:",
        len(
            CASES
        ),
    )

    print(
        "Cases SHA256:",
        payload[
            "cases_sha256"
        ],
    )

    print(
        "System prompt SHA256:",
        payload[
            "system_prompt_sha256"
        ],
    )

    print(
        "Freeze artifact:",
        FREEZE_PATH,
    )

    print(
        "Freeze SHA256:",
        file_sha256(
            FREEZE_PATH
        ),
    )

    print()

    print(
        "NO MODEL CALL WAS EXECUTED."
    )

    print(
        "Run the same command again "
        "for the official first holdout run."
    )


# ============================================================
# FREEZE VERIFICATION
# ============================================================

def load_freeze() -> dict[
    str,
    Any,
]:
    return json.loads(
        FREEZE_PATH.read_text(
            encoding="utf-8"
        )
    )


def verify_freeze(
    freeze: dict[
        str,
        Any,
    ],
) -> None:
    current = (
        build_freeze_payload()
    )


    fields = [
        "holdout_version",
        "explanation_rule_version",
        "model",
        "case_count",
        "cases_sha256",
        "system_prompt_sha256",
        "preregistered_gates",
    ]


    mismatches = [
        field

        for field
        in fields

        if (
            freeze.get(
                field
            )
            !=
            current.get(
                field
            )
        )
    ]


    if mismatches:
        raise RuntimeError(
            (
                "Frozen explanation holdout "
                "does not match the current "
                "implementation. Mismatches: "
                +
                ", ".join(
                    mismatches
                )
            )
        )


# ============================================================
# OFFICIAL HOLDOUT RUN
# ============================================================

def run_holdout() -> None:
    if FIRST_RUN_PATH.exists():
        raise FileExistsError(
            (
                "Official first-run artifact "
                "already exists. Refusing to "
                "rerun or overwrite:\n"
                f"{FIRST_RUN_PATH}"
            )
        )


    freeze = (
        load_freeze()
    )


    verify_freeze(
        freeze
    )


    FIRST_RUN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "=" * 100
    )

    print(
        "DATALENS — INDEPENDENT RAG EXPLANATION HOLDOUT"
    )

    print(
        "=" * 100
    )

    print(
        "Holdout:",
        HOLDOUT_VERSION,
    )

    print(
        "Rule:",
        RAG_EXPLANATION_RULE_VERSION,
    )

    print(
        "Model:",
        DEFAULT_EXPLANATION_MODEL,
    )

    print(
        "Cases hash match:",
        True,
    )

    print(
        "System prompt hash match:",
        True,
    )

    print(
        "Cases:",
        len(
            CASES
        ),
    )


    # ========================================================
    # COUNTERS
    # ========================================================

    positive_case_count = 0

    successful_positive_cases = 0

    abstention_case_count = 0

    successful_abstentions = 0

    invalid_citations = 0

    invalid_evidence_quotes = 0

    numeric_violations = 0

    validation_errors = 0


    results: list[
        dict[
            str,
            Any,
        ]
    ] = []


    # ========================================================
    # CASE EXECUTION
    # ========================================================

    for (
        index,
        case,
    ) in enumerate(
        CASES,
        start=1,
    ):
        hits = (
            build_case_hits(
                case_id=
                    case[
                        "case_id"
                    ],

                sources=
                    case[
                        "sources"
                    ],
            )
        )


        expected_abstention = bool(
            case[
                "expect_abstention"
            ]
        )


        if expected_abstention:
            abstention_case_count += 1

        else:
            positive_case_count += 1


        try:
            result = (
                generate_grounded_explanation(
                    finding_text=
                        case[
                            "finding"
                        ],

                    accepted_hits=
                        hits,
                )
            )


            # =================================================
            # EXPECTED ABSTENTION
            # =================================================

            if expected_abstention:
                abstention_correct = (
                    result.status
                    ==
                    "abstained"
                    and
                    len(
                        result.claims
                    )
                    ==
                    0
                )


                if abstention_correct:
                    successful_abstentions += 1


                results.append(
                    {
                        "case_id":
                            case[
                                "case_id"
                            ],

                        "expected_abstention":
                            True,

                        "status":
                            result.status,

                        "correct":
                            abstention_correct,

                        "claim_count":
                            len(
                                result.claims
                            ),

                        "abstention_reason":
                            result.abstention_reason,
                    }
                )


                print()

                print(
                    f"[{index:02d}]",
                    case[
                        "case_id"
                    ],
                )

                print(
                    "  expected: abstained"
                )

                print(
                    "  status:",
                    result.status,
                )

                print(
                    "  correct:",
                    abstention_correct,
                )


                continue


            # =================================================
            # POSITIVE CASE
            # =================================================

            if (
                result.status
                !=
                "ready"
            ):
                results.append(
                    {
                        "case_id":
                            case[
                                "case_id"
                            ],

                        "expected_abstention":
                            False,

                        "status":
                            result.status,

                        "success":
                            False,
                    }
                )


                print()

                print(
                    f"[{index:02d}]",
                    case[
                        "case_id"
                    ],
                )

                print(
                    "  status:",
                    result.status,
                )


                continue


            # =================================================
            # VALIDATE OUTPUT AGAINST ALLOWED HITS
            # =================================================

            allowed_chunk_ids = {
                hit.chunk_id

                for hit
                in hits
            }


            source_text_by_chunk = {
                hit.chunk_id:
                    " ".join(
                        hit.text
                        .split()
                    )

                for hit
                in hits
            }


            claims_payload = []


            case_valid = True


            for claim in result.claims:
                citation_valid = (
                    claim.citation.chunk_id
                    in
                    allowed_chunk_ids
                )


                if not citation_valid:
                    invalid_citations += 1

                    case_valid = False


                normalized_evidence = (
                    " ".join(
                        claim.evidence_quote
                        .split()
                    )
                )


                source_text = (
                    source_text_by_chunk
                    .get(
                        claim.citation.chunk_id,
                        "",
                    )
                )


                evidence_valid = (
                    bool(
                        normalized_evidence
                    )
                    and
                    normalized_evidence
                    in
                    source_text
                )


                if not evidence_valid:
                    invalid_evidence_quotes += 1

                    case_valid = False


                claims_payload.append(
                    {
                        "statement":
                            claim.statement,

                        "evidence_quote":
                            claim.evidence_quote,

                        "citation":
                            claim.citation
                            .model_dump(),

                        "citation_valid":
                            citation_valid,

                        "evidence_valid":
                            evidence_valid,
                    }
                )


            if (
                case_valid
                and
                result.claims
            ):
                successful_positive_cases += 1


            results.append(
                {
                    "case_id":
                        case[
                            "case_id"
                        ],

                    "expected_abstention":
                        False,

                    "status":
                        result.status,

                    "success":
                        (
                            case_valid
                            and
                            bool(
                                result.claims
                            )
                        ),

                    "explanation":
                        result.explanation,

                    "claims":
                        claims_payload,
                }
            )


            print()

            print(
                f"[{index:02d}]",
                case[
                    "case_id"
                ],
            )

            print(
                "  status:",
                result.status,
            )

            print(
                "  explanation:",
                result.explanation,
            )

            print(
                "  claims:",
                len(
                    result.claims
                ),
            )


            for claim in result.claims:
                print(
                    "   -",
                    claim.statement,
                )

                print(
                    "     source:",
                    claim.citation.filename,
                )

                print(
                    "     evidence:",
                    claim.evidence_quote,
                )


        except RuntimeError as error:
            validation_errors += 1


            error_message = str(
                error
            )


            normalized_error = (
                error_message
                .lower()
            )


            if (
                "chunk"
                in normalized_error
            ):
                invalid_citations += 1


            if (
                "preuve"
                in normalized_error
            ):
                invalid_evidence_quotes += 1


            if (
                "numérique"
                in normalized_error
            ):
                numeric_violations += 1


            results.append(
                {
                    "case_id":
                        case[
                            "case_id"
                        ],

                    "status":
                        "validation_error",

                    "success":
                        False,

                    "error":
                        error_message,
                }
            )


            print()

            print(
                f"[{index:02d}]",
                case[
                    "case_id"
                ],
            )

            print(
                "  validation error:",
                error_message,
            )


    # ========================================================
    # METRICS
    # ========================================================

    generation_success_rate = (
        successful_positive_cases
        /
        positive_case_count

        if positive_case_count
        else 0.0
    )


    deterministic_abstention_rate = (
        successful_abstentions
        /
        abstention_case_count

        if abstention_case_count
        else 0.0
    )


    generation_success_pass = (
        generation_success_rate
        >=
        MIN_GENERATION_SUCCESS_RATE
    )


    abstention_pass = (
        deterministic_abstention_rate
        >=
        MIN_DETERMINISTIC_ABSTENTION_RATE
    )


    citation_pass = (
        invalid_citations
        <=
        MAX_INVALID_CITATIONS
    )


    evidence_pass = (
        invalid_evidence_quotes
        <=
        MAX_INVALID_EVIDENCE_QUOTES
    )


    numeric_pass = (
        numeric_violations
        <=
        MAX_NUMERIC_VIOLATIONS
    )


    holdout_pass = (
        generation_success_pass
        and
        abstention_pass
        and
        citation_pass
        and
        evidence_pass
        and
        numeric_pass
    )


    metrics = {
        "positive_case_count":
            positive_case_count,

        "successful_positive_cases":
            successful_positive_cases,

        "generation_success_rate":
            generation_success_rate,

        "abstention_case_count":
            abstention_case_count,

        "successful_abstentions":
            successful_abstentions,

        "deterministic_abstention_rate":
            deterministic_abstention_rate,

        "validation_errors":
            validation_errors,

        "invalid_citations":
            invalid_citations,

        "invalid_evidence_quotes":
            invalid_evidence_quotes,

        "numeric_violations":
            numeric_violations,
    }


    gates = {
        "minimum_generation_success_rate":
            MIN_GENERATION_SUCCESS_RATE,

        "minimum_deterministic_abstention_rate":
            MIN_DETERMINISTIC_ABSTENTION_RATE,

        "maximum_invalid_citations":
            MAX_INVALID_CITATIONS,

        "maximum_invalid_evidence_quotes":
            MAX_INVALID_EVIDENCE_QUOTES,

        "maximum_numeric_violations":
            MAX_NUMERIC_VIOLATIONS,

        "generation_success_pass":
            generation_success_pass,

        "abstention_pass":
            abstention_pass,

        "citation_pass":
            citation_pass,

        "evidence_pass":
            evidence_pass,

        "numeric_pass":
            numeric_pass,

        "holdout_pass":
            holdout_pass,
    }


    # ========================================================
    # ARTIFACT
    # ========================================================

    artifact = {
        "holdout_version":
            HOLDOUT_VERSION,

        "explanation_rule_version":
            RAG_EXPLANATION_RULE_VERSION,

        "model":
            DEFAULT_EXPLANATION_MODEL,

        "freeze_sha256":
            file_sha256(
                FREEZE_PATH
            ),

        "cases_sha256":
            freeze[
                "cases_sha256"
            ],

        "system_prompt_sha256":
            freeze[
                "system_prompt_sha256"
            ],

        "metrics":
            metrics,

        "gates":
            gates,

        "cases":
            results,
    }


    FIRST_RUN_PATH.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "HOLDOUT RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        "Generation success rate:",
        round(
            generation_success_rate,
            6,
        ),

        "| gate:",
        MIN_GENERATION_SUCCESS_RATE,

        "| pass:",
        generation_success_pass,
    )

    print(
        "Deterministic abstention rate:",
        round(
            deterministic_abstention_rate,
            6,
        ),

        "| gate:",
        MIN_DETERMINISTIC_ABSTENTION_RATE,

        "| pass:",
        abstention_pass,
    )

    print()

    print(
        "Validation errors:",
        validation_errors,
    )

    print(
        "Invalid citations:",
        invalid_citations,
    )

    print(
        "Invalid evidence quotes:",
        invalid_evidence_quotes,
    )

    print(
        "Numeric violations:",
        numeric_violations,
    )

    print()

    print(
        "INDEPENDENT HOLDOUT PASS:",
        holdout_pass,
    )

    print()

    print(
        "First-run artifact:",
        FIRST_RUN_PATH,
    )

    print(
        "SHA256:",
        file_sha256(
            FIRST_RUN_PATH
        ),
    )


# ============================================================
# ENTRYPOINT
# ============================================================

def main() -> None:
    if not FREEZE_PATH.exists():
        create_freeze()

        return


    run_holdout()


if __name__ == "__main__":
    main()
    