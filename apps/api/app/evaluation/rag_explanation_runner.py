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
    generate_grounded_explanation,
)

from app.rag_retrieval import (
    RagSearchHit,
)


# ============================================================
# VERSION
# ============================================================

BENCHMARK_VERSION = (
    "rag_explanation_benchmark_v0.2"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = Path(
    "artifacts/evaluation/experiments/"
    "rag_explanation_benchmark_v0.2.json"
)


# ============================================================
# GATES
# ============================================================

MIN_SUCCESS_RATE = 0.80

MAX_INVALID_CITATIONS = 0

MAX_INVALID_EVIDENCE_QUOTES = 0

MAX_NUMERIC_VIOLATIONS = 0


# ============================================================
# CASES
#
# Same corpus as v0.1.
# Only the explanation implementation changed.
# ============================================================

CASES = [
    {
        "case_id":
            "water_gap",

        "finding": (
            "Écart entre l'accès basique "
            "à l'eau potable et l'accès "
            "géré en toute sécurité."
        ),

        "source": (
            "L'accès basique et l'accès sécurisé "
            "ne représentent pas le même niveau "
            "de service. La modernisation peut "
            "être étudiée en examinant l'écart "
            "entre ces niveaux de couverture."
        ),
    },

    {
        "case_id":
            "missing_values",

        "finding": (
            "Le dataset contient une proportion "
            "importante de valeurs manquantes."
        ),

        "source": (
            "Les valeurs manquantes doivent être "
            "examinées avant l'analyse. Leur "
            "répartition peut modifier "
            "l'interprétation des résultats et "
            "la qualité des estimations."
        ),
    },

    {
        "case_id":
            "join_grain",

        "finding": (
            "Une jointure présente un risque "
            "de multiplication des observations."
        ),

        "source": (
            "Une relation plusieurs à plusieurs "
            "peut dupliquer les observations lors "
            "d'une jointure. Le grain analytique "
            "des tables doit être vérifié avant "
            "l'appariement."
        ),
    },

    {
        "case_id":
            "correlation",

        "finding": (
            "Une association statistique a été "
            "observée entre deux variables."
        ),

        "source": (
            "Une corrélation mesure la force "
            "et la direction d'une association "
            "entre deux variables. Elle ne permet "
            "pas à elle seule de conclure à une "
            "relation causale."
        ),
    },

    {
        "case_id":
            "sensor_drift",

        "finding": (
            "Les mesures d'un capteur s'écartent "
            "progressivement d'une référence."
        ),

        "source": (
            "La dérive d'un capteur correspond "
            "à une perte progressive d'étalonnage. "
            "Elle peut être détectée lorsque "
            "l'erreur par rapport à une référence "
            "stable augmente avec le temps."
        ),
    },

    {
        "case_id":
            "least_privilege",

        "finding": (
            "Certains comptes disposent de droits "
            "qui ne sont pas nécessaires."
        ),

        "source": (
            "Le principe du moindre privilège "
            "consiste à attribuer uniquement "
            "les permissions nécessaires à "
            "l'exécution d'une tâche."
        ),
    },
]


# ============================================================
# HIT BUILDER
# ============================================================

def build_test_hit(
    *,
    index: int,
    case_id: str,
    text: str,
) -> RagSearchHit:
    return RagSearchHit(
        rank=
            1,

        score=
            1.0,

        chunk_id=
            (
                "benchmark:"
                f"{case_id}:chunk:0001"
            ),

        document_id=
            (
                "benchmark:"
                f"{case_id}"
            ),

        filename=
            (
                f"{case_id}.txt"
            ),

        extension=
            ".txt",

        chunk_index=
            1,

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


# ============================================================
# HASH
# ============================================================

def cases_sha256() -> str:
    payload = json.dumps(
        CASES,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


    return hashlib.sha256(
        payload
    ).hexdigest()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            (
                "Evaluation artifact already exists. "
                "Refusing to overwrite:\n"
                f"{OUTPUT_PATH}"
            )
        )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "=" * 100
    )

    print(
        "DATALENS — RAG EXPLANATION BENCHMARK"
    )

    print(
        "=" * 100
    )

    print(
        "Benchmark:",
        BENCHMARK_VERSION,
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


    results: list[
        dict[
            str,
            Any,
        ]
    ] = []


    successful_cases = 0

    invalid_citations = 0

    invalid_evidence_quotes = 0

    numeric_violations = 0

    abstentions = 0

    validation_errors = 0


    for (
        index,
        case,
    ) in enumerate(
        CASES,
        start=1,
    ):
        hit = (
            build_test_hit(
                index=
                    index,

                case_id=
                    case[
                        "case_id"
                    ],

                text=
                    case[
                        "source"
                    ],
            )
        )


        try:
            result = (
                generate_grounded_explanation(
                    finding_text=
                        case[
                            "finding"
                        ],

                    accepted_hits=[
                        hit
                    ],
                )
            )


            if (
                result.status
                ==
                "abstained"
            ):
                abstentions += 1


                results.append(
                    {
                        "case_id":
                            case[
                                "case_id"
                            ],

                        "status":
                            "abstained",

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
                    "  status: abstained"
                )


                continue


            successful_cases += 1


            claims_payload = []


            for claim in result.claims:
                citation_valid = (
                    claim.citation.chunk_id
                    ==
                    hit.chunk_id
                )


                evidence_valid = (
                    " ".join(
                        claim.evidence_quote
                        .split()
                    )
                    in
                    " ".join(
                        hit.text
                        .split()
                    )
                )


                if not citation_valid:
                    invalid_citations += 1


                if not evidence_valid:
                    invalid_evidence_quotes += 1


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


            results.append(
                {
                    "case_id":
                        case[
                            "case_id"
                        ],

                    "status":
                        result.status,

                    "success":
                        True,

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

    case_count = len(
        CASES
    )


    success_rate = (
        successful_cases
        /
        case_count
    )


    success_rate_pass = (
        success_rate
        >=
        MIN_SUCCESS_RATE
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


    benchmark_pass = (
        success_rate_pass
        and
        citation_pass
        and
        evidence_pass
        and
        numeric_pass
    )


    metrics = {
        "case_count":
            case_count,

        "successful_cases":
            successful_cases,

        "abstentions":
            abstentions,

        "validation_errors":
            validation_errors,

        "success_rate":
            success_rate,

        "invalid_citations":
            invalid_citations,

        "invalid_evidence_quotes":
            invalid_evidence_quotes,

        "numeric_violations":
            numeric_violations,
    }


    gates = {
        "minimum_success_rate":
            MIN_SUCCESS_RATE,

        "maximum_invalid_citations":
            MAX_INVALID_CITATIONS,

        "maximum_invalid_evidence_quotes":
            MAX_INVALID_EVIDENCE_QUOTES,

        "maximum_numeric_violations":
            MAX_NUMERIC_VIOLATIONS,

        "success_rate_pass":
            success_rate_pass,

        "citation_pass":
            citation_pass,

        "evidence_pass":
            evidence_pass,

        "numeric_pass":
            numeric_pass,

        "benchmark_pass":
            benchmark_pass,
    }


    artifact = {
        "benchmark_version":
            BENCHMARK_VERSION,

        "explanation_rule_version":
            RAG_EXPLANATION_RULE_VERSION,

        "model":
            DEFAULT_EXPLANATION_MODEL,

        "cases_sha256":
            cases_sha256(),

        "metrics":
            metrics,

        "gates":
            gates,

        "cases":
            results,
    }


    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    artifact_sha256 = (
        hashlib.sha256(
            OUTPUT_PATH
            .read_bytes()
        )
        .hexdigest()
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        "Success rate:",
        round(
            success_rate,
            6,
        ),
    )

    print(
        "Abstentions:",
        abstentions,
    )

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
        "BENCHMARK PASS:",
        benchmark_pass,
    )

    print()

    print(
        "Artifact:",
        OUTPUT_PATH,
    )

    print(
        "SHA256:",
        artifact_sha256,
    )


if __name__ == "__main__":
    main()