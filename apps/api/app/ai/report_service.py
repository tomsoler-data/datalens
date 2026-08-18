from __future__ import annotations

from collections import (
    Counter,
)

from typing import (
    Any,
)

from app.ai.provider import (
    DEFAULT_MODEL,
)

from app.ai.report_provider import (
    call_semantic_candidate_model,
)

from app.ai.report_schemas import (
    SemanticAnalysisReport,
    SemanticCandidateAssessment,
    SemanticCandidateAssessmentDraft,
    SemanticLanguage,
    SemanticPriority,
    SemanticReason,
    SemanticReportFinding,
)

from app.reporting.unified_schemas import (
    ReportFinding,
    UnifiedAnalysisReport,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SEMANTIC_ATTEMPTS = 2

DEFAULT_CANDIDATE_LIMIT = 12

DEFAULT_MAIN_FINDING_LIMIT = 6

MAX_FINDINGS_PER_FAMILY = 2


# ============================================================
# CANDIDATE POOL
# ============================================================

def build_semantic_candidate_pool(
    report: UnifiedAnalysisReport,
    *,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[
    ReportFinding
]:
    by_id: dict[
        str,
        ReportFinding,
    ] = {}


    for finding in (
        report.main_findings
    ):
        by_id[
            finding.analysis_id
        ] = finding


    additional_sorted = sorted(
        report.additional_findings,
        key=lambda item:
            (
                item.interestingness_score,
                item.signal_score,
            ),
        reverse=True,
    )


    for finding in (
        additional_sorted
    ):
        if (
            finding.analysis_id
            in by_id
        ):
            continue


        by_id[
            finding.analysis_id
        ] = finding


    candidates = list(
        by_id.values()
    )


    if (
        len(
            candidates
        )
        <=
        candidate_limit
    ):
        return candidates


    main_ids = {
        finding.analysis_id
        for finding
        in report.main_findings
    }


    main_candidates = [
        finding
        for finding
        in candidates
        if (
            finding.analysis_id
            in main_ids
        )
    ]


    other_candidates = [
        finding
        for finding
        in candidates
        if (
            finding.analysis_id
            not in main_ids
        )
    ]


    other_candidates.sort(
        key=lambda item:
            (
                item.interestingness_score,
                item.signal_score,
            ),
        reverse=True,
    )


    remaining_slots = max(
        0,
        candidate_limit
        -
        len(
            main_candidates
        ),
    )


    return (
        main_candidates
        +
        other_candidates[
            :remaining_slots
        ]
    )


# ============================================================
# KEYS
# ============================================================

def build_candidate_key_map(
    candidates: list[
        ReportFinding
    ],
) -> dict[
    str,
    ReportFinding,
]:
    return {
        f"C{index:02d}":
            finding
        for index, finding
        in enumerate(
            candidates,
            start=1,
        )
    }


# ============================================================
# COMPACT EVIDENCE
# ============================================================

def compact_finding_evidence(
    *,
    candidate_key: str,
    finding: ReportFinding,
) -> dict[
    str,
    Any,
]:
    return {
        "candidate_key":
            candidate_key,

        "title":
            finding.title,

        "family":
            finding.family,

        "scope":
            finding.scope,

        "tier":
            finding.tier,

        "execution_status":
            finding.execution_status,

        "interestingness_score":
            finding.interestingness_score,

        "signal_score":
            finding.signal_score,

        "coverage_score":
            finding.coverage_score,

        "consistency_score":
            finding.consistency_score,

        "direction":
            finding.direction,

        "strength":
            finding.strength,

        "sample_size":
            finding.sample_size,

        "period_count":
            finding.period_count,

        "datasets":
            list(
                finding.datasets
            ),

        "reasons":
            list(
                finding.reasons
            ),

        "caveats":
            list(
                finding.caveats
            ),
    }


def build_semantic_context(
    report: UnifiedAnalysisReport,
    *,
    objective: str | None,
    language: SemanticLanguage,
    candidate_limit: int,
    main_finding_limit: int,
) -> tuple[
    dict[
        str,
        Any,
    ],
    list[
        ReportFinding
    ],
]:
    candidates = (
        build_semantic_candidate_pool(
            report,
            candidate_limit=
                candidate_limit,
        )
    )


    key_map = (
        build_candidate_key_map(
            candidates
        )
    )


    catalog = [
        compact_finding_evidence(
            candidate_key=
                candidate_key,

            finding=
                finding,
        )
        for candidate_key, finding
        in key_map.items()
    ]


    context = {
        "language":
            language,

        "objective":
            objective,

        "main_finding_limit":
            min(
                main_finding_limit,
                len(
                    candidates
                ),
            ),

        "candidate_catalog":
            catalog,
    }


    return (
        context,
        candidates,
    )


# ============================================================
# REASON NORMALIZATION
# ============================================================

def deduplicate_reasons(
    reasons: list[
        SemanticReason
    ],
) -> list[
    SemanticReason
]:
    result: list[
        SemanticReason
    ] = []


    seen = set()


    for reason in reasons:
        if reason in seen:
            continue


        seen.add(
            reason
        )


        result.append(
            reason
        )


    return result[
        :3
    ]


# ============================================================
# FALLBACK METADATA
# ============================================================

def fallback_priority_for(
    finding: ReportFinding,
) -> SemanticPriority:
    if (
        finding.tier
        ==
        "key_finding"
    ):
        return "high"


    if (
        finding.tier
        ==
        "supporting_finding"
    ):
        return "medium"


    return "low"


def fallback_reasons_for(
    finding: ReportFinding,
) -> list[
    SemanticReason
]:
    if (
        finding.scope
        ==
        "cross_dataset"
    ):
        return [
            "cross_dataset_relationship",
            "complementary_perspective",
        ]


    if (
        finding.family
        ==
        "time_series"
    ):
        return [
            "meaningful_trend",
        ]


    if (
        finding.family
        ==
        "derived_gap"
    ):
        return [
            "meaningful_gap",
        ]


    if (
        finding.family
        ==
        "group_comparison"
    ):
        return [
            "group_difference",
        ]


    return [
        "high_analytical_signal",
    ]


# ============================================================
# SINGLE CANDIDATE ASSESSMENT
# ============================================================

def assess_candidate(
    *,
    candidate_key: str,
    finding: ReportFinding,
    candidate_catalog: list[
        dict[
            str,
            Any,
        ]
    ],
    objective: str | None,
    model: str,
) -> SemanticCandidateAssessment:
    target = next(
        item
        for item
        in candidate_catalog
        if (
            item[
                "candidate_key"
            ]
            ==
            candidate_key
        )
    )


    for attempt in range(
        MAX_SEMANTIC_ATTEMPTS
    ):
        try:
            content = (
                call_semantic_candidate_model(
                    target_key=
                        candidate_key,

                    target=
                        target,

                    candidate_catalog=
                        candidate_catalog,

                    objective=
                        objective,

                    model=
                        model,

                    strict_retry=(
                        attempt > 0
                    ),
                )
            )


            draft = (
                SemanticCandidateAssessmentDraft
                .model_validate_json(
                    content
                )
            )


            reasons = (
                deduplicate_reasons(
                    list(
                        draft.reasons
                    )
                )
            )


            if not reasons:
                raise ValueError(
                    (
                        "Semantic assessment "
                        "contains no usable reason."
                    )
                )


            return (
                SemanticCandidateAssessment(
                    candidate_key=
                        candidate_key,

                    analysis_id=
                        finding.analysis_id,

                    semantic_relevance=
                        draft.semantic_relevance,

                    semantic_priority=
                        draft.semantic_priority,

                    semantic_reasons=
                        reasons,

                    assessment_source=
                        "llm",
                )
            )


        except Exception:
            continue


    return (
        SemanticCandidateAssessment(
            candidate_key=
                candidate_key,

            analysis_id=
                finding.analysis_id,

            semantic_relevance=
                None,

            semantic_priority=
                fallback_priority_for(
                    finding
                ),

            semantic_reasons=
                fallback_reasons_for(
                    finding
                ),

            assessment_source=
                "fallback",
        )
    )


# ============================================================
# ALL CANDIDATE ASSESSMENTS
# ============================================================

def assess_candidates(
    *,
    candidates: list[
        ReportFinding
    ],
    objective: str | None,
    model: str,
) -> list[
    SemanticCandidateAssessment
]:
    key_map = (
        build_candidate_key_map(
            candidates
        )
    )


    catalog = [
        compact_finding_evidence(
            candidate_key=
                candidate_key,

            finding=
                finding,
        )
        for candidate_key, finding
        in key_map.items()
    ]


    assessments: list[
        SemanticCandidateAssessment
    ] = []


    for candidate_key, finding in (
        key_map.items()
    ):
        assessment = (
            assess_candidate(
                candidate_key=
                    candidate_key,

                finding=
                    finding,

                candidate_catalog=
                    catalog,

                objective=
                    objective,

                model=
                    model,
            )
        )


        assessments.append(
            assessment
        )


    return assessments


# ============================================================
# EFFECTIVE SELECTION SCORE
# ============================================================

def effective_semantic_score(
    *,
    assessment: SemanticCandidateAssessment,
    finding: ReportFinding,
) -> float:
    """
    When the LLM assessment exists, use it.

    Otherwise use the deterministic analytical score
    only as a safe fallback for portfolio selection.
    """

    if (
        assessment.semantic_relevance
        is not None
    ):
        return float(
            assessment.semantic_relevance
        )


    return float(
        finding.interestingness_score
    )


# ============================================================
# PYTHON PORTFOLIO SELECTION
# ============================================================

def select_semantic_findings(
    *,
    candidates: list[
        ReportFinding
    ],
    assessments: list[
        SemanticCandidateAssessment
    ],
    main_finding_limit: int,
) -> list[
    tuple[
        ReportFinding,
        SemanticCandidateAssessment,
    ]
]:
    finding_map = {
        finding.analysis_id:
            finding
        for finding
        in candidates
    }


    assessment_map = {
        assessment.analysis_id:
            assessment
        for assessment
        in assessments
    }


    ranked = [
        (
            finding,
            assessment_map[
                finding.analysis_id
            ],
        )
        for finding
        in candidates
    ]


    ranked.sort(
        key=lambda item:
            (
                effective_semantic_score(
                    assessment=
                        item[
                            1
                        ],

                    finding=
                        item[
                            0
                        ],
                ),

                item[
                    0
                ].interestingness_score,

                item[
                    0
                ].signal_score,
            ),
        reverse=True,
    )


    limit = min(
        main_finding_limit,
        len(
            ranked
        ),
    )


    selected: list[
        tuple[
            ReportFinding,
            SemanticCandidateAssessment,
        ]
    ] = []


    selected_ids: set[
        str
    ] = set()


    family_counts: Counter[
        str
    ] = Counter()


    # ========================================================
    # PASS 1
    #
    # Capture the strongest representative of each
    # available analytical family.
    # ========================================================

    available_families = []


    for finding, assessment in ranked:
        if (
            finding.family
            not in available_families
        ):
            available_families.append(
                finding.family
            )


    for family in available_families:
        if (
            len(
                selected
            )
            >=
            limit
        ):
            break


        candidate = next(
            (
                item
                for item
                in ranked
                if (
                    item[
                        0
                    ].family
                    ==
                    family
                    and
                    item[
                        0
                    ].analysis_id
                    not in selected_ids
                )
            ),
            None,
        )


        if candidate is None:
            continue


        finding, assessment = (
            candidate
        )


        selected.append(
            candidate
        )


        selected_ids.add(
            finding.analysis_id
        )


        family_counts[
            finding.family
        ] += 1


    # ========================================================
    # PASS 2
    #
    # Fill remaining positions by semantic strength,
    # with a maximum of two findings per family.
    # ========================================================

    for finding, assessment in ranked:
        if (
            len(
                selected
            )
            >=
            limit
        ):
            break


        if (
            finding.analysis_id
            in selected_ids
        ):
            continue


        if (
            family_counts[
                finding.family
            ]
            >=
            MAX_FINDINGS_PER_FAMILY
        ):
            continue


        selected.append(
            (
                finding,
                assessment,
            )
        )


        selected_ids.add(
            finding.analysis_id
        )


        family_counts[
            finding.family
        ] += 1


    # ========================================================
    # FINAL ORDER
    #
    # Report order reflects semantic relevance after
    # diversity constraints have been satisfied.
    # ========================================================

    selected.sort(
        key=lambda item:
            (
                effective_semantic_score(
                    assessment=
                        item[
                            1
                        ],

                    finding=
                        item[
                            0
                        ],
                ),

                item[
                    0
                ].interestingness_score,
            ),
        reverse=True,
    )


    return selected


# ============================================================
# INTERPRETATION
# ============================================================

def interpretation_for(
    finding: ReportFinding,
    *,
    language: SemanticLanguage,
) -> str:
    if language == "en":
        if (
            finding.family
            ==
            "time_series"
        ):
            if (
                finding.direction
                ==
                "positive"
            ):
                return (
                    "The observed measure follows "
                    "an upward temporal pattern."
                )


            if (
                finding.direction
                ==
                "negative"
            ):
                return (
                    "The observed measure follows "
                    "a downward temporal pattern."
                )


            return (
                "The observed measure shows a "
                "temporal pattern worth examining."
            )


        if (
            finding.family
            ==
            "group_comparison"
        ):
            return (
                "The observed groups show "
                "descriptive differences."
            )


        if (
            finding.family
            ==
            "derived_gap"
        ):
            return (
                "The two related measures show "
                "an observable gap."
            )


        if (
            finding.family
            ==
            "quantitative_association"
        ):
            if (
                finding.direction
                ==
                "positive"
            ):
                if (
                    finding.strength
                    in {
                        "strong",
                        "moderately_strong",
                    }
                ):
                    return (
                        "The two measures show a "
                        "notable positive descriptive "
                        "association."
                    )


                return (
                    "The two measures show a positive "
                    "descriptive association."
                )


            if (
                finding.direction
                ==
                "negative"
            ):
                if (
                    finding.strength
                    in {
                        "strong",
                        "moderately_strong",
                    }
                ):
                    return (
                        "The two measures show a "
                        "notable negative descriptive "
                        "association."
                    )


                return (
                    "The two measures show a negative "
                    "descriptive association."
                )


            return (
                "The two measures show a descriptive "
                "association worth examining."
            )


        if (
            finding.family
            ==
            "categorical_association"
        ):
            return (
                "The categorical variables show "
                "an observable descriptive "
                "association."
            )


        return (
            "The analysis produced a descriptive "
            "signal worth examining."
        )


    # ========================================================
    # FRENCH
    # ========================================================

    if (
        finding.family
        ==
        "time_series"
    ):
        if (
            finding.direction
            ==
            "positive"
        ):
            return (
                "La mesure observée présente une "
                "tendance temporelle à la hausse."
            )


        if (
            finding.direction
            ==
            "negative"
        ):
            return (
                "La mesure observée présente une "
                "tendance temporelle à la baisse."
            )


        return (
            "La mesure observée présente une "
            "évolution temporelle qui mérite "
            "d'être examinée."
        )


    if (
        finding.family
        ==
        "group_comparison"
    ):
        return (
            "Les groupes observés présentent des "
            "différences descriptives."
        )


    if (
        finding.family
        ==
        "derived_gap"
    ):
        return (
            "Les deux mesures liées présentent "
            "un écart observable."
        )


    if (
        finding.family
        ==
        "quantitative_association"
    ):
        if (
            finding.direction
            ==
            "positive"
        ):
            if (
                finding.strength
                in {
                    "strong",
                    "moderately_strong",
                }
            ):
                return (
                    "Les deux mesures présentent une "
                    "association descriptive positive "
                    "notable."
                )


            return (
                "Les deux mesures présentent une "
                "association descriptive positive."
            )


        if (
            finding.direction
            ==
            "negative"
        ):
            if (
                finding.strength
                in {
                    "strong",
                    "moderately_strong",
                }
            ):
                return (
                    "Les deux mesures présentent une "
                    "association descriptive négative "
                    "notable."
                )


            return (
                "Les deux mesures présentent une "
                "association descriptive négative."
            )


        return (
            "Les deux mesures présentent une "
            "association descriptive qui mérite "
            "d'être examinée."
        )


    if (
        finding.family
        ==
        "categorical_association"
    ):
        return (
            "Les variables catégorielles présentent "
            "une association descriptive observable."
        )


    return (
        "L'analyse produit un signal descriptif "
        "qui mérite d'être examiné."
    )


# ============================================================
# WHY IT MATTERS
# ============================================================

def why_it_matters_for(
    finding: ReportFinding,
    *,
    semantic_reasons: list[
        SemanticReason
    ],
    language: SemanticLanguage,
) -> str:
    reasons = set(
        semantic_reasons
    )


    if language == "en":
        if (
            "conceptual_redundancy"
            in reasons
        ):
            return (
                "This result is analytically valid, "
                "but its interpretation overlaps with "
                "closely related measures."
            )


        if (
            "cross_dataset_relationship"
            in reasons
            or
            finding.scope
            ==
            "cross_dataset"
        ):
            return (
                "This result connects information "
                "from distinct datasets and adds a "
                "complementary analytical perspective."
            )


        if (
            "meaningful_gap"
            in reasons
        ):
            return (
                "This result highlights a difference "
                "between related measures that may "
                "deserve closer examination."
            )


        if (
            "meaningful_trend"
            in reasons
        ):
            return (
                "This result contributes a temporal "
                "perspective to the overall analysis."
            )


        if (
            "group_difference"
            in reasons
        ):
            return (
                "This result helps compare how the "
                "observed measure differs across "
                "groups."
            )


        if (
            "distinct_concepts"
            in reasons
        ):
            return (
                "This result connects conceptually "
                "distinct measures and adds a useful "
                "analytical perspective."
            )


        return (
            "This result contributes a distinct "
            "analytical perspective to the report."
        )


    # ========================================================
    # FRENCH
    # ========================================================

    if (
        "conceptual_redundancy"
        in reasons
    ):
        return (
            "Ce résultat est analytiquement valide, "
            "mais son interprétation recouvre en "
            "partie celle de mesures proches."
        )


    if (
        "cross_dataset_relationship"
        in reasons
        or
        finding.scope
        ==
        "cross_dataset"
    ):
        return (
            "Ce résultat rapproche des informations "
            "issues de jeux de données distincts et "
            "apporte un angle analytique "
            "complémentaire."
        )


    if (
        "meaningful_gap"
        in reasons
    ):
        return (
            "Ce résultat met en évidence une "
            "différence entre deux mesures liées qui "
            "mérite d'être examinée plus précisément."
        )


    if (
        "meaningful_trend"
        in reasons
    ):
        return (
            "Ce résultat apporte une lecture "
            "temporelle complémentaire à l'analyse "
            "globale."
        )


    if (
        "group_difference"
        in reasons
    ):
        return (
            "Ce résultat permet de comparer les "
            "différences observées entre les groupes."
        )


    if (
        "distinct_concepts"
        in reasons
    ):
        return (
            "Ce résultat rapproche des mesures "
            "conceptuellement distinctes et apporte "
            "un angle analytique utile."
        )


    return (
        "Ce résultat apporte un angle analytique "
        "distinct à la lecture globale du rapport."
    )


# ============================================================
# METHOD
# ============================================================

def method_explanation_for(
    finding: ReportFinding,
    *,
    language: SemanticLanguage,
) -> str:
    if language == "en":
        mapping = {
            "time_series":
                (
                    "Descriptive time-series analysis "
                    "used to examine how the measure "
                    "changes across observed periods."
                ),

            "group_comparison":
                (
                    "Descriptive comparison of the "
                    "measure across observed groups."
                ),

            "derived_gap":
                (
                    "Deterministic comparison of two "
                    "related measures through their "
                    "observed gap."
                ),

            "quantitative_association":
                (
                    "Quantitative association analysis "
                    "adapted to the observational "
                    "structure of the data."
                ),

            "categorical_association":
                (
                    "Descriptive association analysis "
                    "between categorical variables."
                ),
        }


        return mapping.get(
            finding.family,
            (
                "Deterministic descriptive analysis "
                "performed by the DataLens analytical "
                "engine."
            ),
        )


    mapping = {
        "time_series":
            (
                "Analyse temporelle descriptive "
                "permettant d'examiner l'évolution "
                "de la mesure au fil des périodes "
                "observées."
            ),

        "group_comparison":
            (
                "Comparaison descriptive de la "
                "mesure entre les groupes observés."
            ),

        "derived_gap":
            (
                "Comparaison déterministe de deux "
                "mesures liées à travers l'écart "
                "observé entre elles."
            ),

        "quantitative_association":
            (
                "Analyse d'association quantitative "
                "adaptée à la structure des "
                "observations."
            ),

        "categorical_association":
            (
                "Analyse descriptive de "
                "l'association entre variables "
                "catégorielles."
            ),
    }


    return mapping.get(
        finding.family,
        (
            "Analyse descriptive déterministe "
            "réalisée par le moteur analytique "
            "DataLens."
        ),
    )


# ============================================================
# REPORT MODE
# ============================================================

def determine_generation_mode(
    assessments: list[
        SemanticCandidateAssessment
    ],
) -> str:
    llm_count = sum(
        assessment.assessment_source
        ==
        "llm"
        for assessment
        in assessments
    )


    if llm_count == 0:
        return (
            "deterministic_fallback"
        )


    if (
        llm_count
        ==
        len(
            assessments
        )
    ):
        return "llm"


    return "hybrid"


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def build_executive_summary(
    *,
    findings: list[
        SemanticReportFinding
    ],
    language: SemanticLanguage,
    generation_mode: str,
) -> str:
    if language == "en":
        if (
            generation_mode
            ==
            "llm"
        ):
            return (
                "The main findings were selected by "
                "combining validated semantic "
                "assessments with deterministic "
                "portfolio constraints. All numerical "
                "evidence remains unchanged."
            )


        if (
            generation_mode
            ==
            "hybrid"
        ):
            return (
                "The main findings combine validated "
                "semantic assessments with "
                "deterministic fallback scores where "
                "needed. All analytical evidence "
                "remains unchanged."
            )


        return (
            "The report retains the deterministic "
            "selection because semantic assessments "
            "could not be validated."
        )


    if (
        generation_mode
        ==
        "llm"
    ):
        return (
            "Les principaux résultats ont été "
            "hiérarchisés à partir d'évaluations "
            "sémantiques validées, puis sélectionnés "
            "par Python selon des règles de diversité. "
            "Les résultats calculés restent inchangés."
        )


    if (
        generation_mode
        ==
        "hybrid"
    ):
        return (
            "Les principaux résultats combinent des "
            "évaluations sémantiques validées et des "
            "scores déterministes de secours lorsque "
            "nécessaire. Les résultats calculés "
            "restent inchangés."
        )


    return (
        "Le rapport conserve la sélection "
        "déterministe car les évaluations "
        "sémantiques n'ont pas pu être validées."
    )


# ============================================================
# DETERMINISTIC FALLBACK REPORT
# ============================================================

def build_deterministic_fallback(
    *,
    report: UnifiedAnalysisReport,
    candidates: list[
        ReportFinding
    ],
    assessments: list[
        SemanticCandidateAssessment
    ],
    language: SemanticLanguage,
    objective: str | None,
    model: str,
    main_finding_limit: int,
) -> SemanticAnalysisReport:
    candidate_ids = {
        finding.analysis_id
        for finding
        in candidates
    }


    selected = [
        finding
        for finding
        in report.main_findings
        if (
            finding.analysis_id
            in candidate_ids
        )
    ][
        :main_finding_limit
    ]


    assessment_map = {
        assessment.analysis_id:
            assessment
        for assessment
        in assessments
    }


    semantic_findings: list[
        SemanticReportFinding
    ] = []


    for finding in selected:
        assessment = (
            assessment_map.get(
                finding.analysis_id
            )
        )


        reasons = (
            assessment.semantic_reasons
            if assessment is not None
            else fallback_reasons_for(
                finding
            )
        )


        priority = (
            assessment.semantic_priority
            if assessment is not None
            else fallback_priority_for(
                finding
            )
        )


        semantic_findings.append(
            SemanticReportFinding(
                analysis_id=
                    finding.analysis_id,

                semantic_relevance=
                    None,

                semantic_priority=
                    priority,

                semantic_reasons=
                    reasons,

                interpretation=
                    interpretation_for(
                        finding,
                        language=
                            language,
                    ),

                why_it_matters=
                    why_it_matters_for(
                        finding,
                        semantic_reasons=
                            reasons,
                        language=
                            language,
                    ),

                method_explanation=
                    method_explanation_for(
                        finding,
                        language=
                            language,
                    ),

                attention_points=list(
                    finding.caveats[
                        :4
                    ]
                ),

                source_finding=
                    finding,
            )
        )


    selected_ids = {
        finding.analysis_id
        for finding
        in selected
    }


    return SemanticAnalysisReport(
        language=
            language,

        objective=
            objective,

        model=
            model,

        generation_mode=
            "deterministic_fallback",

        executive_summary=
            build_executive_summary(
                findings=
                    semantic_findings,

                language=
                    language,

                generation_mode=
                    "deterministic_fallback",
            ),

        main_findings=
            semantic_findings,

        candidate_assessments=
            assessments,

        candidate_analysis_ids=[
            finding.analysis_id
            for finding
            in candidates
        ],

        not_selected_analysis_ids=[
            finding.analysis_id
            for finding
            in candidates
            if (
                finding.analysis_id
                not in selected_ids
            )
        ],

        source_report=
            report,

        semantic_rule_version=
            "semantic_report_v0.3",
    )


# ============================================================
# COMPOSE FINAL SEMANTIC REPORT
# ============================================================

def compose_semantic_report(
    *,
    report: UnifiedAnalysisReport,
    candidates: list[
        ReportFinding
    ],
    assessments: list[
        SemanticCandidateAssessment
    ],
    selected: list[
        tuple[
            ReportFinding,
            SemanticCandidateAssessment,
        ]
    ],
    language: SemanticLanguage,
    objective: str | None,
    model: str,
    generation_mode: str,
) -> SemanticAnalysisReport:
    semantic_findings: list[
        SemanticReportFinding
    ] = []


    selected_ids: set[
        str
    ] = set()


    for finding, assessment in selected:
        selected_ids.add(
            finding.analysis_id
        )


        semantic_findings.append(
            SemanticReportFinding(
                analysis_id=
                    finding.analysis_id,

                semantic_relevance=
                    assessment.semantic_relevance,

                semantic_priority=
                    assessment.semantic_priority,

                semantic_reasons=list(
                    assessment.semantic_reasons
                ),

                interpretation=
                    interpretation_for(
                        finding,
                        language=
                            language,
                    ),

                why_it_matters=
                    why_it_matters_for(
                        finding,
                        semantic_reasons=
                            list(
                                assessment.semantic_reasons
                            ),
                        language=
                            language,
                    ),

                method_explanation=
                    method_explanation_for(
                        finding,
                        language=
                            language,
                    ),

                attention_points=list(
                    finding.caveats[
                        :4
                    ]
                ),

                source_finding=
                    finding,
            )
        )


    return SemanticAnalysisReport(
        language=
            language,

        objective=
            objective,

        model=
            model,

        generation_mode=
            generation_mode,

        executive_summary=
            build_executive_summary(
                findings=
                    semantic_findings,

                language=
                    language,

                generation_mode=
                    generation_mode,
            ),

        main_findings=
            semantic_findings,

        candidate_assessments=
            assessments,

        candidate_analysis_ids=[
            finding.analysis_id
            for finding
            in candidates
        ],

        not_selected_analysis_ids=[
            finding.analysis_id
            for finding
            in candidates
            if (
                finding.analysis_id
                not in selected_ids
            )
        ],

        source_report=
            report,

        semantic_rule_version=
            "semantic_report_v0.3",
    )


# ============================================================
# PUBLIC SERVICE
# ============================================================

def enrich_analysis_report(
    report: UnifiedAnalysisReport,
    *,
    objective: str | None = None,
    language: SemanticLanguage = "fr",
    model: str = DEFAULT_MODEL,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    main_finding_limit: int = DEFAULT_MAIN_FINDING_LIMIT,
) -> SemanticAnalysisReport:
    """
    Semantic Report v0.3.

    The local LLM independently assesses each
    defensible candidate.

    Python performs the final portfolio selection.

    LLM:
    - semantic relevance
    - semantic priority
    - semantic reason codes

    Python:
    - exact analysis identity
    - analytical calculations
    - final selection
    - diversity constraints
    - interpretation
    - methodology
    - caveats
    - report composition
    """

    candidates = (
        build_semantic_candidate_pool(
            report,
            candidate_limit=
                candidate_limit,
        )
    )


    if not candidates:
        return (
            build_deterministic_fallback(
                report=
                    report,

                candidates=
                    candidates,

                assessments=
                    [],

                language=
                    language,

                objective=
                    objective,

                model=
                    model,

                main_finding_limit=
                    main_finding_limit,
            )
        )


    assessments = (
        assess_candidates(
            candidates=
                candidates,

            objective=
                objective,

            model=
                model,
        )
    )


    generation_mode = (
        determine_generation_mode(
            assessments
        )
    )


    if (
        generation_mode
        ==
        "deterministic_fallback"
    ):
        return (
            build_deterministic_fallback(
                report=
                    report,

                candidates=
                    candidates,

                assessments=
                    assessments,

                language=
                    language,

                objective=
                    objective,

                model=
                    model,

                main_finding_limit=
                    main_finding_limit,
            )
        )


    selected = (
        select_semantic_findings(
            candidates=
                candidates,

            assessments=
                assessments,

            main_finding_limit=
                main_finding_limit,
        )
    )


    return (
        compose_semantic_report(
            report=
                report,

            candidates=
                candidates,

            assessments=
                assessments,

            selected=
                selected,

            language=
                language,

            objective=
                objective,

            model=
                model,

            generation_mode=
                generation_mode,
        )
    )
