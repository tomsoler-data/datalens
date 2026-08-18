from app.ingestion.schemas import (
    MultiDatasetIngestion,
)

from app.planning.schemas import (
    AnalysisPlanReport,
)

from app.reporting.schemas import (
    AnalysisReport,
    ReportAnalysis,
    ReportDataSuggestion,
    ReportDataset,
    ReportRelationship,
    ReportVariable,
)


def build_executive_summary(
    *,
    ingestion: MultiDatasetIngestion,
    plan: AnalysisPlanReport,
) -> list[
    str
]:
    analysis_count = len(
        plan.recommended_analyses
    )

    executable_count = sum(
        1
        for analysis
        in plan.recommended_analyses
        if (
            analysis.readiness
            ==
            "executable_now"
        )
    )

    high_priority_count = sum(
        1
        for analysis
        in plan.recommended_analyses
        if (
            analysis.priority_score
            >=
            80
        )
    )

    relationship_count = len(
        plan.cross_dataset_opportunities
    )

    suggestion_count = len(
        plan.additional_data_suggestions
    )


    summary = [
        (
            f"{ingestion.dataset_count} "
            "fichier(s) de données ont été "
            "analysés, représentant "
            f"{ingestion.total_rows:,} lignes."
        ),

        (
            f"{analysis_count} piste(s) "
            "d'analyse ont été retenues "
            "par le moteur de planification."
        ),

        (
            f"{high_priority_count} analyse(s) "
            "présentent actuellement une "
            "priorité élevée selon les règles "
            "déterministes du moteur."
        ),

        (
            f"{executable_count} analyse(s) "
            "peuvent déjà être exécutées "
            "par les moteurs statistiques "
            "actuellement disponibles."
        ),

        (
            f"{relationship_count} relation(s) "
            "potentielles entre fichiers ont "
            "été identifiées."
        ),
    ]


    if (
        suggestion_count
        >
        0
    ):
        summary.append(
            (
                f"{suggestion_count} catégorie(s) "
                "de données complémentaires "
                "pourraient enrichir de futures "
                "analyses."
            )
        )


    if plan.objective:
        summary.append(
            (
                "L'objectif fourni par "
                "l'utilisateur a été pris en "
                "compte pour prioriser les "
                "analyses."
            )
        )

    else:
        summary.append(
            (
                "Aucun objectif spécifique "
                "n'a été fourni : le rapport "
                "correspond à une exploration "
                "automatique des données."
            )
        )


    return summary


def compose_analysis_report(
    *,
    ingestion: MultiDatasetIngestion,
    plan: AnalysisPlanReport,
) -> AnalysisReport:
    datasets = [
        ReportDataset(
            dataset_id=
                dataset.dataset_id,

            filename=
                dataset.filename,

            row_count=
                dataset.row_count,

            column_count=
                dataset.column_count,

            memory_bytes=
                dataset.memory_bytes,
        )
        for dataset
        in ingestion.datasets
    ]


    analyses = [
        ReportAnalysis(
            analysis_id=
                analysis.analysis_id,

            dataset_id=
                analysis.dataset_id,

            dataset_filename=
                analysis.dataset_filename,

            title=
                analysis.title,

            family=
                analysis.family,

            priority_score=
                analysis.priority_score,

            readiness=
                analysis.readiness,

            chart_type=
                analysis.chart_type,

            statistical_strategy=
                analysis.statistical_strategy,

            variables=[
                ReportVariable(
                    column=
                        variable.column,

                    role=
                        variable.role,

                    analysis_kind=
                        variable.analysis_kind,
                )
                for variable
                in analysis.variables
            ],

            reasons=list(
                analysis.reasons
            ),

            limitations=list(
                analysis.limitations
            ),
        )
        for analysis
        in plan.recommended_analyses
    ]


    relationships = [
        ReportRelationship(
            opportunity_id=
                relationship.opportunity_id,

            dataset_filenames=list(
                relationship.dataset_filenames
            ),

            shared_columns=list(
                relationship.shared_columns
            ),

            reason=
                relationship.reason,

            requires_relationship_validation=(
                relationship
                .requires_relationship_validation
            ),
        )
        for relationship
        in plan.cross_dataset_opportunities
    ]


    suggestions = [
        ReportDataSuggestion(
            suggestion_id=
                suggestion.suggestion_id,

            title=
                suggestion.title,

            priority=
                suggestion.priority,

            rationale=
                suggestion.rationale,

            example_fields=list(
                suggestion.example_fields
            ),

            required_for_current_analysis=(
                suggestion
                .required_for_current_analysis
            ),
        )
        for suggestion
        in plan.additional_data_suggestions
    ]


    return AnalysisReport(
        title=(
            "Rapport d'analyse DataLens"
        ),

        objective=
            plan.objective,

        dataset_count=
            ingestion.dataset_count,

        total_rows=
            ingestion.total_rows,

        datasets=
            datasets,

        executive_summary=(
            build_executive_summary(
                ingestion=
                    ingestion,

                plan=
                    plan,
            )
        ),

        analyses=
            analyses,

        relationships=
            relationships,

        additional_data_suggestions=
            suggestions,

        methodology_notes=[
            *plan.planner_notes,

            (
                "Les analyses proposées sont "
                "des pistes analytiques. "
                "Une proposition ne constitue "
                "pas à elle seule une preuve "
                "statistique ou causale."
            ),

            (
                "Les calculs statistiques sont "
                "réalisés par les moteurs Python "
                "déterministes lorsqu'un moteur "
                "d'exécution compatible est "
                "disponible."
            ),

            (
                "Les relations entre fichiers "
                "doivent être validées en termes "
                "de cardinalité, de grain et de "
                "couverture avant toute jointure."
            ),
        ],
    )