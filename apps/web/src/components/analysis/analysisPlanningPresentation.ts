/*
 * DATALENS_ANALYSIS_PLANNER_PRODUCT_LANGUAGE_V0_1
 *
 * Business-facing planner copy describes product roles:
 * the local model interprets when necessary and the
 * deterministic engine validates before execution.
 *
 * Internal implementation identifiers remain unchanged.
 */


export function familyLabel(
  family:
    string |
    null |
    undefined
): string {
  if (
    !family
  ) {
    return "Analyse";
  }


  switch (
    family
  ) {
    case "time_series":
      return "Évolution temporelle";

    case "group_comparison":
      return "Comparaison de groupes";

    case "quantitative_association":
      return "Association quantitative";

    case "categorical_association":
      return "Association catégorielle";

    case "aggregate_breakdown":
      return "Répartition";

    case "derived_gap":
      return "Écart calculé";

    case "distribution":
      return "Distribution";

    case "geographic_comparison":
      return "Comparaison géographique";

    case "data_quality":
      return "Qualité des données";

    case "descriptive_metric":
      return "Indicateur descriptif";

    case "aggregation":
      return "Agrégation";

    case "ranking":
      return "Classement";

    case "categorical_breakdown":
      return "Répartition catégorielle";

    case "inequality":
      return "Concentration";

    default:
      return family
        .replace(
          /_/g,
          " "
        );
  }
}


export function isDeterministicPlannerModel(
  model:
    string |
    null |
    undefined
): boolean {
  return Boolean(
    model
      ?.trim()
      .startsWith(
        "python:"
      )
  );
}


export function plannerEngineLabel(
  model:
    string |
    null |
    undefined
): string {
  if (
    !model
  ) {
    return "DataLens";
  }


  if (
    isDeterministicPlannerModel(
      model
    )
  ) {
    return "Moteur déterministe";
  }


  if (
    model
      .toLowerCase()
      .includes(
        "gemma"
      )
  ) {
    return "Modèle local";
  }


  return model;
}


export function plannerUiCopy(
  model:
    string |
    null |
    undefined
) {
  if (
    isDeterministicPlannerModel(
      model
    )
  ) {
    return {
      eyebrow:
        "Plan analytique · déterministe",

      title:
        "Plan construit par DataLens",

      description:
        (
          "DataLens a reconnu une demande analytique générique. " +
          "Le moteur déterministe a sélectionné les variables compatibles " +
          "depuis le catalogue analytique validé, sans demander au modèle " +
          "local d’inventer le périmètre."
        ),

      details:
        (
          "Ce chemin ne nécessite pas d’interprétation par le modèle local. " +
          "Les appels d’outils restent contrôlés et le moteur déterministe " +
          "vérifie les arguments avant tout calcul."
        ),
    };
  }


  if (
    model
  ) {
    return {
      eyebrow:
        "Planification · modèle local",

      title:
        "Plan proposé par le modèle local",

      description:
        (
          "Le modèle local traduit votre demande en contrat analytique. " +
          "Le moteur déterministe vérifie ensuite le dataset, les colonnes, " +
          "leurs rôles et les invariants avant toute exécution."
        ),

      details:
        (
          "La planification utilise le modèle local lorsque la demande " +
          "nécessite une interprétation sémantique. Les appels d’outils " +
          "sont également contrôlés avant le calcul déterministe."
        ),
    };
  }


  return {
    eyebrow:
      "Planification · locale",

    title:
      "Préparer le plan analytique",

    description:
      (
        "DataLens choisit automatiquement le chemin le plus sûr : " +
        "résolution déterministe lorsque l’intention est générique et " +
        "suffisamment claire, ou modèle local lorsqu’une interprétation " +
        "sémantique est nécessaire."
      ),

    details:
      (
        "Dans tous les cas, le moteur déterministe reste l’autorité " +
        "de validation avant l’exécution statistique."
      ),
  };
}