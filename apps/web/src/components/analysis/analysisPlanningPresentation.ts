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
    return "Python déterministe";
  }


  if (
    model
      .toLowerCase()
      .includes(
        "gemma"
      )
  ) {
    return "Gemma · IA locale";
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
          "Python a sélectionné les variables compatibles depuis " +
          "le catalogue analytique validé, sans demander au LLM " +
          "d’inventer le périmètre."
        ),

      details:
        (
          "Ce chemin ne nécessite pas de génération LLM pour la planification. " +
          "Le tool calling local reste contrôlé et Python vérifie les arguments " +
          "avant tout calcul."
        ),
    };
  }


  if (
    model
  ) {
    return {
      eyebrow:
        "AI Planner · local",

      title:
        "Plan proposé par l’IA locale",

      description:
        (
          "Le modèle local traduit votre demande en contrat analytique. " +
          "Python vérifie ensuite le dataset, les colonnes, leurs rôles " +
          "et les invariants avant toute exécution."
        ),

      details:
        (
          "La planification utilise le modèle local lorsque la demande " +
          "nécessite une interprétation sémantique. Le function calling est " +
          "également contrôlé avant le calcul Python."
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
        "suffisamment claire, ou planner IA local lorsqu’une interprétation " +
        "sémantique est nécessaire."
      ),

    details:
      (
        "Dans tous les cas, Python reste l’autorité de validation avant " +
        "l’exécution statistique."
      ),
  };
}
