import styles from "./PrioritizationAuditPanel.module.css";


export type PrioritizationDecisionKind =
  | "selected"
  | "deferred"
  | "rejected";


export type PrioritizationReasonCode =
  | "selected_by_priority"
  | "quality_guard"
  | "not_executable_now"
  | "identifier_misuse"
  | "record_label_dimension"
  | "fragmented_group_dimension"
  | "sparse_categorical_structure"
  | "priority_below_threshold"
  | "family_budget_exhausted"
  | "variable_budget_exhausted"
  | "global_budget_exhausted";


export type AnalysisPrioritizationDecisionView = {
  analysis_id:
    string;

  family:
    string;

  title:
    string;

  original_priority_score:
    number;

  execution_priority_score:
    number;

  decision:
    PrioritizationDecisionKind;

  reason_code:
    PrioritizationReasonCode;

  reasons:
    string[];

  variable_keys:
    string[];
};


export type PrioritizationAuditView = {
  discovered_count:
    number;

  selected_for_execution_count:
    number;

  deferred_count:
    number;

  rejected_count:
    number;

  decision_counts:
    Record<string, number>;

  reason_counts:
    Record<string, number>;

  non_execution_reason_counts:
    Record<string, number>;

  family_selected_counts:
    Record<string, number>;

  decisions:
    AnalysisPrioritizationDecisionView[];

  prioritization_rule_version:
    string;

  analytical_value_guard_rule_version:
    string;

  audit_rule_version:
    string;
};


type Props = {
  audit:
    PrioritizationAuditView;
};


type ReasonPresentation = {
  label:
    string;

  description:
    string;
};


const REASON_PRESENTATION:
  Record<
    PrioritizationReasonCode,
    ReasonPresentation
  > = {
    selected_by_priority: {
      label:
        "Retenue par priorité",
      description:
        "Le candidat respecte les garde-fous et reste dans les budgets de diversité et de calcul.",
    },

    quality_guard: {
      label:
        "Contrôle qualité conservé",
      description:
        "Le contrôle qualité est conservé comme preuve méthodologique.",
    },

    not_executable_now: {
      label:
        "Pas exécutable immédiatement",
      description:
        "La Discovery a identifié le candidat, mais son contrat n'est pas encore suffisamment sûr pour une exécution automatique.",
    },

    identifier_misuse: {
      label:
        "Identifiant utilisé comme variable analytique",
      description:
        "Une colonne identifiante est utilisée dans un rôle analytique qui pourrait produire un résultat trompeur.",
    },

    record_label_dimension: {
      label:
        "Dimension de libellé d'entité",
      description:
        "La dimension ressemble à un libellé ligne par ligne et sa forte cardinalité fragmente trop l'analyse automatique.",
    },

    fragmented_group_dimension: {
      label:
        "Dimension trop fragmentée",
      description:
        "Le nombre de groupes est trop élevé par rapport au nombre d'observations disponibles.",
    },

    sparse_categorical_structure: {
      label:
        "Structure catégorielle trop creuse",
      description:
        "Le croisement catégoriel contient trop peu d'observations par cellule pour être prioritaire automatiquement.",
    },

    priority_below_threshold: {
      label:
        "Priorité insuffisante",
      description:
        "Le score déterministe est inférieur au seuil retenu pour consommer le budget d'exécution exploratoire.",
    },

    family_budget_exhausted: {
      label:
        "Budget de famille atteint",
      description:
        "Assez d'analyses de cette famille ont déjà été retenues pour préserver la diversité du résultat.",
    },

    variable_budget_exhausted: {
      label:
        "Variable déjà suffisamment représentée",
      description:
        "Cette variable apparaît déjà assez souvent dans le sous-ensemble retenu.",
    },

    global_budget_exhausted: {
      label:
        "Budget global atteint",
      description:
        "Le nombre maximal d'analyses exploratoires automatiques est déjà atteint.",
    },
  };


const DECISION_LABELS:
  Record<
    PrioritizationDecisionKind,
    string
  > = {
    selected:
      "Retenue",
    deferred:
      "Différée",
    rejected:
      "Rejetée",
  };


const FAMILY_LABELS:
  Record<string, string> = {
    data_quality:
      "Qualité des données",
    time_series:
      "Série temporelle",
    quantitative_association:
      "Association quantitative",
    group_comparison:
      "Comparaison de groupes",
    categorical_association:
      "Association catégorielle",
    distribution:
      "Distribution",
    derived_gap:
      "Écart dérivé",
    entity_ranking:
      "Classement d'entités",
    ranking:
      "Classement",
    inequality:
      "Inégalités",
    geographic_comparison:
      "Comparaison géographique",
  };


function reasonPresentation(
  reasonCode:
    string
): ReasonPresentation {
  if (
    reasonCode in
    REASON_PRESENTATION
  ) {
    return REASON_PRESENTATION[
      reasonCode as
        PrioritizationReasonCode
    ];
  }


  return {
    label:
      reasonCode,
    description:
      "Décision produite par le moteur déterministe de priorisation.",
  };
}


function familyLabel(
  family:
    string
): string {
  return (
    FAMILY_LABELS[
      family
    ] ??
    family
  );
}


function formatScore(
  value:
    number
): string {
  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          1,
      }
    )
    .format(
      value
    );
}


function DecisionList({
  title,
  decision,
  decisions,
  defaultOpen = false,
}: {
  title:
    string;

  decision:
    PrioritizationDecisionKind;

  decisions:
    AnalysisPrioritizationDecisionView[];

  defaultOpen?:
    boolean;
}) {
  if (
    decisions.length ===
    0
  ) {
    return null;
  }


  return (
    <details
      className={
        styles.decisionGroup
      }
      open={
        defaultOpen
      }
    >
      <summary
        className={
          styles.decisionGroupSummary
        }
      >
        <span>
          {title}
        </span>

        <span
          className={
            styles.groupCount
          }
        >
          {
            decisions.length
          }
        </span>
      </summary>

      <div
        className={
          styles.decisionList
        }
      >
        {
          decisions.map(
            (
              item,
              itemIndex
            ) => {
              const reason =
                reasonPresentation(
                  item.reason_code
                );


              return (
                <details
                  className={
                    styles.decisionCard
                  }
                  key={
                    [
                      decision,
                      item.analysis_id,
                      item.family,
                      item.reason_code,
                      item.variable_keys.join(
                        "|"
                      ),
                      itemIndex,
                    ].join(
                      "::"
                    )
                  }
                >
                  <summary
                    className={
                      styles.decisionSummary
                    }
                  >
                    <div
                      className={
                        styles.decisionMain
                      }
                    >
                      <span
                        className={
                          `${styles.decisionBadge} ${styles[decision]}`
                        }
                      >
                        {
                          DECISION_LABELS[
                            decision
                          ]
                        }
                      </span>

                      <div>
                        <strong>
                          {
                            item.title
                          }
                        </strong>

                        <small>
                          {
                            familyLabel(
                              item.family
                            )
                          }
                          {" · "}
                          Score d'exécution
                          {" "}
                          {
                            formatScore(
                              item.execution_priority_score
                            )
                          }
                        </small>
                      </div>
                    </div>

                    <span
                      className={
                        styles.reasonLabel
                      }
                    >
                      {
                        reason.label
                      }
                    </span>
                  </summary>

                  <div
                    className={
                      styles.decisionBody
                    }
                  >
                    <p>
                      {
                        reason.description
                      }
                    </p>

                    {
                      item.reasons.length >
                      0
                        ? (
                            <ul>
                              {
                                item.reasons.map(
                                  (
                                    explanation,
                                    index
                                  ) => (
                                    <li
                                      key={
                                        `${item.analysis_id}-reason-${index}`
                                      }
                                    >
                                      {
                                        explanation
                                      }
                                    </li>
                                  )
                                )
                              }
                            </ul>
                          )
                        : null
                    }

                    {
                      item.variable_keys.length >
                      0
                        ? (
                            <div
                              className={
                                styles.variableKeys
                              }
                            >
                              <span>
                                Variables concernées
                              </span>

                              <code>
                                {
                                  item.variable_keys.join(
                                    " · "
                                  )
                                }
                              </code>
                            </div>
                          )
                        : null
                    }

                    <div
                      className={
                        styles.scoreRow
                      }
                    >
                      <span>
                        Score Discovery
                        {" "}
                        <strong>
                          {
                            formatScore(
                              item.original_priority_score
                            )
                          }
                        </strong>
                      </span>

                      <span>
                        Score exécution
                        {" "}
                        <strong>
                          {
                            formatScore(
                              item.execution_priority_score
                            )
                          }
                        </strong>
                      </span>
                    </div>
                  </div>
                </details>
              );
            }
          )
        }
      </div>
    </details>
  );
}


export default function PrioritizationAuditPanel({
  audit,
}: Props) {
  const deferred =
    audit.decisions.filter(
      (
        item
      ) =>
        item.decision ===
        "deferred"
    );

  const rejected =
    audit.decisions.filter(
      (
        item
      ) =>
        item.decision ===
        "rejected"
    );

  const selected =
    audit.decisions.filter(
      (
        item
      ) =>
        item.decision ===
        "selected"
    );

  const nonExecutionReasons =
    Object.entries(
      audit.non_execution_reason_counts
    );


  return (
    <section
      className={
        styles.panel
      }
    >
      <div
        className={
          styles.header
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Priorisation analytique
          </span>

          <h3>
            Pourquoi toutes les analyses ne sont-elles pas exécutées ?
          </h3>

          <p>
            La Discovery conserve tous les candidats. Python applique ensuite des garde-fous de valeur analytique, de diversité et de coût pour choisir le sous-ensemble envoyé aux exécuteurs automatiques.
          </p>
        </div>

        <span
          className={
            styles.pythonBadge
          }
        >
          Décision Python déterministe
        </span>
      </div>

      <div
        className={
          styles.metrics
        }
      >
        <div>
          <span>
            Découvertes
          </span>
          <strong>
            {
              audit.discovered_count
            }
          </strong>
        </div>

        <div>
          <span>
            Retenues
          </span>
          <strong>
            {
              audit.selected_for_execution_count
            }
          </strong>
        </div>

        <div>
          <span>
            Différées
          </span>
          <strong>
            {
              audit.deferred_count
            }
          </strong>
        </div>

        <div>
          <span>
            Rejetées
          </span>
          <strong>
            {
              audit.rejected_count
            }
          </strong>
        </div>
      </div>

      {
        nonExecutionReasons.length >
        0
          ? (
              <div
                className={
                  styles.reasonSection
                }
              >
                <div
                  className={
                    styles.sectionTitle
                  }
                >
                  <strong>
                    Motifs de non-exécution automatique
                  </strong>

                  <span>
                    {
                      audit.deferred_count +
                      audit.rejected_count
                    }
                    {" "}
                    candidats concernés
                  </span>
                </div>

                <div
                  className={
                    styles.reasonGrid
                  }
                >
                  {
                    nonExecutionReasons.map(
                      ([
                        reasonCode,
                        count,
                      ]) => {
                        const reason =
                          reasonPresentation(
                            reasonCode
                          );


                        return (
                          <div
                            className={
                              styles.reasonItem
                            }
                            key={
                              reasonCode
                            }
                          >
                            <span
                              className={
                                styles.reasonCount
                              }
                            >
                              {count}
                            </span>

                            <div>
                              <strong>
                                {
                                  reason.label
                                }
                              </strong>

                              <small>
                                {
                                  reason.description
                                }
                              </small>
                            </div>
                          </div>
                        );
                      }
                    )
                  }
                </div>
              </div>
            )
          : null
      }

      <div
        className={
          styles.auditLists
        }
      >
        <DecisionList
          title="Analyses différées"
          decision="deferred"
          decisions={
            deferred
          }
        />

        <DecisionList
          title="Analyses rejetées"
          decision="rejected"
          decisions={
            rejected
          }
        />

        <DecisionList
          title="Analyses retenues"
          decision="selected"
          decisions={
            selected
          }
        />
      </div>

      <div
        className={
          styles.footerNote
        }
      >
        <p>
          Une analyse différée n'est pas présentée comme un échec statistique : elle n'a simplement pas consommé le budget exploratoire automatique. Les demandes explicites suivent un flux séparé et restent soumises à leurs propres validations.
        </p>

        <small>
          {
            audit.audit_rule_version
          }
          {" · "}
          {
            audit.prioritization_rule_version
          }
          {" · "}
          {
            audit.analytical_value_guard_rule_version
          }
        </small>
      </div>
    </section>
  );
}
