"use client";


import {
  useEffect,
  useState,
} from "react";


import {
  getModelLabModelHealth,
  getModelLabMonitoringAlert,
  ModelLabApiError,
} from "./modelLabApi";


import type {
  ModelLabModelHealthReason,
  ModelLabModelHealthStatus,
  ModelLabModelHealthSummary,
  ModelLabMonitoringAlertAction,
  ModelLabMonitoringAlertCategory,
  ModelLabMonitoringAlertDecision,
  ModelLabMonitoringAlertSeverity,
} from "./modelLabTypes";


import ModelMonitoringHistoryPanel
  from "./ModelMonitoringHistoryPanel";


import styles
  from "./ModelObservabilityPanel.module.css";


type ModelObservabilityState = {
  workflowId:
    string;

  modelId:
    string;

  refreshKey:
    number;

  status:
    | "ready"
    | "error";

  health:
    ModelLabModelHealthSummary
    | null;

  alert:
    ModelLabMonitoringAlertDecision
    | null;

  error:
    string
    | null;
};


/* ============================================================
   PRESENTATION
============================================================ */


function healthStatusLabel(
  value:
    ModelLabModelHealthStatus
): string {
  const labels:
    Record<
      ModelLabModelHealthStatus,
      string
    > = {
      insufficient_evidence:
        "Preuves insuffisantes",

      healthy:
        "Sain",

      attention:
        "À surveiller",

      critical:
        "Critique",
    };


  return labels[
    value
  ];
}


function healthReasonLabel(
  value:
    ModelLabModelHealthReason
): string {
  const labels:
    Record<
      ModelLabModelHealthReason,
      string
    > = {
      no_monitoring_evidence:
        "Aucune preuve de monitoring",

      drift_only_ok:
        "Drift vérifié uniquement",

      performance_only_ok:
        "Performance vérifiée uniquement",

      aligned_evidence_ok:
        "Drift et performance alignés",

      evidence_misaligned:
        "Snapshots désalignés",

      evidence_unverifiable:
        "Alignement non vérifiable",

      drift_signal:
        "Signal de dérive détecté",

      performance_warning:
        "Performance en vigilance",

      performance_degraded:
        "Performance dégradée",
    };


  return labels[
    value
  ];
}


function alertSeverityLabel(
  value:
    ModelLabMonitoringAlertSeverity
): string {
  const labels:
    Record<
      ModelLabMonitoringAlertSeverity,
      string
    > = {
      none:
        "Aucune",

      info:
        "Information",

      warning:
        "Avertissement",

      critical:
        "Critique",
    };


  return labels[
    value
  ];
}


function alertCategoryLabel(
  value:
    ModelLabMonitoringAlertCategory
): string {
  const labels:
    Record<
      ModelLabMonitoringAlertCategory,
      string
    > = {
      none:
        "Aucune alerte",

      monitoring_gap:
        "Couverture de monitoring incomplète",

      evidence_alignment_gap:
        "Preuves non alignées",

      data_shift:
        "Dérive des données",

      performance_warning:
        "Performance à surveiller",

      performance_degradation:
        "Dégradation de performance",
    };


  return labels[
    value
  ];
}


function alertActionLabel(
  value:
    ModelLabMonitoringAlertAction
): string {
  const labels:
    Record<
      ModelLabMonitoringAlertAction,
      string
    > = {
      no_action:
        "Aucune action requise",

      establish_monitoring_evidence:
        "Établir les premières preuves de monitoring",

      complete_monitoring_evidence:
        "Compléter les preuves de monitoring",

      align_monitoring_snapshots:
        "Réaligner les snapshots observés",

      review_observed_data_distribution:
        "Examiner la distribution des données observées",

      review_model_performance:
        "Examiner les performances du modèle",

      investigate_model_degradation:
        "Investiguer la dégradation du modèle",
    };


  return labels[
    value
  ];
}


function driftStatusLabel(
  value:
    ModelLabModelHealthSummary[
      "drift_status"
    ]
): string {
  if (
    value ===
    "ok"
  ) {
    return "Stable";
  }


  if (
    value ===
    "warning"
  ) {
    return "Vigilance";
  }


  if (
    value ===
    "drift"
  ) {
    return "Dérive détectée";
  }


  return "Non évalué";
}


function performanceStatusLabel(
  value:
    ModelLabModelHealthSummary[
      "performance_status"
    ]
): string {
  if (
    value ===
    "ok"
  ) {
    return "Stable";
  }


  if (
    value ===
    "warning"
  ) {
    return "Vigilance";
  }


  if (
    value ===
    "degraded"
  ) {
    return "Dégradée";
  }


  return "Non évaluée";
}


function evidenceAlignmentLabel(
  value:
    ModelLabModelHealthSummary[
      "evidence_alignment"
    ]
): string {
  if (
    value ===
    "aligned"
  ) {
    return "Snapshots alignés";
  }


  if (
    value ===
    "misaligned"
  ) {
    return "Snapshots désalignés";
  }


  if (
    value ===
    "unverifiable"
  ) {
    return "Alignement non vérifiable";
  }


  if (
    value ===
    "single_source"
  ) {
    return "Une seule source de preuve";
  }


  return "Aucune preuve";
}


function formatDate(
  value:
    string
    | null
): string | null {
  if (
    !value
  ) {
    return null;
  }


  const date =
    new Date(
      value
    );


  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return value;
  }


  return (
    new Intl.DateTimeFormat(
      "fr-FR",
      {
        dateStyle:
          "medium",

        timeStyle:
          "short",
      }
    ).format(
      date
    )
  );
}


function shortIdentifier(
  value:
    string
    | null
): string | null {
  if (
    !value
  ) {
    return null;
  }


  if (
    value.length <=
      22
  ) {
    return value;
  }


  return (
    `${value.slice(0, 10)}…${value.slice(-7)}`
  );
}


function errorMessage(
  error:
    unknown
): string {
  if (
    error instanceof
      ModelLabApiError
  ) {
    return error.message;
  }


  if (
    error instanceof
      Error
  ) {
    return error.message;
  }


  return (
    "Impossible de charger l’observabilité du modèle."
  );
}


/* ============================================================
   SNAPSHOT CONSISTENCY
============================================================ */


function monitoringSnapshotConsistent(
  health:
    ModelLabModelHealthSummary,

  alert:
    ModelLabMonitoringAlertDecision
): boolean {
  return (
    health.workflow_id ===
      alert.workflow_id
    &&
    health.model_id ===
      alert.model_id
    &&
    health.health_status ===
      alert.health_status
    &&
    health.health_reason ===
      alert.health_reason
    &&
    health.evidence_alignment ===
      alert.evidence_alignment
    &&
    health.joint_interpretation_allowed ===
      alert.joint_interpretation_allowed
    &&
    health.drift_evaluation_id ===
      alert.drift_evaluation_id
    &&
    health.performance_evaluation_id ===
      alert.performance_evaluation_id
    &&
    health.privacy_scope ===
      "aggregate_only"
    &&
    alert.privacy_scope ===
      "aggregate_only"
  );
}


/* ============================================================
   STATUS STYLE
============================================================ */


function healthToneClass(
  status:
    ModelLabModelHealthStatus
): string {
  if (
    status ===
    "healthy"
  ) {
    return styles.positive;
  }


  if (
    status ===
    "critical"
  ) {
    return styles.critical;
  }


  if (
    status ===
    "attention"
  ) {
    return styles.warning;
  }


  return styles.info;
}


function severityToneClass(
  severity:
    ModelLabMonitoringAlertSeverity
): string {
  if (
    severity ===
    "critical"
  ) {
    return styles.critical;
  }


  if (
    severity ===
    "warning"
  ) {
    return styles.warning;
  }


  if (
    severity ===
    "info"
  ) {
    return styles.info;
  }


  return styles.positive;
}


/* ============================================================
   COMPONENT
============================================================ */


export default function ModelObservabilityPanel({
  workflowId,
  modelId,
}: {
  workflowId:
    string;

  modelId:
    string;
}) {
  const [
    state,
    setState,
  ] = useState<
    ModelObservabilityState
    | null
  >(
    null
  );


  const [
    refreshKey,
    setRefreshKey,
  ] = useState(
    0
  );


  useEffect(
    () => {
      const controller =
        new AbortController();


      void (
        async () => {
          try {
            const [
              health,
              alert,
            ] = await Promise.all([
              getModelLabModelHealth(
                workflowId,
                modelId,
                controller.signal
              ),

              getModelLabMonitoringAlert(
                workflowId,
                modelId,
                controller.signal
              ),
            ]);


            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            if (
              health.workflow_id !==
                workflowId
              ||
              health.model_id !==
                modelId
              ||
              alert.workflow_id !==
                workflowId
              ||
              alert.model_id !==
                modelId
            ) {
              throw new Error(
                (
                  "Le serveur a retourné une identité " +
                  "d’observabilité incohérente."
                )
              );
            }


            if (
              !monitoringSnapshotConsistent(
                health,
                alert
              )
            ) {
              throw new Error(
                (
                  "Les preuves d’observabilité ont changé " +
                  "pendant le chargement. Actualisez le panneau."
                )
              );
            }


            setState({
              workflowId,
              modelId,

              refreshKey,

              status:
                "ready",

              health,
              alert,

              error:
                null,
            });
          } catch (
            error
          ) {
            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            setState({
              workflowId,
              modelId,

              refreshKey,

              status:
                "error",

              health:
                null,

              alert:
                null,

              error:
                errorMessage(
                  error
                ),
            });
          }
        }
      )();


      return () => {
        controller.abort();
      };
    },
    [
      workflowId,
      modelId,
      refreshKey,
    ]
  );


  const stateIdentityMatches =
    Boolean(
      state
      &&
      state.workflowId ===
        workflowId
      &&
      state.modelId ===
        modelId
    );


  const stateRevisionMatches =
    Boolean(
      stateIdentityMatches
      &&
      state
        ?.refreshKey ===
        refreshKey
    );


  const loading =
    !stateIdentityMatches;


  const refreshing =
    Boolean(
      stateIdentityMatches
      &&
      !stateRevisionMatches
    );


  const error =
    stateRevisionMatches
    &&
    state
      ?.status ===
      "error"
      ? state.error
      : null;


  const health =
    stateIdentityMatches
    &&
    state
      ?.status ===
      "ready"
      ? state.health
      : null;


  const alert =
    stateIdentityMatches
    &&
    state
      ?.status ===
      "ready"
      ? state.alert
      : null;


  const driftDate =
    formatDate(
      health
        ?.drift_evaluated_at_utc ??
      null
    );


  const performanceDate =
    formatDate(
      health
        ?.performance_evaluated_at_utc ??
      null
    );


  return (
    <section
      className={
        styles.panel
      }
      aria-label="Observabilité du modèle"
    >
      <div
        className={
          styles.header
        }
      >
        <div>
          <p
            className={
              styles.eyebrow
            }
          >
            Observabilité
          </p>

          <h3>
            Santé du modèle
          </h3>

          <p
            className={
              styles.subtitle
            }
          >
            Lecture des dernières preuves
            Drift et Performance persistées
            par le backend.
          </p>
        </div>


        <button
          type="button"
          className={
            styles.refreshButton
          }
          disabled={
            loading
            ||
            refreshing
          }
          onClick={
            () => {
              setRefreshKey(
                (
                  current
                ) =>
                  current + 1
              );
            }
          }
        >
          {
            loading
            ||
            refreshing
              ? "Lecture…"
              : "Actualiser"
          }
        </button>
      </div>


      {
        loading
          ? (
              <div
                className={
                  styles.loadingState
                }
              >
                <span
                  className={
                    styles.loadingDot
                  }
                  aria-hidden="true"
                />

                <span>
                  Lecture des preuves
                  d’observabilité…
                </span>
              </div>
            )
          : null
      }


      {
        error
          ? (
              <div
                className={
                  styles.errorState
                }
              >
                <strong>
                  Observabilité indisponible
                </strong>

                <span>
                  {
                    error
                  }
                </span>
              </div>
            )
          : null
      }


      {
        health
        &&
        alert
          ? (
              <>
                <div
                  className={
                    styles.summaryGrid
                  }
                >
                  <article
                    className={
                      styles.summaryCard
                    }
                  >
                    <span
                      className={
                        styles.cardLabel
                      }
                    >
                      Santé
                    </span>

                    <strong
                      className={
                        `${styles.cardValue} ${healthToneClass(
                          health.health_status
                        )}`
                      }
                    >
                      {
                        healthStatusLabel(
                          health.health_status
                        )
                      }
                    </strong>

                    <small>
                      {
                        healthReasonLabel(
                          health.health_reason
                        )
                      }
                    </small>
                  </article>


                  <article
                    className={
                      styles.summaryCard
                    }
                  >
                    <span
                      className={
                        styles.cardLabel
                      }
                    >
                      Alerte
                    </span>

                    <strong
                      className={
                        `${styles.cardValue} ${severityToneClass(
                          alert.severity
                        )}`
                      }
                    >
                      {
                        alertSeverityLabel(
                          alert.severity
                        )
                      }
                    </strong>

                    <small>
                      {
                        alertCategoryLabel(
                          alert.alert_category
                        )
                      }
                    </small>
                  </article>


                  <article
                    className={
                      styles.summaryCard
                    }
                  >
                    <span
                      className={
                        styles.cardLabel
                      }
                    >
                      Data Drift
                    </span>

                    <strong
                      className={
                        styles.cardValue
                      }
                    >
                      {
                        driftStatusLabel(
                          health.drift_status
                        )
                      }
                    </strong>

                    {
                      health.drift_evaluation_id
                        ? (
                            <div
                              className={
                                styles.evidenceMeta
                              }
                            >
                              <span>
                                Rev. {
                                  health
                                    .drift_observed_preparation_session_revision ??
                                  "historique"
                                }
                                {" · "}
                                {
                                  health
                                    .drift_observed_row_count ??
                                  "?"
                                } lignes
                              </span>

                              {
                                driftDate
                                  ? (
                                      <span>
                                        {
                                          driftDate
                                        }
                                      </span>
                                    )
                                  : null
                              }

                              <code
                                title={
                                  health.drift_evaluation_id
                                }
                              >
                                {
                                  shortIdentifier(
                                    health.drift_evaluation_id
                                  )
                                }
                              </code>
                            </div>
                          )
                        : (
                            <small>
                              Aucune évaluation Drift
                            </small>
                          )
                    }
                  </article>


                  <article
                    className={
                      styles.summaryCard
                    }
                  >
                    <span
                      className={
                        styles.cardLabel
                      }
                    >
                      Performance
                    </span>

                    <strong
                      className={
                        styles.cardValue
                      }
                    >
                      {
                        performanceStatusLabel(
                          health.performance_status
                        )
                      }
                    </strong>

                    {
                      health.performance_evaluation_id
                        ? (
                            <div
                              className={
                                styles.evidenceMeta
                              }
                            >
                              <span>
                                Rev. {
                                  health
                                    .performance_observed_preparation_session_revision ??
                                  "historique"
                                }
                                {" · "}
                                {
                                  health
                                    .performance_observed_row_count ??
                                  "?"
                                } lignes
                              </span>

                              {
                                performanceDate
                                  ? (
                                      <span>
                                        {
                                          performanceDate
                                        }
                                      </span>
                                    )
                                  : null
                              }

                              <code
                                title={
                                  health.performance_evaluation_id
                                }
                              >
                                {
                                  shortIdentifier(
                                    health.performance_evaluation_id
                                  )
                                }
                              </code>
                            </div>
                          )
                        : (
                            <small>
                              Aucune évaluation Performance
                            </small>
                          )
                    }
                  </article>
                </div>


                <div
                  className={
                    health
                      .joint_interpretation_allowed
                      ? `${styles.interpretation} ${styles.interpretationAligned}`
                      : styles.interpretation
                  }
                >
                  <div>
                    <span>
                      Alignement des preuves
                    </span>

                    <strong>
                      {
                        evidenceAlignmentLabel(
                          health.evidence_alignment
                        )
                      }
                    </strong>
                  </div>

                  <p>
                    {
                      health
                        .joint_interpretation_allowed
                        ? (
                            "Drift et Performance décrivent le même snapshot observé. Leur lecture conjointe est autorisée."
                          )
                        : (
                            "DataLens ne fusionne pas causalement Drift et Performance lorsque les preuves ne décrivent pas le même snapshot."
                          )
                    }
                  </p>
                </div>


                <div
                  className={
                    styles.actionGrid
                  }
                >
                  <div
                    className={
                      styles.actionCard
                    }
                  >
                    <span>
                      Action recommandée
                    </span>

                    <strong>
                      {
                        alertActionLabel(
                          alert.recommended_action
                        )
                      }
                    </strong>
                  </div>


                  <div
                    className={
                      styles.actionCard
                    }
                  >
                    <span>
                      Signal de notification
                    </span>

                    <strong
                      className={
                        alert
                          .notification_recommended
                          ? styles.warning
                          : styles.positive
                      }
                    >
                      {
                        alert
                          .notification_recommended
                          ? "Recommandé"
                          : "Non recommandé"
                      }
                    </strong>

                    <small>
                      Décision uniquement :
                      aucun email ou webhook
                      n’est envoyé par cet écran.
                    </small>
                  </div>
                </div>


                <ModelMonitoringHistoryPanel
                  workflowId={
                    workflowId
                  }
                  modelId={
                    modelId
                  }
                />
              </>
            )
          : null
      }
    </section>
  );
}
