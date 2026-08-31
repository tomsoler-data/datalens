"use client";


import {
  useEffect,
  useMemo,
  useState,
} from "react";


import {
  getModelLabDriftHistory,
  getModelLabPerformanceHistory,
  ModelLabApiError,
} from "./modelLabApi";


import type {
  ModelLabDriftEvaluationRecord,
  ModelLabDriftHistoryResponse,
  ModelLabPerformanceEvaluationRecord,
  ModelLabPerformanceHistoryResponse,
  ModelLabPerformanceMetricComparison,
} from "./modelLabTypes";


import styles
  from "./ModelMonitoringHistoryPanel.module.css";


const MAX_VISIBLE_EVALUATIONS =
  5;


type ModelMonitoringHistoryState = {
  workflowId:
    string;

  modelId:
    string;

  loadKey:
    number;

  status:
    | "ready"
    | "error";

  drift:
    ModelLabDriftHistoryResponse
    | null;

  performance:
    ModelLabPerformanceHistoryResponse
    | null;

  error:
    string
    | null;
};


/* ============================================================
   PRESENTATION
============================================================ */


function shortIdentifier(
  value:
    string
): string {
  if (
    value.length <=
    22
  ) {
    return value;
  }


  return (
    `${value.slice(0, 10)}\u2026${value.slice(-7)}`
  );
}


function formatDate(
  value:
    string
): string {
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


function formatNumber(
  value:
    number
): string {
  if (
    !Number.isFinite(
      value
    )
  ) {
    return "-";
  }


  return (
    new Intl.NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          4,
      }
    ).format(
      value
    )
  );
}


function formatRatio(
  value:
    number
): string {
  return (
    new Intl.NumberFormat(
      "fr-FR",
      {
        style:
          "percent",

        maximumFractionDigits:
          1,
      }
    ).format(
      value
    )
  );
}


function metricLabel(
  value:
    string
): string {
  const labels:
    Record<
      string,
      string
    > = {
      f1_macro:
        "F1 macro",

      rmse:
        "RMSE",

      accuracy:
        "Accuracy",

      balanced_accuracy:
        "Balanced accuracy",

      precision_macro:
        "Precision macro",

      recall_macro:
        "Recall macro",

      mae:
        "MAE",

      median_absolute_error:
        "Median absolute error",

      explained_variance:
        "Explained variance",
    };


  return (
    labels[
      value
    ] ??
    value
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
    "Impossible de charger l\u2019historique de monitoring."
  );
}


/* ============================================================
   RUNTIME CONSISTENCY
============================================================ */


function driftHistoryConsistent(
  response:
    ModelLabDriftHistoryResponse,

  {
    workflowId,
    modelId,
  }: {
    workflowId:
      string;

    modelId:
      string;
  }
): boolean {
  return (
    response.workflow_id ===
      workflowId
    &&
    response.model_id ===
      modelId
    &&
    response.evaluation_count ===
      response.evaluations.length
    &&
    response.evaluations.every(
      (
        evaluation
      ) =>
        (
          evaluation.workflow_id ===
            workflowId
          &&
          evaluation.model_id ===
            modelId
          &&
          evaluation.privacy_scope ===
            "aggregate_only"
        )
    )
  );
}


function performanceHistoryConsistent(
  response:
    ModelLabPerformanceHistoryResponse,

  {
    workflowId,
    modelId,
  }: {
    workflowId:
      string;

    modelId:
      string;
  }
): boolean {
  return (
    response.workflow_id ===
      workflowId
    &&
    response.model_id ===
      modelId
    &&
    response.evaluation_count ===
      response.evaluations.length
    &&
    response.evaluations.every(
      (
        evaluation
      ) =>
        (
          evaluation.workflow_id ===
            workflowId
          &&
          evaluation.model_id ===
            modelId
          &&
          evaluation.privacy_scope ===
            "aggregate_only"
        )
    )
  );
}


/* ============================================================
   ORDER
============================================================ */


function dateRank(
  value:
    string
): number {
  const parsed =
    Date.parse(
      value
    );


  return (
    Number.isFinite(
      parsed
    )
      ? parsed
      : 0
  );
}


function sortedDriftEvaluations(
  response:
    ModelLabDriftHistoryResponse
): ModelLabDriftEvaluationRecord[] {
  return (
    [
      ...response.evaluations,
    ].sort(
      (
        left,
        right
      ) => {
        const dateDifference =
          dateRank(
            right.evaluated_at_utc
          )
          -
          dateRank(
            left.evaluated_at_utc
          );


        if (
          dateDifference !==
          0
        ) {
          return dateDifference;
        }


        return (
          right.evaluation_id
            .localeCompare(
              left.evaluation_id
            )
        );
      }
    )
  );
}


function sortedPerformanceEvaluations(
  response:
    ModelLabPerformanceHistoryResponse
): ModelLabPerformanceEvaluationRecord[] {
  return (
    [
      ...response.evaluations,
    ].sort(
      (
        left,
        right
      ) => {
        const dateDifference =
          dateRank(
            right.evaluated_at_utc
          )
          -
          dateRank(
            left.evaluated_at_utc
          );


        if (
          dateDifference !==
          0
        ) {
          return dateDifference;
        }


        return (
          right
            .performance_evaluation_id
            .localeCompare(
              left.performance_evaluation_id
            )
        );
      }
    )
  );
}


/* ============================================================
   DRIFT
============================================================ */


function driftStatusLabel(
  value:
    ModelLabDriftEvaluationRecord[
      "overall_status"
    ]
): string {
  if (
    value ===
    "drift"
  ) {
    return (
      "D\u00e9rive"
    );
  }


  if (
    value ===
    "warning"
  ) {
    return (
      "Vigilance"
    );
  }


  return "Stable";
}


function driftToneClass(
  value:
    ModelLabDriftEvaluationRecord[
      "overall_status"
    ]
): string {
  if (
    value ===
    "drift"
  ) {
    return styles.critical;
  }


  if (
    value ===
    "warning"
  ) {
    return styles.warning;
  }


  return styles.positive;
}


/* ============================================================
   PERFORMANCE
============================================================ */


function performanceStatusLabel(
  value:
    ModelLabPerformanceEvaluationRecord[
      "performance_status"
    ]
): string {
  if (
    value ===
    "degraded"
  ) {
    return (
      "D\u00e9grad\u00e9e"
    );
  }


  if (
    value ===
    "warning"
  ) {
    return (
      "Vigilance"
    );
  }


  return "Stable";
}


function performanceToneClass(
  value:
    ModelLabPerformanceEvaluationRecord[
      "performance_status"
    ]
): string {
  if (
    value ===
    "degraded"
  ) {
    return styles.critical;
  }


  if (
    value ===
    "warning"
  ) {
    return styles.warning;
  }


  return styles.positive;
}


function primaryMetricResult(
  evaluation:
    ModelLabPerformanceEvaluationRecord
): ModelLabPerformanceMetricComparison | null {
  return (
    evaluation.metric_results.find(
      (
        metric
      ) =>
        metric.metric_name ===
        evaluation.primary_metric
    )
    ??
    null
  );
}


/* ============================================================
   COMPONENT
============================================================ */


type Props = {
  workflowId:
    string;

  modelId:
    string;
};


export default function ModelMonitoringHistoryPanel(
  {
    workflowId,
    modelId,
  }:
    Props
) {
  const [
    expanded,
    setExpanded,
  ] = useState(
    false
  );

  const [
    loadKey,
    setLoadKey,
  ] = useState(
    0
  );

  const [
    state,
    setState,
  ] = useState<
    ModelMonitoringHistoryState
    | null
  >(
    null
  );


  useEffect(
    () => {
      if (
        !expanded
      ) {
        return;
      }


      const controller =
        new AbortController();


      void (
        async () => {
          try {
            const [
              drift,
              performance,
            ] =
              await Promise.all([
                getModelLabDriftHistory(
                  workflowId,
                  modelId,
                  controller.signal
                ),

                getModelLabPerformanceHistory(
                  workflowId,
                  modelId,
                  controller.signal
                ),
              ]);


            if (
              controller.signal.aborted
            ) {
              return;
            }


            if (
              !driftHistoryConsistent(
                drift,
                {
                  workflowId,
                  modelId,
                }
              )
              ||
              !performanceHistoryConsistent(
                performance,
                {
                  workflowId,
                  modelId,
                }
              )
            ) {
              throw (
                new Error(
                  "Le serveur a retourn\u00e9 un historique de monitoring incoh\u00e9rent."
                )
              );
            }


            setState({
              workflowId,
              modelId,
              loadKey,

              status:
                "ready",

              drift,
              performance,

              error:
                null,
            });
          }
          catch (
            error
          ) {
            if (
              controller.signal.aborted
            ) {
              return;
            }


            setState(
              (
                current
              ) => {
                const sameIdentity =
                  Boolean(
                    current
                    &&
                    current.workflowId ===
                      workflowId
                    &&
                    current.modelId ===
                      modelId
                  );


                return {
                  workflowId,
                  modelId,
                  loadKey,

                  status:
                    "error",

                  drift:
                    sameIdentity
                      ? current
                          ?.drift ??
                        null
                      : null,

                  performance:
                    sameIdentity
                      ? current
                          ?.performance ??
                        null
                      : null,

                  error:
                    errorMessage(
                      error
                    ),
                };
              }
            );
          }
        }
      )();


      return () => {
        controller.abort();
      };
    },
    [
      expanded,
      workflowId,
      modelId,
      loadKey,
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
        ?.loadKey ===
        loadKey
    );


  const loading =
    Boolean(
      expanded
      &&
      !stateIdentityMatches
    );


  const refreshing =
    Boolean(
      expanded
      &&
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


  const driftHistory =
    stateIdentityMatches
      ? state
          ?.drift ??
        null
      : null;


  const performanceHistory =
    stateIdentityMatches
      ? state
          ?.performance ??
        null
      : null;


  const driftEvaluations =
    useMemo(
      () =>
        (
          driftHistory
            ? sortedDriftEvaluations(
                driftHistory
              )
            : []
        ),
      [
        driftHistory,
      ]
    );


  const performanceEvaluations =
    useMemo(
      () =>
        (
          performanceHistory
            ? sortedPerformanceEvaluations(
                performanceHistory
              )
            : []
        ),
      [
        performanceHistory,
      ]
    );


  const visibleDrift =
    driftEvaluations.slice(
      0,
      MAX_VISIBLE_EVALUATIONS
    );


  const visiblePerformance =
    performanceEvaluations.slice(
      0,
      MAX_VISIBLE_EVALUATIONS
    );


  return (
    <section
      className={
        styles.panel
      }
      aria-label={
        "Historique de monitoring"
      }
    >
      <button
        type="button"
        className={
          styles.toggle
        }
        aria-expanded={
          expanded
        }
        onClick={
          () => {
            if (
              !expanded
            ) {
              setLoadKey(
                (
                  current
                ) =>
                  current + 1
              );
            }


            setExpanded(
              (
                current
              ) =>
                !current
            );
          }
        }
      >
        <span
          className={
            styles.toggleCopy
          }
        >
          <strong>
            Historique de monitoring
          </strong>

          <small>
            {
              "Consulter les preuves Drift et Performance persist\u00e9es."
            }
          </small>
        </span>

        <span
          className={
            styles.toggleAction
          }
        >
          {
            expanded
              ? "Masquer"
              : (
                  "Voir l\u2019historique"
                )
          }
        </span>
      </button>


      {
        expanded
          ? (
              <div
                className={
                  styles.content
                }
              >
                <div
                  className={
                    styles.toolbar
                  }
                >
                  <p>
                    {
                      "Les deux historiques restent ind\u00e9pendants : DataLens ne transforme pas leur proximit\u00e9 temporelle en lien causal."
                    }
                  </p>

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
                        setLoadKey(
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
                        ? "Lecture\u2026"
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
                            {
                              "Lecture de l\u2019historique\u2026"
                            }
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
                            {
                              "Historique indisponible"
                            }
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
                  driftHistory
                  &&
                  performanceHistory
                    ? (
                        <>
                          <div
                            className={
                              styles.historyGrid
                            }
                          >
                            <article
                              className={
                                styles.historyColumn
                              }
                            >
                              <header
                                className={
                                  styles.columnHeader
                                }
                              >
                                <div>
                                  <span
                                    className={
                                      styles.columnEyebrow
                                    }
                                  >
                                    Data Drift
                                  </span>

                                  <strong>
                                    {
                                      "D\u00e9rive des donn\u00e9es"
                                    }
                                  </strong>
                                </div>

                                <span
                                  className={
                                    styles.count
                                  }
                                >
                                  {
                                    driftHistory
                                      .evaluation_count
                                  }
                                </span>
                              </header>


                              {
                                visibleDrift.length ===
                                  0
                                  ? (
                                      <div
                                        className={
                                          styles.emptyState
                                        }
                                      >
                                        {
                                          "Aucune \u00e9valuation Drift persist\u00e9e."
                                        }
                                      </div>
                                    )
                                  : (
                                      <div
                                        className={
                                          styles.list
                                        }
                                      >
                                        {
                                          visibleDrift.map(
                                            (
                                              evaluation
                                            ) => (
                                              <div
                                                key={
                                                  evaluation.evaluation_id
                                                }
                                                className={
                                                  styles.historyItem
                                                }
                                              >
                                                <div
                                                  className={
                                                    styles.itemHeader
                                                  }
                                                >
                                                  <strong>
                                                    {
                                                      formatDate(
                                                        evaluation
                                                          .evaluated_at_utc
                                                      )
                                                    }
                                                  </strong>

                                                  <span
                                                    className={
                                                      `${styles.status} ${driftToneClass(
                                                        evaluation
                                                          .overall_status
                                                      )}`
                                                    }
                                                  >
                                                    {
                                                      driftStatusLabel(
                                                        evaluation
                                                          .overall_status
                                                      )
                                                    }
                                                  </span>
                                                </div>

                                                <div
                                                  className={
                                                    styles.meta
                                                  }
                                                >
                                                  <span>
                                                    Rev. {
                                                      evaluation
                                                        .observed_preparation_session_revision ??
                                                      "historique"
                                                    }
                                                    {" \u00b7 "}
                                                    {
                                                      evaluation
                                                        .observed_row_count
                                                    } lignes
                                                  </span>

                                                  <span>
                                                    {
                                                      evaluation
                                                        .drift_feature_count
                                                    } {
                                                      "en d\u00e9rive"
                                                    }
                                                    {" \u00b7 "}
                                                    {
                                                      evaluation
                                                        .warning_feature_count
                                                    } en vigilance
                                                  </span>
                                                </div>

                                                <code
                                                  title={
                                                    evaluation.evaluation_id
                                                  }
                                                >
                                                  {
                                                    shortIdentifier(
                                                      evaluation.evaluation_id
                                                    )
                                                  }
                                                </code>
                                              </div>
                                            )
                                          )
                                        }
                                      </div>
                                    )
                              }


                              {
                                driftEvaluations.length >
                                  MAX_VISIBLE_EVALUATIONS
                                  ? (
                                      <small
                                        className={
                                          styles.more
                                        }
                                      >
                                        +{
                                          driftEvaluations.length
                                          -
                                          MAX_VISIBLE_EVALUATIONS
                                        } {
                                          "\u00e9valuation(s) plus ancienne(s)"
                                        }
                                      </small>
                                    )
                                  : null
                              }
                            </article>


                            <article
                              className={
                                styles.historyColumn
                              }
                            >
                              <header
                                className={
                                  styles.columnHeader
                                }
                              >
                                <div>
                                  <span
                                    className={
                                      styles.columnEyebrow
                                    }
                                  >
                                    Performance
                                  </span>

                                  <strong>
                                    {
                                      "Performance supervis\u00e9e"
                                    }
                                  </strong>
                                </div>

                                <span
                                  className={
                                    styles.count
                                  }
                                >
                                  {
                                    performanceHistory
                                      .evaluation_count
                                  }
                                </span>
                              </header>


                              {
                                visiblePerformance.length ===
                                  0
                                  ? (
                                      <div
                                        className={
                                          styles.emptyState
                                        }
                                      >
                                        {
                                          "Aucune \u00e9valuation Performance persist\u00e9e."
                                        }
                                      </div>
                                    )
                                  : (
                                      <div
                                        className={
                                          styles.list
                                        }
                                      >
                                        {
                                          visiblePerformance.map(
                                            (
                                              evaluation
                                            ) => {
                                              const primary =
                                                primaryMetricResult(
                                                  evaluation
                                                );


                                              return (
                                                <div
                                                  key={
                                                    evaluation
                                                      .performance_evaluation_id
                                                  }
                                                  className={
                                                    styles.historyItem
                                                  }
                                                >
                                                  <div
                                                    className={
                                                      styles.itemHeader
                                                    }
                                                  >
                                                    <strong>
                                                      {
                                                        formatDate(
                                                          evaluation
                                                            .evaluated_at_utc
                                                        )
                                                      }
                                                    </strong>

                                                    <span
                                                      className={
                                                        `${styles.status} ${performanceToneClass(
                                                          evaluation
                                                            .performance_status
                                                        )}`
                                                      }
                                                    >
                                                      {
                                                        performanceStatusLabel(
                                                          evaluation
                                                            .performance_status
                                                        )
                                                      }
                                                    </span>
                                                  </div>

                                                  <div
                                                    className={
                                                      styles.meta
                                                    }
                                                  >
                                                    <span>
                                                      Rev. {
                                                        evaluation
                                                          .observed_preparation_session_revision
                                                      }
                                                      {" \u00b7 "}
                                                      {
                                                        evaluation
                                                          .observed_row_count
                                                      } lignes
                                                    </span>

                                                    {
                                                      primary
                                                        ? (
                                                            <span>
                                                              {
                                                                metricLabel(
                                                                  primary.metric_name
                                                                )
                                                              }
                                                              {" \u00b7 "}
                                                              {
                                                                "R\u00e9f. "
                                                              }
                                                              {
                                                                formatNumber(
                                                                  primary.reference_value
                                                                )
                                                              }
                                                              {" \u00b7 "}
                                                              Obs. {
                                                                formatNumber(
                                                                  primary.observed_value
                                                                )
                                                              }
                                                            </span>
                                                          )
                                                        : null
                                                    }

                                                    <span>
                                                      {
                                                        "D\u00e9gradation "
                                                      }
                                                      {
                                                        formatNumber(
                                                          evaluation
                                                            .primary_metric_degradation_amount
                                                        )
                                                      }
                                                      {
                                                        evaluation
                                                          .primary_metric_degradation_ratio !==
                                                          null
                                                          ? (
                                                              ` \u00b7 ${formatRatio(
                                                                evaluation
                                                                  .primary_metric_degradation_ratio
                                                              )}`
                                                            )
                                                          : ""
                                                      }
                                                    </span>
                                                  </div>

                                                  <code
                                                    title={
                                                      evaluation
                                                        .performance_evaluation_id
                                                    }
                                                  >
                                                    {
                                                      shortIdentifier(
                                                        evaluation
                                                          .performance_evaluation_id
                                                      )
                                                    }
                                                  </code>
                                                </div>
                                              );
                                            }
                                          )
                                        }
                                      </div>
                                    )
                              }


                              {
                                performanceEvaluations.length >
                                  MAX_VISIBLE_EVALUATIONS
                                  ? (
                                      <small
                                        className={
                                          styles.more
                                        }
                                      >
                                        +{
                                          performanceEvaluations.length
                                          -
                                          MAX_VISIBLE_EVALUATIONS
                                        } {
                                          "\u00e9valuation(s) plus ancienne(s)"
                                        }
                                      </small>
                                    )
                                  : null
                              }
                            </article>
                          </div>


                          <p
                            className={
                              styles.privacyNote
                            }
                          >
                            {
                              "Historique aggregate-only : aucune ligne brute, cible brute ou pr\u00e9diction individuelle n\u2019est affich\u00e9e."
                            }
                          </p>
                        </>
                      )
                    : null
                }
              </div>
            )
          : null
      }
    </section>
  );
}
