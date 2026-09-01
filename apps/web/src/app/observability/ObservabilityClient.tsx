"use client";

// DATALENS_OBSERVABILITY_UNIFIED_SHELL_V0_1
// DATALENS_OBSERVABILITY_UNIFIED_SHELL_ENCODING_R2

// DATALENS_OBSERVABILITY_HYDRATION_GUARD_V0_1

// DATALENS_OBSERVABILITY_PRODUCT_LANGUAGE_V0_1


import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import Link from "next/link";

import {
  useRouter,
  useSearchParams,
} from "next/navigation";

import styles from "./observability.module.css";

import WorkspaceNavigation
  from "../../components/workspace/WorkspaceNavigation";

import type {
  WorkspaceStep,
} from "../../components/workspace/workspaceNavigationTypes";

import {
  persistActiveWorkspaceStep,
} from "../../components/workspace/workspaceNavigationStorage";

import {
  readActivePreparationWorkflowId,
} from "../../components/preparation/preparationWorkflowStorage";

import workspaceStyles
  from "../page.module.css";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


type TraceSummary = {
  trace_id:
    string;

  created_at_utc:
    string;

  trace_rule_version:
    string;

  workflow_id:
    string |
    null;

  analysis_id:
    string |
    null;

  analysis_source_type:
    AnalysisSourceType |
    null;

  run_status:
    "completed" |
    "failed";

  failure_stage:
    string |
    null;

  objective:
    string;

  dataset_filenames:
    string[];

  planner_status:
    string |
    null;

  planner_model:
    string |
    null;

  planner_rule_version:
    string |
    null;

  planner_attempt_count:
    number |
    null;

  planner_retry_count:
    number |
    null;

  planner_normalization_count:
    number |
    null;

  pipeline_status:
    string |
    null;

  pipeline_rule_version:
    string |
    null;

  tool_model:
    string |
    null;

  executed_count:
    number |
    null;

  total_ms:
    number;
};


type TraceListResponse = {
  trace_store_rule_version:
    string;

  trace_count:
    number;

  malformed_line_count:
    number;

  returned_count:
    number;

  traces:
    TraceSummary[];
};


type AggregateLatencyMetrics = {
  median_ms:
    number;

  p95_ms:
    number;

  mean_ms:
    number;
};


type AggregateCategoryCount = {
  name:
    string;

  count:
    number;
};


type TraceMetricsResponse = {
  trace_store_rule_version:
    string;

  trace_count:
    number;

  malformed_line_count:
    number;

  analyzed_trace_count:
    number;

  completed_trace_count:
    number;

  failed_trace_count:
    number;

  failure_rate:
    number;

  failure_stages:
    AggregateCategoryCount[];

  detailed_trace_count:
    number;

  executed_trace_count:
    number;

  execution_rate:
    number;

  planner_retry_trace_count:
    number;

  planner_retry_rate:
    number;

  planner_normalized_trace_count:
    number;

  planner_normalization_rate:
    number;

  planner_share_median:
    number;

  total_latency:
    AggregateLatencyMetrics;

  planner_latency:
    AggregateLatencyMetrics;

  native_pipeline_latency:
    AggregateLatencyMetrics;

  ingestion_latency:
    AggregateLatencyMetrics;

  planner_prompt_latency:
    AggregateLatencyMetrics;

  planner_model_inference_latency:
    AggregateLatencyMetrics;

  planner_structured_parse_latency:
    AggregateLatencyMetrics;

  planner_python_validation_latency:
    AggregateLatencyMetrics;

  planner_retry_feedback_latency:
    AggregateLatencyMetrics;

  tool_prompt_latency:
    AggregateLatencyMetrics;

  tool_model_inference_latency:
    AggregateLatencyMetrics;

  tool_response_parse_latency:
    AggregateLatencyMetrics;

  tool_python_validation_latency:
    AggregateLatencyMetrics;

  deterministic_execution_latency:
    AggregateLatencyMetrics;

  planner_model_share_median:
    number;

  tool_model_share_median:
    number;

  deterministic_execution_share_median:
    number;

  planner_models:
    AggregateCategoryCount[];

  tool_models:
    AggregateCategoryCount[];

  families:
    AggregateCategoryCount[];

  requested_tools:
    AggregateCategoryCount[];

  pipeline_statuses:
    AggregateCategoryCount[];
};


type TraceDataset = {
  dataset_id?:
    string;

  filename?:
    string;

  row_count?:
    number;

  column_count?:
    number;

  columns?:
    Array<{
      name?:
        string;

      dtype?:
        string;

      analysis_kind?:
        string;
    }>;
};


type TraceTiming = {
  ingestion_ms?:
    number;

  planner_ms?:
    number;

  native_pipeline_ms?:
    number;

  total_ms?:
    number;

  planner_prompt_construction_ms?:
    number;

  planner_model_inference_ms?:
    number;

  planner_structured_parse_ms?:
    number;

  planner_python_validation_ms?:
    number;

  planner_retry_feedback_ms?:
    number;

  tool_prompt_construction_ms?:
    number;

  tool_model_inference_ms?:
    number;

  tool_response_parse_ms?:
    number;

  tool_python_validation_ms?:
    number;

  deterministic_execution_ms?:
    number;
};


type TracePrivacy = {
  storage_scope?:
    string;

  contains_raw_dataset_rows?:
    boolean;

  contains_uploaded_file_contents?:
    boolean;

  contains_document_chunks?:
    boolean;

  contains_objective_text?:
    boolean;

  note?:
    string;
};


type AnalysisSourceType =
  | "initial_request"
  | "follow_up_prompt"
  | "document_request"
  | "automatic";


type TraceFailure = {
  stage:
    string;

  error_type:
    string;

  message_safe:
    string;
};


type TraceRecord = {
  trace_id:
    string;

  created_at_utc:
    string;

  trace_rule_version:
    string;

  workflow_id:
    string |
    null;

  analysis_id:
    string |
    null;

  analysis_source_type:
    AnalysisSourceType |
    null;

  run_status:
    "completed" |
    "failed";

  failure:
    TraceFailure |
    null;

  objective:
    string;

  objective_sha256:
    string;

  datasets:
    TraceDataset[];

  planner:
    Record<
      string,
      unknown
    >;

  native_pipeline:
    Record<
      string,
      unknown
    >;

  timings:
    TraceTiming;

  privacy:
    TracePrivacy;
};


type PlannerItem = {
  proposal_index?:
    number;

  validation_status?:
    string;

  raw_proposal?:
    Record<
      string,
      unknown
    > |
    null;

  canonical_proposal?:
    Record<
      string,
      unknown
    > |
    null;

  errors?:
    string[];

  warnings?:
    string[];

  normalizations?:
    string[];

  contract?:
    Record<
      string,
      unknown
    > |
    null;
};


type NativeItem = {
  contract_id?:
    string;

  family?:
    string;

  pipeline_status?:
    string;

  pipeline_errors?:
    string[];

  pipeline_warnings?:
    string[];

  tool_call?:
    Record<
      string,
      unknown
    > |
    null;

  execution?:
    Record<
      string,
      unknown
    > |
    null;
};


function isRecord(
  value:
    unknown
): value is
  Record<
    string,
    unknown
  > {
  return (
    typeof value ===
      "object" &&
    value !==
      null &&
    !Array.isArray(
      value
    )
  );
}


function recordValue(
  source:
    unknown,

  key:
    string
): unknown {
  return isRecord(
    source
  )
    ? source[
        key
      ]
    : undefined;
}


function stringValue(
  source:
    unknown,

  key:
    string
): string |
  null {
  const value =
    recordValue(
      source,
      key
    );


  return typeof value ===
    "string"
      ? value
      : null;
}


function numberValue(
  source:
    unknown,

  key:
    string
): number |
  null {
  const value =
    recordValue(
      source,
      key
    );


  return (
    typeof value ===
      "number" &&
    Number.isFinite(
      value
    )
  )
    ? value
    : null;
}


function booleanValue(
  source:
    unknown,

  key:
    string
): boolean |
  null {
  const value =
    recordValue(
      source,
      key
    );


  return typeof value ===
    "boolean"
      ? value
      : null;
}


function stringArrayValue(
  source:
    unknown,

  key:
    string
): string[] {
  const value =
    recordValue(
      source,
      key
    );


  if (
    !Array.isArray(
      value
    )
  ) {
    return [];
  }


  return value.filter(
    (
      item
    ): item is string =>
      typeof item ===
      "string"
  );
}


function recordArrayValue(
  source:
    unknown,

  key:
    string
): Record<
  string,
  unknown
>[] {
  const value =
    recordValue(
      source,
      key
    );


  if (
    !Array.isArray(
      value
    )
  ) {
    return [];
  }


  return value.filter(
    isRecord
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


  return new Intl
    .DateTimeFormat(
      "fr-FR",
      {
        day:
          "2-digit",

        month:
          "short",

        year:
          "numeric",

        hour:
          "2-digit",

        minute:
          "2-digit",
      }
    )
    .format(
      date
    );
}


function formatCount(
  value:
    number
): string {
  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          0,
      }
    )
    .format(
      value
    );
}


function formatLatency(
  value:
    number |
    null |
    undefined
): string {
  if (
    value ===
      null ||
    value ===
      undefined ||
    !Number.isFinite(
      value
    )
  ) {
    return "—";
  }


  if (
    value >=
    1000
  ) {
    return new Intl
      .NumberFormat(
        "fr-FR",
        {
          maximumFractionDigits:
            2,
        }
      )
      .format(
        value /
        1000
      ) + " s";
  }


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
    ) + " ms";
}


function formatPercentRate(
  value:
    number
): string {
  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        style:
          "percent",

        maximumFractionDigits:
          1,
      }
    )
    .format(
      value
    );
}


function formatPercentNumber(
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
    ) + " %";
}


function compactTraceId(
  traceId:
    string
): string {
  if (
    traceId.length <=
    22
  ) {
    return traceId;
  }


  return (
    traceId.slice(
      0,
      12
    ) +
    "…" +
    traceId.slice(
      -7
    )
  );
}


function statusLabel(
  value:
    string |
    null |
    undefined
): string {
  switch (
    value
  ) {
    case "ready":
      return "Prêt";

    case "validated":
      return "Validé";

    case "executed":
      return "Exécuté";

    case "completed":
      return "Terminée";

    case "failed":
      return "Échec";

    case "blocked":
      return "Bloqué";

    case "ambiguous":
      return "Ambigu";

    case "rejected":
      return "Rejeté";

    case "not_supported":
      return "Non supporté";

    default:
      return value ??
        "—";
  }
}




function statusTone(
  value:
    string |
    null |
    undefined
): "good" |
  "neutral" |
  "bad" {
  if (
    value ===
      "ready" ||
    value ===
      "validated" ||
    value ===
      "executed" ||
    value ===
      "completed"
  ) {
    return "good";
  }


  if (
    value ===
      "blocked" ||
    value ===
      "rejected" ||
    value ===
      "failed"
  ) {
    return "bad";
  }


  return "neutral";
}




function analysisSourceLabel(
  value:
    AnalysisSourceType |
    null |
    undefined
): string {
  switch (
    value
  ) {
    case "initial_request":
      return "Requête initiale";

    case "follow_up_prompt":
      return "Suivi";

    case "document_request":
      return "Document";

    case "automatic":
      return "Automatique";

    default:
      return "—";
  }
}




function MetaBadge({
  label,
  value,
}: {
  label:
    string;

  value:
    string;
}) {
  return (
    <span
      className={
        styles.metaBadge
      }
    >
      <span>
        {
          label
        }
      </span>

      <strong>
        {
          value
        }
      </strong>
    </span>
  );
}


function StatusBadge({
  value,
}: {
  value:
    string |
    null |
    undefined;
}) {
  const tone =
    statusTone(
      value
    );


  return (
    <span
      className={
        `${styles.statusBadge} ${
          tone ===
            "good"
            ? styles.statusGood
            : tone ===
                "bad"
              ? styles.statusBad
              : styles.statusNeutral
        }`
      }
    >
      {
        statusLabel(
          value
        )
      }
    </span>
  );
}


function EmptyState() {
  return (
    <div
      className={
        styles.emptyState
      }
    >
      <span
        className={
          styles.eyebrow
        }
      >
        Aucune exécution
      </span>

      <h2>
        Les traces apparaîtront ici.
      </h2>

      <p>
        Lance une analyse IA depuis le workspace.
        DataLens enregistrera ensuite la chaîne de décision
        locale sans stocker les lignes brutes des datasets.
      </p>

      <Link
        className={
          styles.primaryLink
        }
        href="/"
      >
        Retour au workspace →
      </Link>
    </div>
  );
}


function TraceList({
  traces,
  selectedTraceId,
  onSelect,
}: {
  traces:
    TraceSummary[];

  selectedTraceId:
    string |
    null;

  onSelect:
    (
      traceId:
        string
    ) => void;
}) {
  return (
    <div
      className={
        styles.traceList
      }
    >
      {
        traces.map(
          (
            trace
          ) => (
            <button
              key={
                trace.trace_id
              }
              className={
                `${styles.traceCard} ${
                  selectedTraceId ===
                    trace.trace_id
                    ? styles.traceCardActive
                    : ""
                }`
              }
              type="button"
              onClick={
                () =>
                  onSelect(
                    trace.trace_id
                  )
              }
            >
              <div
                className={
                  styles.traceCardTop
                }
              >
                <span
                  className={
                    styles.traceId
                  }
                  title={
                    trace.trace_id
                  }
                >
                  {
                    compactTraceId(
                      trace.trace_id
                    )
                  }
                </span>

                <StatusBadge
                  value={
                    trace.run_status
                  }
                />
              </div>

              <strong
                className={
                  styles.traceObjective
                }
              >
                {
                  trace.objective
                }
              </strong>

              <div
                className={
                  styles.traceMeta
                }
              >
                <span>
                  {
                    formatDate(
                      trace.created_at_utc
                    )
                  }
                </span>

                <span>
                  {
                    trace.run_status ===
                      "failed" &&
                    trace.failure_stage
                      ? `Étape · ${trace.failure_stage}`
                      : formatLatency(
                          trace.total_ms
                        )
                  }
                </span>
              </div>

              <div
                className={
                  styles.datasetLine
                }
              >
                {
                  trace.dataset_filenames.length >
                    0
                    ? trace.dataset_filenames.join(
                        " · "
                      )
                    : "Dataset non renseigné"
                }
              </div>
            </button>
          )
        )
      }
    </div>
  );
}


function AggregateOverview({
  metrics,
}: {
  metrics:
    TraceMetricsResponse;
}) {
  const sampleLimited =
    metrics.analyzed_trace_count <
    10;


  const latencyDominatedByPlanner =
    metrics.planner_share_median >=
    60;


  const topFamilies =
    metrics.families.slice(
      0,
      4
    );


  const topTools =
    metrics.requested_tools.slice(
      0,
      4
    );


  return (
    <section
      className={
        styles.aggregatePanel
      }
      aria-labelledby="aggregate-observability-title"
    >
      <div
        className={
          styles.aggregateHead
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            VUE GLOBALE · TRACES LOCALES
          </span>

          <h2
            id="aggregate-observability-title"
          >
            Santé du pipeline IA
          </h2>

          <p>
            Agrégation des
            {" "}
            {
              metrics.analyzed_trace_count
            }
            {" "}
            dernière
            {
              metrics.analyzed_trace_count >
              1
                ? "s"
                : ""
            }
            {" trace"}
            {
              metrics.analyzed_trace_count >
              1
                ? "s"
                : ""
            }
            {" valide"}
            {
              metrics.analyzed_trace_count >
              1
                ? "s"
                : ""
            }
            .
          </p>
        </div>

        <span
          className={
            styles.aggregateVersion
          }
        >
          {
            metrics.trace_store_rule_version
          }
        </span>
      </div>


      {
        sampleLimited
          ? (
              <div
                className={
                  styles.sampleNotice
                }
              >
                <strong>
                  Échantillon limité
                </strong>

                <span>
                  Ces métriques sont opérationnelles,
                  mais
                  {" "}
                  {
                    metrics.analyzed_trace_count
                  }
                  {" trace"}
                  {
                    metrics.analyzed_trace_count >
                    1
                      ? "s"
                      : ""
                  }
                  {" ne suffisent pas encore pour conclure sur la fiabilité globale du système."}
                </span>
              </div>
            )
          : null
      }


      <div
        className={
          styles.aggregateMetricGrid
        }
      >
        <article
          className={
            styles.aggregateMetricCard
          }
        >
          <span>
            Terminées
          </span>

          <strong>
            {
              formatCount(
                metrics.completed_trace_count
              )
            }
          </strong>

          <small>
            {
              metrics.completed_trace_count
            }
            {" / "}
            {
              metrics.analyzed_trace_count
            }
            {" traces"}
          </small>
        </article>


        <article
          className={
            styles.aggregateMetricCard
          }
        >
          <span>
            Échecs
          </span>

          <strong>
            {
              formatCount(
                metrics.failed_trace_count
              )
            }
          </strong>

          <small>
            Traces interrompues
          </small>
        </article>


        <article
          className={
            styles.aggregateMetricCard
          }
        >
          <span>
            Taux d’échec
          </span>

          <strong>
            {
              formatPercentRate(
                metrics.failure_rate
              )
            }
          </strong>

          <small>
            Sur les traces analysées
          </small>
        </article>


        <article
          className={
            styles.aggregateMetricCard
          }
        >
          <span>
            Taux d’exécution
          </span>

          <strong>
            {
              formatPercentRate(
                metrics.execution_rate
              )
            }
          </strong>

          <small>
            {
              metrics.executed_trace_count
            }
            {" / "}
            {
              metrics.analyzed_trace_count
            }
            {" traces exécutées"}
          </small>
        </article>


        <article
          className={
            styles.aggregateMetricCard
          }
        >
          <span>
            Retry planner
          </span>

          <strong>
            {
              formatPercentRate(
                metrics.planner_retry_rate
              )
            }
          </strong>

          <small>
            {
              metrics.planner_retry_trace_count
            }
            {" trace(s) avec retry"}
          </small>
        </article>


        <article
          className={
            styles.aggregateMetricCard
          }
        >
          <span>
            Normalisation déterministe
          </span>

          <strong>
            {
              formatPercentRate(
                metrics.planner_normalization_rate
              )
            }
          </strong>

          <small>
            {
              metrics.planner_normalized_trace_count
            }
            {" trace(s) normalisée(s)"}
          </small>
        </article>


        <article
          className={
            styles.aggregateMetricCard
          }
        >
          <span>
            Latence médiane
          </span>

          <strong>
            {
              formatLatency(
                metrics.total_latency.median_ms
              )
            }
          </strong>

          <small>
            {"p95 "}
            {
              formatLatency(
                metrics.total_latency.p95_ms
              )
            }
          </small>
        </article>
      </div>


      {
        metrics.failure_stages.length >
          0
          ? (
              <div
                className={
                  styles.failureStagePanel
                }
              >
                <div>
                  <span
                    className={
                      styles.eyebrow
                    }
                  >
                    ÉCHECS · ÉTAPES
                  </span>

                  <strong>
                    Où les exécutions échouent
                  </strong>
                </div>

                <div
                  className={
                    styles.failureStageTags
                  }
                >
                  {
                    metrics.failure_stages.map(
                      (
                        item
                      ) => (
                        <span
                          key={
                            `failure-stage-${item.name}`
                          }
                        >
                          {
                            item.name
                          }
                          {" · "}
                          {
                            item.count
                          }
                        </span>
                      )
                    )
                  }
                </div>
              </div>
            )
          : null
      }


      <div
        className={
          styles.aggregateBreakdown
        }
      >
        <article
          className={
            styles.aggregateBreakdownCard
          }
        >
          <div
            className={
              styles.breakdownHead
            }
          >
            <div>
              <span
                className={
                  styles.eyebrow
                }
              >
                LATENCE
              </span>

              <strong>
                Répartition médiane
              </strong>
            </div>

            <span
              className={
                latencyDominatedByPlanner
                  ? styles.signalWarn
                  : styles.signalGood
              }
            >
              Planner
              {" "}
              {
                formatPercentNumber(
                  metrics.planner_share_median
                )
              }
            </span>
          </div>


          <div
            className={
              styles.latencyRows
            }
          >
            <div>
              <span>
                Planner
              </span>

              <strong>
                {
                  formatLatency(
                    metrics.planner_latency.median_ms
                  )
                }
              </strong>
            </div>

            <div>
              <span>
                Pipeline natif
              </span>

              <strong>
                {
                  formatLatency(
                    metrics.native_pipeline_latency.median_ms
                  )
                }
              </strong>
            </div>

            <div>
              <span>
                Ingestion
              </span>

              <strong>
                {
                  formatLatency(
                    metrics.ingestion_latency.median_ms
                  )
                }
              </strong>
            </div>
          </div>
        </article>


        <article
          className={
            styles.aggregateBreakdownCard
          }
        >
          <div
            className={
              styles.breakdownHead
            }
          >
            <div>
              <span
                className={
                  styles.eyebrow
                }
              >
                ROUTAGE
              </span>

              <strong>
                Familles et outils observés
              </strong>
            </div>
          </div>


          <div
            className={
              styles.aggregateTags
            }
          >
            {
              topFamilies.length >
              0
                ? topFamilies.map(
                    (
                      item
                    ) => (
                      <span
                        key={
                          `family-${item.name}`
                        }
                      >
                        {
                          item.name
                        }
                        {" · "}
                        {
                          item.count
                        }
                      </span>
                    )
                  )
                : (
                    <span>
                      Aucune famille
                    </span>
                  )
            }
          </div>


          <div
            className={
              styles.aggregateTags
            }
          >
            {
              topTools.length >
              0
                ? topTools.map(
                    (
                      item
                    ) => (
                      <span
                        key={
                          `tool-${item.name}`
                        }
                      >
                        {
                          item.name
                        }
                        {" · "}
                        {
                          item.count
                        }
                      </span>
                    )
                  )
                : (
                    <span>
                      Aucun outil exécuté
                    </span>
                  )
            }
          </div>


          <div
            className={
              styles.modelLine
            }
          >
            <span>
              Planner
            </span>

            <strong>
              {
                metrics.planner_models[
                  0
                ]?.name ??
                "—"
              }
            </strong>

            <span>
              Tool model
            </span>

            <strong>
              {
                metrics.tool_models[
                  0
                ]?.name ??
                "—"
              }
            </strong>
          </div>
        </article>
      </div>


      <article
        className={
          styles.aggregateTimingPanel
        }
      >
        <div
          className={
            styles.breakdownHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              DÉTAIL MÉDIAN · V0.3
            </span>

            <strong>
              Inférence, validation et exécution
            </strong>

            <span
              style={{
                display:
                  "block",

                marginTop:
                  "4px",

                color:
                  "rgba(233, 240, 251, 0.38)",

                fontSize:
                  "0.59rem",
              }}
            >
              {
                formatCount(
                  metrics.detailed_trace_count
                )
              }
              {" trace(s) avec télémétrie détaillée"}
            </span>
          </div>

          <span
            className={
              metrics.planner_model_share_median >=
              60
                ? styles.signalWarn
                : styles.signalGood
            }
          >
            Gemma
            {" "}
            {
              formatPercentNumber(
                metrics.planner_model_share_median
              )
            }
            {" du total"}
          </span>
        </div>


        <div
          className={
            styles.aggregateTimingGrid
          }
        >
          <div>
            <span>
              Prompt planner
            </span>

            <strong>
              {
                formatLatency(
                  metrics.planner_prompt_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Inférence Gemma
            </span>

            <strong>
              {
                formatLatency(
                  metrics.planner_model_inference_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Parsing planner
            </span>

            <strong>
              {
                formatLatency(
                  metrics.planner_structured_parse_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Validation planner
            </span>

            <strong>
              {
                formatLatency(
                  metrics.planner_python_validation_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Prompt tool
            </span>

            <strong>
              {
                formatLatency(
                  metrics.tool_prompt_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Inférence Qwen
            </span>

            <strong>
              {
                formatLatency(
                  metrics.tool_model_inference_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Parsing tool
            </span>

            <strong>
              {
                formatLatency(
                  metrics.tool_response_parse_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Validation tool
            </span>

            <strong>
              {
                formatLatency(
                  metrics.tool_python_validation_latency.median_ms
                )
              }
            </strong>
          </div>

          <div>
            <span>
              Exécution déterministe
            </span>

            <strong>
              {
                formatLatency(
                  metrics.deterministic_execution_latency.median_ms
                )
              }
            </strong>
          </div>
        </div>
      </article>
    </section>
  );
}


type TimingStep = {
  label:
    string;

  detail:
    string;

  value:
    number;
};


function safeTimingValue(
  value:
    number |
    null |
    undefined
): number {
  return (
    typeof value ===
      "number" &&
    Number.isFinite(
      value
    ) &&
    value >
      0
  )
    ? value
    : 0;
}


function timingShare(
  value:
    number,

  total:
    number
): number {
  if (
    total <=
    0
  ) {
    return 0;
  }


  return Math.max(
    0,
    Math.min(
      100,
      (
        value /
        total
      ) *
      100
    )
  );
}


function TimingStage({
  title,
  model,
  totalMs,
  steps,
}: {
  title:
    string;

  model:
    string;

  totalMs:
    number;

  steps:
    TimingStep[];
}) {
  const trackedMs =
    steps.reduce(
      (
        total,
        step
      ) =>
        total +
        step.value,
      0
    );


  const overheadMs =
    Math.max(
      0,
      totalMs -
      trackedMs
    );


  const visibleSteps = [
    ...steps,
    ...(
      overheadMs >
      0.05
        ? [
            {
              label:
                "Orchestration",

              detail:
                "Temps restant du stage",

              value:
                overheadMs,
            },
          ]
        : []
    ),
  ];


  const dominantStep =
    visibleSteps.reduce<
      TimingStep |
      null
    >(
      (
        current,
        step
      ) =>
        (
          current ===
            null ||
          step.value >
            current.value
        )
          ? step
          : current,
      null
    );


  return (
    <article
      className={
        styles.timingStage
      }
    >
      <div
        className={
          styles.timingStageHead
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            {
              model
            }
          </span>

          <h3>
            {
              title
            }
          </h3>
        </div>

        <strong>
          {
            formatLatency(
              totalMs
            )
          }
        </strong>
      </div>


      <div
        className={
          styles.timingStepList
        }
      >
        {
          visibleSteps.map(
            (
              step
            ) => {
              const share =
                timingShare(
                  step.value,
                  totalMs
                );


              return (
                <div
                  className={
                    styles.timingStep
                  }
                  key={
                    `${title}-${step.label}`
                  }
                >
                  <div
                    className={
                      styles.timingStepMeta
                    }
                  >
                    <div>
                      <strong>
                        {
                          step.label
                        }
                      </strong>

                      <span>
                        {
                          step.detail
                        }
                      </span>
                    </div>

                    <div
                      className={
                        styles.timingStepValue
                      }
                    >
                      <strong>
                        {
                          formatLatency(
                            step.value
                          )
                        }
                      </strong>

                      <span>
                        {
                          formatPercentNumber(
                            share
                          )
                        }
                      </span>
                    </div>
                  </div>

                  <div
                    className={
                      styles.timingBar
                    }
                    aria-hidden="true"
                  >
                    <span
                      style={{
                        width:
                          `${Math.max(
                            step.value >
                              0
                              ? 1.2
                              : 0,
                            share
                          )}%`,
                      }}
                    />
                  </div>
                </div>
              );
            }
          )
        }
      </div>


      {
        dominantStep
          ? (
              <p
                className={
                  styles.timingStageSignal
                }
              >
                <strong>
                  Étape dominante :
                </strong>
                {" "}
                {
                  dominantStep.label
                }
                {" · "}
                {
                  formatPercentNumber(
                    timingShare(
                      dominantStep.value,
                      totalMs
                    )
                  )
                }
                {" du stage."}
              </p>
            )
          : null
      }
    </article>
  );
}


function DetailedTimingPanel({
  timings,
  plannerModel,
  toolModel,
}: {
  timings:
    TraceTiming;

  plannerModel:
    string |
    null;

  toolModel:
    string |
    null;
}) {
  const plannerMs =
    safeTimingValue(
      timings.planner_ms
    );

  const nativeMs =
    safeTimingValue(
      timings.native_pipeline_ms
    );


  const plannerSteps:
    TimingStep[] = [
      {
        label:
          "Construction du prompt",

        detail:
          "Préparation du contexte envoyé à Gemma",

        value:
          safeTimingValue(
            timings.planner_prompt_construction_ms
          ),
      },
      {
        label:
          "Inférence Gemma",

        detail:
          "Génération du plan analytique structuré",

        value:
          safeTimingValue(
            timings.planner_model_inference_ms
          ),
      },
      {
        label:
          "Parsing structuré",

        detail:
          "Lecture de la sortie du planner",

        value:
          safeTimingValue(
            timings.planner_structured_parse_ms
          ),
      },
      {
        label:
          "Validation déterministe",

        detail:
          "Contrat, colonnes, rôles et garde-fous",

        value:
          safeTimingValue(
            timings.planner_python_validation_ms
          ),
      },
      {
        label:
          "Préparation retry",

        detail:
          "Feedback construit avant une nouvelle tentative",

        value:
          safeTimingValue(
            timings.planner_retry_feedback_ms
          ),
      },
    ];


  const toolSteps:
    TimingStep[] = [
      {
        label:
          "Construction du prompt",

        detail:
          "Catalogue d’outils et contrat validé",

        value:
          safeTimingValue(
            timings.tool_prompt_construction_ms
          ),
      },
      {
        label:
          "Inférence Qwen",

        detail:
          "Sélection de l’outil et de ses arguments",

        value:
          safeTimingValue(
            timings.tool_model_inference_ms
          ),
      },
      {
        label:
          "Parsing du tool call",

        detail:
          "Lecture de la réponse structurée",

        value:
          safeTimingValue(
            timings.tool_response_parse_ms
          ),
      },
      {
        label:
          "Validation déterministe",

        detail:
          "Outil et arguments contrôlés exactement",

        value:
          safeTimingValue(
            timings.tool_python_validation_ms
          ),
      },
      {
        label:
          "Exécution déterministe",

        detail:
          "Calcul final exécuté par le moteur déterministe",

        value:
          safeTimingValue(
            timings.deterministic_execution_ms
          ),
      },
    ];


  const detailedTotal =
    [
      ...plannerSteps,
      ...toolSteps,
    ].reduce(
      (
        total,
        step
      ) =>
        total +
        step.value,
      0
    );


  if (
    detailedTotal <=
    0
  ) {
    return (
      <section
        className={
          styles.panel
        }
      >
        <div
          className={
            styles.sectionHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              TIMINGS DÉTAILLÉS
            </span>

            <h2>
              Décomposition du pipeline
            </h2>
          </div>
        </div>

        <div
          className={
            styles.timingLegacyNotice
          }
        >
          Cette trace ne contient pas encore
          la télémétrie détaillée v0.3.
          Les timings apparaîtront sur les
          nouvelles exécutions.
        </div>
      </section>
    );
  }


  const plannerInferenceMs =
    safeTimingValue(
      timings.planner_model_inference_ms
    );

  const toolInferenceMs =
    safeTimingValue(
      timings.tool_model_inference_ms
    );

  const totalMs =
    safeTimingValue(
      timings.total_ms
    );


  return (
    <section
      className={
        styles.panel
      }
    >
      <div
        className={
          styles.sectionHead
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            TIMINGS DÉTAILLÉS · TRACE V0.3
          </span>

          <h2>
            Où le temps est réellement passé
          </h2>
        </div>

        <span
          className={
            styles.sectionCount
          }
        >
          {
            formatLatency(
              detailedTotal
            )
          }
        </span>
      </div>


      <div
        className={
          styles.timingSignalGrid
        }
      >
        <article>
          <span>
            Gemma dans le planner
          </span>

          <strong>
            {
              formatPercentNumber(
                timingShare(
                  plannerInferenceMs,
                  plannerMs
                )
              )
            }
          </strong>

          <small>
            {
              formatLatency(
                plannerInferenceMs
              )
            }
            {" d’inférence"}
          </small>
        </article>

        <article>
          <span>
            Qwen dans le pipeline natif
          </span>

          <strong>
            {
              formatPercentNumber(
                timingShare(
                  toolInferenceMs,
                  nativeMs
                )
              )
            }
          </strong>

          <small>
            {
              formatLatency(
                toolInferenceMs
              )
            }
            {" d’inférence"}
          </small>
        </article>

        <article>
          <span>
            Calcul déterministe
          </span>

          <strong>
            {
              formatLatency(
                timings.deterministic_execution_ms
              )
            }
          </strong>

          <small>
            {
              formatPercentNumber(
                timingShare(
                  safeTimingValue(
                    timings.deterministic_execution_ms
                  ),
                  totalMs
                )
              )
            }
            {" du total"}
          </small>
        </article>
      </div>


      <div
        className={
          styles.timingStageGrid
        }
      >
        <TimingStage
          title="Planner"
          model={
            plannerModel ??
            "Gemma"
          }
          totalMs={
            plannerMs
          }
          steps={
            plannerSteps
          }
        />

        <TimingStage
          title="Tool calling + exécution"
          model={
            toolModel ??
            "Qwen + Python"
          }
          totalMs={
            nativeMs
          }
          steps={
            toolSteps
          }
        />
      </div>
    </section>
  );
}


function TraceOverview({
  trace,
}: {
  trace:
    TraceRecord;
}) {
  const planner =
    trace.planner;

  const pipeline =
    trace.native_pipeline;


  const plannerItems =
    recordArrayValue(
      planner,
      "items"
    ) as PlannerItem[];


  const nativeItems =
    recordArrayValue(
      pipeline,
      "items"
    ) as NativeItem[];


  const plannerModel =
    stringValue(
      planner,
      "model"
    );


  const plannerRule =
    stringValue(
      planner,
      "planner_rule_version"
    );


  const plannerStatus =
    stringValue(
      planner,
      "status"
    );


  const attempts =
    numberValue(
      planner,
      "attempt_count"
    );


  const retries =
    numberValue(
      planner,
      "retry_count"
    );


  const normalizations =
    numberValue(
      planner,
      "normalization_count"
    );


  const pipelineStatus =
    stringValue(
      pipeline,
      "status"
    );


  const pipelineRule =
    stringValue(
      pipeline,
      "pipeline_rule_version"
    );


  const toolModel =
    stringValue(
      pipeline,
      "tool_model"
    );


  return (
    <div
      className={
        styles.detailStack
      }
    >
      <section
        className={
          styles.heroPanel
        }
      >
        <div
          className={
            styles.heroTop
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              AI TRACE · LOCAL
            </span>

            <h1>
              {
                trace.objective
              }
            </h1>
          </div>

          <StatusBadge
            value={
              trace.run_status
            }
          />
        </div>

        <div
          className={
            styles.metaRow
          }
        >
          <MetaBadge
            label="Trace"
            value={
              compactTraceId(
                trace.trace_id
              )
            }
          />

          <MetaBadge
            label="Version"
            value={
              trace.trace_rule_version
            }
          />

          <MetaBadge
            label="Créée"
            value={
              formatDate(
                trace.created_at_utc
              )
            }
          />

          {
            trace.analysis_source_type
              ? (
                  <MetaBadge
                    label="Source"
                    value={
                      analysisSourceLabel(
                        trace.analysis_source_type
                      )
                    }
                  />
                )
              : null
          }

          {
            trace.workflow_id
              ? (
                  <MetaBadge
                    label="Workflow"
                    value={
                      trace.workflow_id
                    }
                  />
                )
              : null
          }

          {
            trace.analysis_id
              ? (
                  <MetaBadge
                    label="Analyse"
                    value={
                      trace.analysis_id
                    }
                  />
                )
              : null
          }
        </div>
      </section>


      {
        trace.failure
          ? (
              <section
                className={
                  `${styles.panel} ${styles.failurePanel}`
                }
              >
                <div
                  className={
                    styles.sectionHead
                  }
                >
                  <div>
                    <span
                      className={
                        styles.eyebrow
                      }
                    >
                      ÉCHEC D’EXÉCUTION
                    </span>

                    <h2>
                      Trace interrompue
                    </h2>
                  </div>

                  <StatusBadge
                    value="failed"
                  />
                </div>

                <div
                  className={
                    styles.metaRow
                  }
                >
                  <MetaBadge
                    label="Étape"
                    value={
                      trace.failure.stage
                    }
                  />

                  <MetaBadge
                    label="Type"
                    value={
                      trace.failure.error_type
                    }
                  />
                </div>

                <p
                  className={
                    styles.failureMessage
                  }
                >
                  {
                    trace.failure.message_safe
                  }
                </p>
              </section>
            )
          : null
      }


      <section
        className={
          styles.metricGrid
        }
      >
        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Total
          </span>

          <strong>
            {
              formatLatency(
                trace.timings.total_ms
              )
            }
          </strong>

          <small>
            Latence bout en bout
          </small>
        </article>

        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Planner
          </span>

          <strong>
            {
              formatLatency(
                trace.timings.planner_ms
              )
            }
          </strong>

          <small>
            {
              plannerModel ??
              "Modèle inconnu"
            }
          </small>
        </article>

        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Pipeline natif
          </span>

          <strong>
            {
              formatLatency(
                trace.timings.native_pipeline_ms
              )
            }
          </strong>

          <small>
            {
              toolModel ??
              "Modèle outil inconnu"
            }
          </small>
        </article>

        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Ingestion
          </span>

          <strong>
            {
              formatLatency(
                trace.timings.ingestion_ms
              )
            }
          </strong>

          <small>
            Préparation des schémas
          </small>
        </article>
      </section>


      <DetailedTimingPanel
        timings={
          trace.timings
        }
        plannerModel={
          plannerModel
        }
        toolModel={
          toolModel
        }
      />


      <section
        className={
          styles.panel
        }
      >
        <div
          className={
            styles.sectionHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              DATASETS
            </span>

            <h2>
              Données vues par le pipeline
            </h2>
          </div>

          <span
            className={
              styles.sectionCount
            }
          >
            {
              trace.datasets.length
            }
          </span>
        </div>

        <div
          className={
            styles.datasetGrid
          }
        >
          {
            trace.datasets.map(
              (
                dataset,
                index
              ) => (
                <article
                  className={
                    styles.datasetCard
                  }
                  key={
                    dataset.dataset_id ??
                    `${dataset.filename}-${index}`
                  }
                >
                  <span
                    className={
                      styles.eyebrow
                    }
                  >
                    {
                      dataset.dataset_id ??
                      `dataset ${index + 1}`
                    }
                  </span>

                  <strong>
                    {
                      dataset.filename ??
                      "Dataset"
                    }
                  </strong>

                  <p>
                    {
                      dataset.row_count ??
                      "—"
                    }
                    {" lignes · "}
                    {
                      dataset.column_count ??
                      dataset.columns?.length ??
                      "—"
                    }
                    {" colonnes"}
                  </p>
                </article>
              )
            )
          }
        </div>
      </section>


      <section
        className={
          styles.panel
        }
      >
        <div
          className={
            styles.sectionHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              PLANNER
            </span>

            <h2>
              Proposition et validation
            </h2>
          </div>

          <StatusBadge
            value={
              plannerStatus
            }
          />
        </div>

        <div
          className={
            styles.metaRow
          }
        >
          <MetaBadge
            label="Modèle"
            value={
              plannerModel ??
              "—"
            }
          />

          <MetaBadge
            label="Règle"
            value={
              plannerRule ??
              "—"
            }
          />

          <MetaBadge
            label="Tentatives"
            value={
              String(
                attempts ??
                0
              )
            }
          />

          <MetaBadge
            label="Retries"
            value={
              String(
                retries ??
                0
              )
            }
          />

          <MetaBadge
            label="Normalisations"
            value={
              String(
                normalizations ??
                0
              )
            }
          />
        </div>

        <div
          className={
            styles.flowList
          }
        >
          {
            plannerItems.length >
              0
              ? plannerItems.map(
                  (
                    item,
                    index
                  ) => {
                    const raw =
                      isRecord(
                        item.raw_proposal
                      )
                        ? item.raw_proposal
                        : {};

                    const canonical =
                      isRecord(
                        item.canonical_proposal
                      )
                        ? item.canonical_proposal
                        : {};

                    const contract =
                      isRecord(
                        item.contract
                      )
                        ? item.contract
                        : {};


                    const rawFamily =
                      stringValue(
                        raw,
                        "family"
                      );

                    const canonicalFamily =
                      stringValue(
                        canonical,
                        "family"
                      );

                    const datasetId =
                      stringValue(
                        canonical,
                        "dataset_id"
                      );

                    const blockers =
                      stringArrayValue(
                        contract,
                        "blockers"
                      );


                    return (
                      <article
                        className={
                          styles.flowCard
                        }
                        key={
                          `${trace.trace_id}-planner-${index}`
                        }
                      >
                        <div
                          className={
                            styles.flowCardTop
                          }
                        >
                          <div>
                            <span
                              className={
                                styles.eyebrow
                              }
                            >
                              PROPOSITION
                              {" "}
                              {
                                String(
                                  item.proposal_index ??
                                  index + 1
                                ).padStart(
                                  2,
                                  "0"
                                )
                              }
                            </span>

                            <h3>
                              {
                                canonicalFamily ??
                                rawFamily ??
                                "Analyse"
                              }
                            </h3>
                          </div>

                          <StatusBadge
                            value={
                              item.validation_status
                            }
                          />
                        </div>

                        <div
                          className={
                            styles.decisionGrid
                          }
                        >
                          <div>
                            <span>
                              Proposition brute
                            </span>

                            <strong>
                              {
                                rawFamily ??
                                "—"
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Contrat canonique
                            </span>

                            <strong>
                              {
                                canonicalFamily ??
                                "—"
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Dataset
                            </span>

                            <strong>
                              {
                                datasetId ??
                                "—"
                              }
                            </strong>
                          </div>
                        </div>

                        {
                          (
                            item.normalizations?.length ??
                            0
                          ) >
                          0
                            ? (
                                <div
                                  className={
                                    styles.noteBlock
                                  }
                                >
                                  <strong>
                                    Normalisation déterministe
                                  </strong>

                                  {
                                    item.normalizations?.map(
                                      (
                                        message,
                                        messageIndex
                                      ) => (
                                        <p
                                          key={
                                            `${messageIndex}-${message}`
                                          }
                                        >
                                          {
                                            message
                                          }
                                        </p>
                                      )
                                    )
                                  }
                                </div>
                              )
                            : null
                        }

                        {
                          blockers.length >
                          0
                            ? (
                                <div
                                  className={
                                    styles.warningBlock
                                  }
                                >
                                  <strong>
                                    Blocage
                                  </strong>

                                  {
                                    blockers.map(
                                      (
                                        blocker
                                      ) => (
                                        <p
                                          key={
                                            blocker
                                          }
                                        >
                                          {
                                            blocker
                                          }
                                        </p>
                                      )
                                    )
                                  }
                                </div>
                              )
                            : null
                        }

                        <details
                          className={
                            styles.rawDetails
                          }
                        >
                          <summary>
                            Voir la proposition brute et canonique
                          </summary>

                          <div
                            className={
                              styles.codeGrid
                            }
                          >
                            <pre>
                              {
                                JSON.stringify(
                                  raw,
                                  null,
                                  2
                                )
                              }
                            </pre>

                            <pre>
                              {
                                JSON.stringify(
                                  canonical,
                                  null,
                                  2
                                )
                              }
                            </pre>
                          </div>
                        </details>
                      </article>
                    );
                  }
                )
              : (
                  <p
                    className={
                      styles.muted
                    }
                  >
                    Aucun item planner enregistré.
                  </p>
                )
          }
        </div>
      </section>


      <section
        className={
          styles.panel
        }
      >
        <div
          className={
            styles.sectionHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              TOOL CALLING
            </span>

            <h2>
              Sélection et exécution
            </h2>
          </div>

          <StatusBadge
            value={
              pipelineStatus
            }
          />
        </div>

        <div
          className={
            styles.metaRow
          }
        >
          <MetaBadge
            label="Modèle"
            value={
              toolModel ??
              "—"
            }
          />

          <MetaBadge
            label="Pipeline"
            value={
              pipelineRule ??
              "—"
            }
          />
        </div>

        <div
          className={
            styles.flowList
          }
        >
          {
            nativeItems.length >
              0
              ? nativeItems.map(
                  (
                    item,
                    index
                  ) => {
                    const toolCall =
                      isRecord(
                        item.tool_call
                      )
                        ? item.tool_call
                        : {};

                    const execution =
                      isRecord(
                        item.execution
                      )
                        ? item.execution
                        : {};


                    const expectedTool =
                      stringValue(
                        toolCall,
                        "expected_tool"
                      );

                    const requestedTool =
                      stringValue(
                        toolCall,
                        "requested_tool"
                      );

                    const validationStatus =
                      stringValue(
                        toolCall,
                        "validation_status"
                      );

                    const toolAttempts =
                      numberValue(
                        toolCall,
                        "attempt_count"
                      );

                    const toolRetries =
                      numberValue(
                        toolCall,
                        "retry_count"
                      );

                    const executionStatus =
                      stringValue(
                        execution,
                        "execution_status"
                      );

                    const datasetFilename =
                      stringValue(
                        execution,
                        "dataset_filename"
                      );

                    const chartType =
                      stringValue(
                        execution,
                        "chart_type"
                      );

                    const ruleVersion =
                      stringValue(
                        execution,
                        "execution_rule_version"
                      );


                    const requestedArguments =
                      recordValue(
                        toolCall,
                        "requested_arguments"
                      );


                    return (
                      <article
                        className={
                          styles.flowCard
                        }
                        key={
                          `${trace.trace_id}-native-${index}`
                        }
                      >
                        <div
                          className={
                            styles.flowCardTop
                          }
                        >
                          <div>
                            <span
                              className={
                                styles.eyebrow
                              }
                            >
                              OUTIL
                              {" "}
                              {
                                String(
                                  index + 1
                                ).padStart(
                                  2,
                                  "0"
                                )
                              }
                            </span>

                            <h3>
                              {
                                requestedTool ??
                                expectedTool ??
                                item.family ??
                                "Aucun outil"
                              }
                            </h3>
                          </div>

                          <StatusBadge
                            value={
                              item.pipeline_status
                            }
                          />
                        </div>

                        <div
                          className={
                            styles.decisionGrid
                          }
                        >
                          <div>
                            <span>
                              Attendu
                            </span>

                            <strong>
                              {
                                expectedTool ??
                                "—"
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Demandé
                            </span>

                            <strong>
                              {
                                requestedTool ??
                                "—"
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Validation
                            </span>

                            <strong>
                              {
                                statusLabel(
                                  validationStatus
                                )
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Tentatives
                            </span>

                            <strong>
                              {
                                toolAttempts ??
                                0
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Retries
                            </span>

                            <strong>
                              {
                                toolRetries ??
                                0
                              }
                            </strong>
                          </div>
                        </div>

                        <div
                          className={
                            styles.executionStrip
                          }
                        >
                          <div>
                            <span>
                              Exécution
                            </span>

                            <strong>
                              {
                                statusLabel(
                                  executionStatus
                                )
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Dataset
                            </span>

                            <strong>
                              {
                                datasetFilename ??
                                "—"
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Visualisation
                            </span>

                            <strong>
                              {
                                chartType ??
                                "—"
                              }
                            </strong>
                          </div>

                          <div>
                            <span>
                              Règle
                            </span>

                            <strong>
                              {
                                ruleVersion ??
                                "—"
                              }
                            </strong>
                          </div>
                        </div>

                        {
                          isRecord(
                            requestedArguments
                          )
                            ? (
                                <details
                                  className={
                                    styles.rawDetails
                                  }
                                >
                                  <summary>
                                    Voir les arguments validés
                                  </summary>

                                  <pre
                                    className={
                                      styles.singleCode
                                    }
                                  >
                                    {
                                      JSON.stringify(
                                        requestedArguments,
                                        null,
                                        2
                                      )
                                    }
                                  </pre>
                                </details>
                              )
                            : null
                        }
                      </article>
                    );
                  }
                )
              : (
                  <p
                    className={
                      styles.muted
                    }
                  >
                    Aucun appel d’outil enregistré.
                  </p>
                )
          }
        </div>
      </section>


      <section
        className={
          styles.panel
        }
      >
        <div
          className={
            styles.sectionHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              PRIVACY
            </span>

            <h2>
              Périmètre de la trace
            </h2>
          </div>
        </div>

        <div
          className={
            styles.privacyGrid
          }
        >
          <article>
            <span>
              Lignes brutes
            </span>

            <strong>
              {
                trace.privacy
                  .contains_raw_dataset_rows
                  ? "Présentes"
                  : "Non stockées"
              }
            </strong>
          </article>

          <article>
            <span>
              Contenu uploadé
            </span>

            <strong>
              {
                trace.privacy
                  .contains_uploaded_file_contents
                  ? "Présent"
                  : "Non stocké"
              }
            </strong>
          </article>

          <article>
            <span>
              Chunks documentaires
            </span>

            <strong>
              {
                trace.privacy
                  .contains_document_chunks
                  ? "Présents"
                  : "Non stockés"
              }
            </strong>
          </article>

          <article>
            <span>
              Stockage
            </span>

            <strong>
              {
                trace.privacy
                  .storage_scope ??
                "local_jsonl"
              }
            </strong>
          </article>
        </div>

        <p
          className={
            styles.privacyNote
          }
        >
          DataLens stocke localement les métadonnées
          analytiques et la trace des décisions IA.
          Les lignes brutes des datasets, le contenu
          des fichiers uploadés et les chunks
          documentaires ne sont pas enregistrés
          dans cette trace d’observabilité.
        </p>
      </section>
    </div>
  );
}


export default function ObservabilityClient() {
  const router =
    useRouter();


  const searchParams =
    useSearchParams();


  const requestedTraceId =
    searchParams.get(
      "trace"
    );


  const [
    traceList,
    setTraceList,
  ] = useState<
    TraceListResponse |
    null
  >(
    null
  );


  const [
    metrics,
    setMetrics,
  ] = useState<
    TraceMetricsResponse |
    null
  >(
    null
  );


  const [
    selectedTrace,
    setSelectedTrace,
  ] = useState<
    TraceRecord |
    null
  >(
    null
  );


  const [
    selectedTraceId,
    setSelectedTraceId,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    loading,
    setLoading,
  ] = useState(
    true
  );


  const [
    hydrated,
    setHydrated,
  ] = useState(
    false
  );


  const [
    detailLoading,
    setDetailLoading,
  ] = useState(
    false
  );


  const [
    error,
    setError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const loadTrace = useCallback(
    async (
      traceId:
        string
    ) => {
      setDetailLoading(
        true
      );


      try {
        const response =
          await fetch(
            `${API_URL}/observability/traces/${encodeURIComponent(
              traceId
            )}`,
            {
              cache:
                "no-store",
            }
          );


        if (
          !response.ok
        ) {
          throw new Error(
            `Impossible de charger la trace (${response.status}).`
          );
        }


        const trace =
          await response.json() as
            TraceRecord;


        setSelectedTrace(
          trace
        );

        setSelectedTraceId(
          trace.trace_id
        );


      } catch (
        caught
      ) {
        setError(
          caught instanceof Error
            ? caught.message
            : "Erreur inconnue."
        );


      } finally {
        setDetailLoading(
          false
        );
      }
    },
    []
  );


  const loadAll = useCallback(
    async () => {
      setLoading(
        true
      );

      setError(
        null
      );


      try {
        const [
          tracesResponse,
          metricsResponse,
        ] =
          await Promise.all(
            [
              fetch(
                `${API_URL}/observability/traces?limit=50`,
                {
                  cache:
                    "no-store",
                }
              ),

              fetch(
                `${API_URL}/observability/metrics?limit=200`,
                {
                  cache:
                    "no-store",
                }
              ),
            ]
          );


        if (
          !tracesResponse.ok
        ) {
          throw new Error(
            `Impossible de charger les traces (${tracesResponse.status}).`
          );
        }


        if (
          !metricsResponse.ok
        ) {
          throw new Error(
            `Impossible de charger les métriques (${metricsResponse.status}).`
          );
        }


        const listing =
          await tracesResponse.json() as
            TraceListResponse;


        const aggregateMetrics =
          await metricsResponse.json() as
            TraceMetricsResponse;


        setTraceList(
          listing
        );


        setMetrics(
          aggregateMetrics
        );


        const requestedTraceExists =
          requestedTraceId !==
            null &&
          listing.traces.some(
            (
              trace
            ) =>
              trace.trace_id ===
              requestedTraceId
          );


        const selectedTraceExists =
          selectedTraceId !==
            null &&
          listing.traces.some(
            (
              trace
            ) =>
              trace.trace_id ===
              selectedTraceId
          );


        const preferredTraceId =
          requestedTraceExists
            ? requestedTraceId
            : selectedTraceExists
              ? selectedTraceId
              : listing.traces[
                  0
                ]?.trace_id ??
                null;


        if (
          preferredTraceId
        ) {
          await loadTrace(
            preferredTraceId
          );


        } else {
          setSelectedTrace(
            null
          );

          setSelectedTraceId(
            null
          );
        }


      } catch (
        caught
      ) {
        setError(
          caught instanceof Error
            ? caught.message
            : "Erreur inconnue."
        );


      } finally {
        setLoading(
          false
        );
      }
    },
    [
      loadTrace,
      requestedTraceId,
      selectedTraceId,
    ]
  );


  useEffect(
    () => {
      setHydrated(
        true
      );

      void loadAll();
    },
    // Initial load only. Refreshes are explicit to avoid
    // continuously polling a local development API.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );


  const plannerShare =
    useMemo(
      () => {
        if (
          !selectedTrace
        ) {
          return null;
        }


        const total =
          selectedTrace
            .timings
            .total_ms ??
          0;

        const planner =
          selectedTrace
            .timings
            .planner_ms ??
          0;


        if (
          total <=
          0
        ) {
          return null;
        }


        return (
          planner /
          total
        ) *
        100;
      },
      [
        selectedTrace,
      ]
    );


  const workspaceWorkflowId =
    hydrated
      ? readActivePreparationWorkflowId()
      : null;


  function handleWorkspaceStepChange(
    step:
      WorkspaceStep
  ): void {
    const activeWorkflowId =
      readActivePreparationWorkflowId();


    if (
      activeWorkflowId
    ) {
      persistActiveWorkspaceStep(
        activeWorkflowId,
        step
      );
    }


    router.push(
      "/"
    );
  }


  return (
    <main
      className={
        workspaceStyles.page
      }
    >
      <header
        className={
          workspaceStyles.header
        }
      >
        <Link
          href="/"
          aria-label="DataLens - workspace"
          className={
            workspaceStyles.brand
          }
        >
          <span
            className={
              workspaceStyles.brandMark
            }
            aria-hidden="true"
          >
            <svg
              className={
                workspaceStyles.brandMarkSvg
              }
              viewBox="0 0 28 28"
              focusable="false"
              aria-hidden="true"
            >
              <path
                className={
                  workspaceStyles.brandMarkOutline
                }
                d="M7 5 H13 C18.2 5 21.5 8.2 21.5 11.1"
              />

              <path
                className={
                  workspaceStyles.brandMarkOutline
                }
                d="M21.5 16.9 C21.5 19.8 18.2 23 13 23 H7 V5"
              />

              <circle
                className={
                  workspaceStyles.brandMarkSignal
                }
                cx="21.5"
                cy="14"
                r="1.75"
              />
            </svg>
          </span>

          <strong>
            DataLens
          </strong>
        </Link>


        <div
          className={
            workspaceStyles.privacyStatus
          }
        >
          <span
            aria-hidden="true"
            className={
              workspaceStyles.statusDot
            }
          />

          <span>
            Traitement local · données privées
          </span>
        </div>
      </header>


      <WorkspaceNavigation
        activeStep={
          null
        }
        onStepChange={
          handleWorkspaceStepChange
        }
        dataReady={
          Boolean(
            workspaceWorkflowId
          )
        }
        reportReady={
          Boolean(
            workspaceWorkflowId
          )
        }
        interventionCount={
          0
        }
        activeAiTool="observability"
      />


      <div
        className={
          `${workspaceStyles.shell} ${styles.unifiedShell}`
        }
      >
        <div
          className={
            styles.unifiedToolbar
          }
        >
          <div>
            <span
              className={
                styles.unifiedEyebrow
              }
            >
              AI ENGINEERING · CONTROL ROOM
            </span>

            <h1
              className={
                styles.unifiedTitle
              }
            >
              Observabilité
            </h1>

            <p
              className={
                styles.unifiedSubtitle
              }
            >
              Surveiller, diagnostiquer et tracer
              les exécutions analytiques locales.
            </p>
          </div>


          <button
            className={
              styles.refreshButton
            }
            type="button"
            onClick={
              () =>
                void loadAll()
            }
            disabled={
              !hydrated ||
              loading
            }
          >
            {
              !hydrated ||
              loading
                ? "Actualisation?"
                : "Actualiser"
            }
          </button>
        </div>


        <div
          className={
            styles.shell
          }
        >
        <aside
          className={
            styles.sidebar
          }
        >
          <div
            className={
              styles.sidebarHead
            }
          >
            <div>
              <span
                className={
                  styles.eyebrow
                }
              >
                EXÉCUTIONS
              </span>

              <h2>
                Traces locales
              </h2>
            </div>

            <span
              className={
                styles.traceCount
              }
            >
              {
                traceList?.trace_count ??
                0
              }
            </span>
          </div>

          {
            traceList
              ? (
                  <div
                    className={
                      styles.storeMeta
                    }
                  >
                    <span>
                      {
                        traceList
                          .trace_store_rule_version
                      }
                    </span>

                    <span>
                      {
                        traceList
                          .malformed_line_count
                      }
                      {" "}
                      {
                        traceList
                          .malformed_line_count ===
                        1
                          ? "ligne invalide"
                          : "lignes invalides"
                      }
                    </span>
                  </div>
                )
              : null
          }

          {
            traceList &&
            traceList.traces.length >
            0
              ? (
                  <TraceList
                    traces={
                      traceList.traces
                    }
                    selectedTraceId={
                      selectedTraceId
                    }
                    onSelect={
                      (
                        traceId
                      ) => {
                        setError(
                          null
                        );


                        router.replace(
                          `/observability?trace=${encodeURIComponent(
                            traceId
                          )}`,
                          {
                            scroll:
                              false,
                          }
                        );


                        void loadTrace(
                          traceId
                        );
                      }
                    }
                  />
                )
              : loading
                ? (
                    <p
                      className={
                        styles.muted
                      }
                    >
                      Chargement des traces…
                    </p>
                  )
                : (
                    <p
                      className={
                        styles.muted
                      }
                    >
                      Aucune trace enregistrée.
                    </p>
                  )
          }
        </aside>


        <section
          className={
            styles.content
          }
        >
          {
            error
              ? (
                  <div
                    className={
                      styles.errorPanel
                    }
                    role="alert"
                  >
                    <strong>
                      Observabilité indisponible
                    </strong>

                    <p>
                      {
                        error
                      }
                    </p>

                    <p>
                      Vérifie que l’API DataLens est démarrée sur
                      {" "}
                      {
                        API_URL
                      }.
                    </p>
                  </div>
                )
              : null
          }


          {
            metrics &&
            metrics.analyzed_trace_count >
              0
              ? (
                  <AggregateOverview
                    metrics={
                      metrics
                    }
                  />
                )
              : null
          }


          {
            plannerShare !==
              null &&
            plannerShare >=
              60
              ? (
                  <div
                    className={
                      styles.performanceSignal
                    }
                  >
                    <span
                      className={
                        styles.eyebrow
                      }
                    >
                      SIGNAL PERFORMANCE
                    </span>

                    <strong>
                      Le planner représente
                      {" "}
                      {
                        new Intl
                          .NumberFormat(
                            "fr-FR",
                            {
                              maximumFractionDigits:
                                0,
                            }
                          )
                          .format(
                            plannerShare
                          )
                      }
                      {" % "}
                      de la latence observée sur cette trace.
                    </strong>
                  </div>
                )
              : null
          }


          {
            detailLoading
              ? (
                  <div
                    className={
                      styles.loadingPanel
                    }
                  >
                    Chargement de la trace…
                  </div>
                )
              : selectedTrace
                ? (
                    <TraceOverview
                      trace={
                        selectedTrace
                      }
                    />
                  )
                : !loading
                  ? (
                      <EmptyState />
                    )
                  : (
                      <div
                        className={
                          styles.loadingPanel
                        }
                      >
                        Chargement de l’observabilité…
                      </div>
                    )
          }
        </section>
      </div>
      </div>
    </main>
  );
}
