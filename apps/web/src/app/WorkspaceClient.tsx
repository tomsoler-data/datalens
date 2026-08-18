"use client";

import {
  ChangeEvent,
  FormEvent,
  useEffect,
  useMemo,
  useState,
} from "react";

import type {
  ReactNode,
} from "react";

import Link from "next/link";

import PreparationWorkflowPanel from "../components/preparation/PreparationWorkflowPanel";
import PreparationFinalizationPanel from "../components/preparation/PreparationFinalizationPanel";
import PreparationSubstepNavigation from "../components/preparation/PreparationSubstepNavigation";
import PreparationTransformPanel from "../components/preparation/PreparationTransformPanel";
import SemanticConfirmationPanel from "../components/preparation/SemanticConfirmationPanel";

import type {
  PreparationSubstep,
} from "../components/preparation/PreparationSubstepNavigation";

import {
  createPreparationSession,
  getPreparationSession,
  validatePreparationSession,
} from "../components/preparation/preparationApi";

import {
  confirmSemanticReview,
  SemanticConfirmationApiError,
} from "../components/preparation/semanticConfirmationApi";

import type {
  SemanticConfirmationReportView,
} from "../components/preparation/semanticConfirmationApi";

import type {
  PreparationSessionView,
} from "../components/preparation/preparationTypes";

import styles from "./page.module.css";

import type {
  ContextualizedAnalysisResponse,
  DatasetManifest,
  FindingRagContext,
  MultiDatasetIngestion,
  RagContextReport,
  ReportBlockedAnalysis,
  ReportChartDatum,
  ReportFinding,
  ReportQualityItem,
  ReportRequestedFinding,
  UnifiedAnalysisReport,
} from "./types";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


type ChartPoint = {
  x: number;
  y: number;
};


type LineBandPoint = {
  period: number;
  median: number;
  q1: number;
  q3: number;
};


type GroupSummaryPoint = {
  group: string;
  median: number;
};


type SignalKpi = {
  key: string;
  label: string;
  value: string;
  context: string;
};


type DocumentCitationView = {
  chunk_id:
    string;

  document_id:
    string;

  filename:
    string;

  source_locator:
    string;

  page_number:
    number |
    null;
};


type DocumentClaimView = {
  category:
    string;

  statement:
    string;

  evidence_quote:
    string;

  evidence_unit_id:
    number;

  context_quote:
    string |
    null;

  context_evidence_unit_id:
    number |
    null;

  citation:
    DocumentCitationView;
};


type DocumentSummaryItemView = {
  document_id:
    string;

  filename:
    string;

  summary_points:
    DocumentClaimView[];

  analytical_requests:
    DocumentClaimView[];

  verified_claim_count:
    number;

  source_chunk_count:
    number;
};


type DocumentSummaryView = {
  status:
    string;

  document_count:
    number;

  chunk_count:
    number;

  verified_claim_count:
    number;

  summary_point_count:
    number;

  analytical_request_count:
    number;

  summary_points:
    DocumentClaimView[];

  analytical_requests:
    DocumentClaimView[];

  documents:
    DocumentSummaryItemView[];

  warnings:
    string[];

  abstention_reason:
    string |
    null;

  model:
    string;
};


type RequestedPlanItemView = {
  request_id:
    string;

  request_text:
    string;

  source_filename:
    string;

  source_locator:
    string;

  page_number:
    number |
    null;

  kind:
    string;

  status:
    "ready" |
    "blocked" |
    "ambiguous";

  blockers:
    string[];
};


type RequestedPlanView = {
  request_count:
    number;

  ready_count:
    number;

  blocked_count:
    number;

  ambiguous_count:
    number;

  requests:
    RequestedPlanItemView[];
};


type AIPlannerBindingView = {
  role:
    string;

  column:
    string;

  dataset_id:
    string |
    null;

  dataset_filename:
    string |
    null;

  semantic_concept:
    string |
    null;

  analysis_kind:
    string |
    null;
};


type AIPlannerContractView = {
  contract_id:
    string;

  contract_version:
    string;

  origin:
    string;

  status:
    string;

  title:
    string;

  request_text:
    string;

  family:
    string;

  required_dataset_ids:
    string[];

  required_dataset_filenames:
    string[];

  analytical_grain:
    string |
    null;

  bindings:
    AIPlannerBindingView[];

  reasons:
    string[];

  blockers:
    string[];

  planner_confidence:
    number |
    null;
};


type AIPlannerProposalView = {
  decision:
    string;

  title:
    string;

  family:
    string;

  dataset_id:
    string |
    null;

  analytical_grain:
    string |
    null;

  x_column:
    string |
    null;

  y_column:
    string |
    null;

  group_column:
    string |
    null;

  value_column:
    string |
    null;

  time_column:
    string |
    null;

  dimension_column:
    string |
    null;

  entity_column:
    string |
    null;

  aggregation_function:
    string;

  ranking_order:
    string;

  ranking_limit:
    number |
    null;

  window_operation:
    string;

  window_size:
    number |
    null;

  blockers:
    string[];

  reasons:
    string[];

  confidence:
    number;
};


type AIPlannerItemView = {
  proposal_index:
    number;

  validation_status:
    "validated" |
    "blocked" |
    "ambiguous" |
    "rejected";

  raw_proposal?:
    AIPlannerProposalView |
    null;

  proposal:
    AIPlannerProposalView;

  contract:
    AIPlannerContractView |
    null;

  errors:
    string[];

  warnings:
    string[];

  normalizations?:
    string[];
};


type AIPlannerReportView = {
  status:
    string;

  objective:
    string;

  model:
    string;

  proposal_count:
    number;

  validated_count:
    number;

  blocked_count:
    number;

  ambiguous_count:
    number;

  rejected_count:
    number;

  items:
    AIPlannerItemView[];

  attempt_count?:
    number;

  retry_count?:
    number;

  retry_triggered?:
    boolean;

  retry_feedback?:
    string[];

  normalization_count?:
    number;

  normalization_applied?:
    boolean;

  planner_rule_version:
    string;
};


type AINativeAttemptView = {
  attempt_index:
    number;

  prompt_variant:
    "standard" |
    "mandatory_retry";

  tool_call_count:
    number;

  assistant_content:
    string;

  selected_tool_name:
    string |
    null;

  errors:
    string[];
};


type AINativeExecutionResultView = {
  analysis_id?:
    string;

  dataset_id?:
    string;

  dataset_filename?:
    string;

  title:
    string;

  family:
    string;

  execution_status:
    string;

  chart_type:
    string;

  summary:
    string[];

  metrics:
    Record<
      string,
      unknown
    >;

  chart_data?:
    ReportChartDatum[];

  statistical_decision?:
    Record<
      string,
      unknown
    > |
    null;

  statistical_result?:
    Record<
      string,
      unknown
    > |
    null;

  warnings:
    string[];

  limitations:
    string[];

  execution_rule_version?:
    string;
};


type AINativeExecutionTraceView = {
  tool_name:
    string |
    null;

  execution_status:
    string;

  dataset_id:
    string |
    null;

  dataset_filename:
    string |
    null;

  arguments:
    {
      family?:
        string;

      dataset_ids?:
        string[];

      analytical_grain?:
        string |
        null;

      variables?:
        Record<
          string,
          string
        >;
    };

  result:
    AINativeExecutionResultView |
    null;

  errors:
    string[];

  warnings:
    string[];
};


type AINativeToolView = {
  model:
    string;

  contract_family?:
    string;

  available_tools?:
    string[];

  expected_tool?:
    string |
    null;

  tool_call_received:
    boolean;

  requested_tool:
    string |
    null;

  requested_arguments:
    Record<
      string,
      unknown
    >;

  validation_status:
    "validated" |
    "rejected";

  validation_errors:
    string[];

  attempt_count:
    number;

  retry_count:
    number;

  attempts:
    AINativeAttemptView[];

  execution:
    AINativeExecutionTraceView |
    null;

  native_tool_rule_version:
    string;
};


type AINativePipelineItemView = {
  contract_id:
    string;

  family:
    string;

  pipeline_status:
    "executed" |
    "not_supported" |
    "rejected";

  native_tool:
    AINativeToolView |
    null;

  errors:
    string[];

  warnings:
    string[];
};


type AINativePipelineReportView = {
  trace_id?:
    string |
    null;

  status:
    string;

  planner:
    AIPlannerReportView;

  planner_model:
    string;

  tool_model:
    string;

  supported_native_families?:
    string[];

  validated_contract_count:
    number;

  pipeline_item_count:
    number;

  executed_count:
    number;

  not_supported_count:
    number;

  rejected_count:
    number;

  items:
    AINativePipelineItemView[];

  notes:
    string[];

  pipeline_rule_version:
    string;
};


/* ============================================================
   ENTITY OUTLIERS — ROUTED USER-FACING FINDING
============================================================ */

type EntityOutlierFindingEvidenceView = {
  metric:
    string;

  metric_label:
    string;

  family:
    string;

  family_label:
    string;

  value:
    number;

  direction:
    string;

  distance_iqr:
    number;
};


type EntityOutlierFindingProfileView = {
  entity:
    string;

  severity:
    string;

  dominant_family:
    string;

  dominant_family_label:
    string;

  signal_count:
    number;

  max_distance_iqr:
    number;

  title:
    string;

  explanation:
    string;

  evidence:
    EntityOutlierFindingEvidenceView[];
};


type EntityOutlierFindingView = {
  analysis_id:
    string;

  status:
    "ready" |
    "blocked";

  title:
    string;

  family:
    "entity_outlier";

  kind:
    "customer_entity_outlier_detection";

  dataset_id:
    string |
    null;

  dataset_filename:
    string |
    null;

  entity_column:
    string |
    null;

  entity_count:
    number;

  raw_flagged_entity_count:
    number;

  priority_profile_count:
    number;

  behavioral_signal_count:
    number;

  summary:
    string[];

  priority_profiles:
    EntityOutlierFindingProfileView[];

  caveats:
    string[];

  methodology:
    string[];

  blockers:
    string[];

  adapter_rule_version:
    string;
};


type RoutedUnifiedAnalysisReportView =
  UnifiedAnalysisReport & {
    entity_outlier_finding?:
      EntityOutlierFindingView |
      null;
  };


type RoutedContextualizedAnalysisResponseView =
  Omit<
    ContextualizedAnalysisResponse,
    "analysis"
  > & {
    analysis:
      RoutedUnifiedAnalysisReportView;
  };


const FRIENDLY_VARIABLE_LABELS:
  Record<string, string> = {
    "Population using at least basic drinking-water services (%)":
      "Accès basique à l’eau potable",

    "Population using safely managed drinking-water services (%)":
      "Accès sécurisé à l’eau potable",

    "Mortality rate attributed to exposure to unsafe WASH services":
      "Mortalité liée à l’eau et aux services WASH",

    "Political Stability":
      "Stabilité politique",

    "Political_Stability":
      "Stabilité politique",

    "WASH deaths":
      "Décès liés aux services WASH",

    Population:
      "Population",

    Year:
      "Année",

    Country:
      "Pays",

    Granularity:
      "Zone",

    "REGION (DISPLAY)":
      "Région",

    "COUNTRY (DISPLAY)":
      "Pays",

    Age:
      "Âge",

    Basket:
      "Panier",

    age_at_first_purchase:
      "Âge au premier achat",

    total_spend:
      "Montant total des achats",

    purchase_sessions:
      "Fréquence d’achat",

    average_basket:
      "Panier moyen",

    median_basket:
      "Panier médian",

    sum_price:
      "Montant agrégé",

    price:
      "Prix",

    event_count:
      "Nombre d’événements",

    categ:
      "Catégorie",

    category:
      "Catégorie",

    gender:
      "Genre",

    customer_id:
      "Client",

    event_time:
      "Date d’achat",
  };


function formatNumber(
  value: number
): string {
  return new Intl
    .NumberFormat(
      "fr-FR"
    )
    .format(
      value
    );
}


function formatDecimal(
  value: number
): string {
  if (
    Math.abs(value) <
      0.001 &&
    value !==
      0
  ) {
    return value
      .toExponential(
        2
      );
  }


  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          3,
      }
    )
    .format(
      value
    );
}


function formatChartNumber(
  value: number
): string {
  const absolute =
    Math.abs(
      value
    );


  if (
    absolute >=
      1000
  ) {
    return new Intl
      .NumberFormat(
        "fr-FR",
        {
          notation:
            "compact",

          maximumFractionDigits:
            1,
        }
      )
      .format(
        value
      );
  }


  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          2,
      }
    )
    .format(
      value
    );
}


function formatAxisNumber(
  value: number
): string {
  const absolute =
    Math.abs(
      value
    );


  if (
    absolute >=
    1_000_000
  ) {
    return new Intl
      .NumberFormat(
        "fr-FR",
        {
          notation:
            "compact",

          maximumFractionDigits:
            1,
        }
      )
      .format(
        value
      );
  }


  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          absolute <
            10
            ? 2
            : 1,
      }
    )
    .format(
      value
    );
}


function formatPercent(
  value: number
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


function formatBytes(
  bytes: number
): string {
  if (
    bytes <
    1024
  ) {
    return `${bytes} o`;
  }


  if (
    bytes <
    1024 * 1024
  ) {
    return `${(
      bytes /
      1024
    ).toFixed(
      1
    )} Ko`;
  }


  return `${(
    bytes /
    (
      1024 *
      1024
    )
  ).toFixed(
    1
  )} Mo`;
}


function formatTemporalDisplayValue(
  value:
    unknown
): string {
  if (
    value ===
      null ||
    value ===
      undefined
  ) {
    return "—";
  }


  if (
    typeof value ===
      "number"
  ) {
    if (
      Number.isInteger(
        value
      ) &&
      value >=
        1000 &&
      value <=
        9999
    ) {
      return String(
        value
      );
    }


    return formatChartNumber(
      value
    );
  }


  const raw =
    String(
      value
    ).trim();


  if (
    !raw
  ) {
    return "—";
  }


  if (
    /^\d{4}$/.test(
      raw
    )
  ) {
    return raw;
  }


  const looksIsoDate =
    /^\d{4}-\d{2}-\d{2}(?:[T\s].*)?$/.test(
      raw
    );


  if (
    !looksIsoDate
  ) {
    return raw;
  }


  const parsed =
    new Date(
      raw
    );


  if (
    Number.isNaN(
      parsed.getTime()
    )
  ) {
    return raw;
  }


  return new Intl.DateTimeFormat(
    "fr-FR",
    {
      day:
        "2-digit",

      month:
        "short",

      year:
        "numeric",

      timeZone:
        "UTC",
    }
  ).format(
    parsed
  );
}


function friendlyVariableLabel(
  value: string
): string {
  const direct =
    FRIENDLY_VARIABLE_LABELS[
      value
    ];


  if (
    direct
  ) {
    return direct;
  }


  return value
    .replace(
      /\s*\(%\)\s*$/i,
      ""
    )
    .replace(
      /_/g,
      " "
    )
    .trim();
}


function analysisKindLabel(
  value: string
): string {
  switch (
    value
  ) {
    case "quantitative":
      return "Quantitative";

    case "temporal":
      return "Temporelle";

    case "categorical":
      return "Catégorielle";

    case "boolean":
      return "Booléenne";

    default:
      return "À déterminer";
  }
}


function familyLabel(
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


function isDeterministicPlannerModel(
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


function plannerEngineLabel(
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


function toolEngineLabel(
  model:
    string |
    null |
    undefined
): string {
  if (
    !model
  ) {
    return "IA locale";
  }


  if (
    model
      .toLowerCase()
      .includes(
        "qwen"
      )
  ) {
    return "Qwen · tool calling local";
  }


  return model;
}


function plannerUiCopy(
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


function metricNumber(
  metrics:
    Record<
      string,
      unknown
    >,

  key: string
): number | null {
  const value =
    metrics[
      key
    ];


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


function metricString(
  metrics:
    Record<
      string,
      unknown
    >,

  key: string
): string | null {
  const value =
    metrics[
      key
    ];


  return typeof value ===
    "string"
      ? value
      : null;
}


function datumNumber(
  datum:
    ReportChartDatum,

  key: string
): number | null {
  const value =
    datum[
      key
    ];


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



function datumLabel(
  datum:
    ReportChartDatum,

  key: string
): string | null {
  const value =
    datum[
      key
    ];


  if (
    typeof value ===
    "string"
  ) {
    const trimmed =
      value.trim();


    return trimmed
      ? trimmed
      : null;
  }


  if (
    typeof value ===
      "number" &&
    Number.isFinite(
      value
    )
  ) {
    return formatDecimal(
      value
    );
  }


  return null;
}


function buildSignalKpis(
  report:
    UnifiedAnalysisReport
): SignalKpi[] {
  const output:
    SignalKpi[] = [];


  for (
    const finding
    of report.main_findings
  ) {
    if (
      finding.family ===
      "time_series"
    ) {
      const change =
        metricNumber(
          finding.metrics,
          "median_change"
        );

      const measure =
        metricString(
          finding.metrics,
          "measure_column"
        );


      if (
        change !==
        null
      ) {
        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-change`,

            label:
              "Évolution médiane",

            value:
              formatDecimal(
                change
              ),

            context:
              measure
                ? friendlyVariableLabel(
                    measure
                  )
                : finding.title,
          }
        );
      }
    }


    if (
      finding.family ===
      "derived_gap"
    ) {
      const gap =
        metricNumber(
          finding.metrics,
          "median_gap"
        );


      if (
        gap !==
        null
      ) {
        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-gap`,

            label:
              "Écart médian",

            value:
              formatDecimal(
                gap
              ),

            context:
              finding.title,
          }
        );
      }
    }


    if (
      finding.family ===
      "quantitative_association"
    ) {
      const association =
        metricNumber(
          finding.metrics,
          "median_period_spearman"
        ) ??
        metricNumber(
          finding.metrics,
          "overall_preliminary_spearman"
        );


      if (
        association !==
        null
      ) {
        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-association`,

            label:
              "Association médiane",

            value:
              formatDecimal(
                association
              ),

            context:
              finding.title,
          }
        );
      }
    }


    if (
      finding.family ===
      "group_comparison"
    ) {
      const rows =
        (
          finding.chart_data ??
          []
        )
          .map(
            (
              datum
            ) => {
              const group =
                datumLabel(
                  datum,
                  "group"
                );

              const median =
                datumNumber(
                  datum,
                  "median"
                );


              if (
                !group ||
                median ===
                  null
              ) {
                return null;
              }


              return {
                group,
                median,
              };
            }
          )
          .filter(
            (
              item
            ): item is {
              group:
                string;

              median:
                number;
            } =>
              item !==
              null
          );


      if (
        rows.length >
        0
      ) {
        const lowest =
          [...rows]
            .sort(
              (
                left,
                right
              ) =>
                left.median -
                right.median
            )[
              0
            ];


        const measure =
          metricString(
            finding.metrics,
            "measure_column"
          );


        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-group`,

            label:
              `Médiane · ${lowest.group}`,

            value:
              formatDecimal(
                lowest.median
              ),

            context:
              measure
                ? friendlyVariableLabel(
                    measure
                  )
                : finding.title,
          }
        );
      }
    }


    if (
      output.length >=
      4
    ) {
      break;
    }
  }


  return output;
}


function ExpandableChart({
  title,
  children,
}: {
  title:
    string;

  children:
    ReactNode;
}) {
  const [
    expanded,
    setExpanded,
  ] = useState(
    false
  );


  useEffect(
    () => {
      if (
        !expanded
      ) {
        return;
      }


      const previousOverflow =
        document.body.style.overflow;


      document.body.style.overflow =
        "hidden";


      function handleKeyDown(
        event:
          KeyboardEvent
      ) {
        if (
          event.key ===
          "Escape"
        ) {
          setExpanded(
            false
          );
        }
      }


      window.addEventListener(
        "keydown",
        handleKeyDown
      );


      return () => {
        document.body.style.overflow =
          previousOverflow;

        window.removeEventListener(
          "keydown",
          handleKeyDown
        );
      };
    },
    [
      expanded,
    ]
  );


  return (
    <>
      <div
        style={{
          position:
            "relative",
        }}
      >
        <div
          style={{
            display:
              "flex",

            justifyContent:
              "flex-end",

            marginBottom:
              "8px",
          }}
        >
          <button
            type="button"
            aria-expanded={
              expanded
            }
            onClick={
              () =>
                setExpanded(
                  true
                )
            }
            style={{
              display:
                "inline-flex",

              alignItems:
                "center",

              justifyContent:
                "center",

              minHeight:
                "32px",

              padding:
                "0 10px",

              border:
                "1px solid rgba(126, 177, 255, 0.18)",

              borderRadius:
                "9px",

              color:
                "inherit",

              background:
                "rgba(126, 177, 255, 0.045)",

              font:
                "inherit",

              fontSize:
                "0.7rem",

              fontWeight:
                700,

              cursor:
                "pointer",
            }}
          >
            Agrandir ↗
          </button>
        </div>


        {
          children
        }
      </div>


      {
        expanded
          ? (
              <div
                role="dialog"
                aria-modal="true"
                aria-label={
                  `Graphique agrandi : ${title}`
                }
                onMouseDown={
                  (
                    event
                  ) => {
                    if (
                      event.target ===
                      event.currentTarget
                    ) {
                      setExpanded(
                        false
                      );
                    }
                  }
                }
                style={{
                  position:
                    "fixed",

                  inset:
                    0,

                  zIndex:
                    1000,

                  display:
                    "grid",

                  placeItems:
                    "center",

                  padding:
                    "3vh 3vw",

                  background:
                    "rgba(2, 8, 18, 0.86)",

                  backdropFilter:
                    "blur(14px)",
                }}
              >
                <section
                  style={{
                    width:
                      "min(1500px, 94vw)",

                    maxHeight:
                      "92vh",

                    display:
                      "grid",

                    gridTemplateRows:
                      "auto minmax(0, 1fr)",

                    overflow:
                      "hidden",

                    border:
                      "1px solid rgba(126, 177, 255, 0.22)",

                    borderRadius:
                      "18px",

                    background:
                      "linear-gradient(180deg, rgba(12, 25, 44, 0.99), rgba(7, 15, 28, 0.99))",

                    boxShadow:
                      "0 28px 90px rgba(0, 0, 0, 0.48)",
                  }}
                >
                  <header
                    style={{
                      display:
                        "flex",

                      alignItems:
                        "center",

                      justifyContent:
                        "space-between",

                      gap:
                        "18px",

                      padding:
                        "14px 16px",

                      borderBottom:
                        "1px solid rgba(255,255,255,0.07)",
                    }}
                  >
                    <div
                      style={{
                        minWidth:
                          0,
                      }}
                    >
                      <span
                        style={{
                          display:
                            "block",

                          marginBottom:
                            "4px",

                          fontSize:
                            "0.62rem",

                          fontWeight:
                            800,

                          letterSpacing:
                            "0.08em",

                          textTransform:
                            "uppercase",

                          opacity:
                            0.48,
                        }}
                      >
                        Visualisation agrandie
                      </span>

                      <strong
                        style={{
                          display:
                            "block",

                          overflow:
                            "hidden",

                          textOverflow:
                            "ellipsis",

                          whiteSpace:
                            "nowrap",

                          fontSize:
                            "0.94rem",
                        }}
                      >
                        {
                          title
                        }
                      </strong>
                    </div>


                    <button
                      type="button"
                      autoFocus
                      onClick={
                        () =>
                          setExpanded(
                            false
                          )
                      }
                      aria-label="Fermer le graphique agrandi"
                      style={{
                        flex:
                          "0 0 auto",

                        minHeight:
                          "34px",

                        padding:
                          "0 11px",

                        border:
                          "1px solid rgba(255,255,255,0.10)",

                        borderRadius:
                          "9px",

                        color:
                          "inherit",

                        background:
                          "rgba(255,255,255,0.035)",

                        font:
                          "inherit",

                        fontSize:
                          "0.7rem",

                        fontWeight:
                          700,

                        cursor:
                          "pointer",
                      }}
                    >
                      Fermer ×
                    </button>
                  </header>


                  <div
                    style={{
                      minHeight:
                        0,

                      overflow:
                        "auto",

                      padding:
                        "18px",

                      display:
                        "grid",

                      alignItems:
                        "center",
                    }}
                  >
                    <div
                      style={{
                        width:
                          "100%",

                        minWidth:
                          "760px",
                      }}
                    >
                      {
                        children
                      }
                    </div>
                  </div>
                </section>
              </div>
            )
          : null
      }
    </>
  );
}


function clampChartTooltipPosition(
  anchorX: number,
  anchorY: number,
  tooltipWidth: number,
  tooltipHeight: number,
  svgWidth: number,
  svgHeight: number
): {
  x: number;
  y: number;
} {
  const margin =
    12;

  const preferredX =
    anchorX +
    14;

  const fallbackX =
    anchorX -
    tooltipWidth -
    14;

  const x =
    preferredX +
      tooltipWidth <=
    svgWidth -
      margin
      ? preferredX
      : Math.max(
          margin,
          fallbackX
        );


  const y =
    Math.max(
      margin,
      Math.min(
        svgHeight -
          tooltipHeight -
          margin,
        anchorY -
          tooltipHeight /
            2
      )
    );


  return {
    x,
    y,
  };
}


function SvgChartTooltip({
  x,
  y,
  lines,
  width =
    220,
}: {
  x:
    number;

  y:
    number;

  lines:
    string[];

  width?:
    number;
}) {
  const padding =
    12;

  const lineHeight =
    18;

  const height =
    padding *
      2 +
    lines.length *
      lineHeight;


  return (
    <g
      transform={
        `translate(${x} ${y})`
      }
      pointerEvents="none"
      aria-hidden="true"
    >
      <rect
        width={
          width
        }
        height={
          height
        }
        rx="10"
        fill="#091321"
        stroke="rgba(164, 199, 255, 0.28)"
        strokeWidth="1"
      />

      {
        lines.map(
          (
            line,
            index
          ) => (
            <text
              key={
                `${index}-${line}`
              }
              x={
                padding
              }
              y={
                padding +
                13 +
                index *
                  lineHeight
              }
              fill="currentColor"
              style={{
                fontSize:
                  index ===
                    0
                    ? "12px"
                    : "11px",

                fontWeight:
                  index ===
                    0
                    ? 700
                    : 500,

                opacity:
                  index ===
                    0
                    ? 0.96
                    : 0.78,
              }}
            >
              {
                line
              }
            </text>
          )
        )
      }
    </g>
  );
}


function ScatterPlot({
  data,
  xLabel =
    "Variable X",
  yLabel =
    "Variable Y",
}: {
  data:
    ReportChartDatum[];

  xLabel?:
    string;

  yLabel?:
    string;
}) {
  const [
    hoveredIndex,
    setHoveredIndex,
  ] = useState<
    number |
    null
  >(
    null
  );


  const points =
    data
      .map(
        (
          datum
        ) => {
          const x =
            datumNumber(
              datum,
              "x"
            );

          const y =
            datumNumber(
              datum,
              "y"
            );


          if (
            x ===
              null ||
            y ===
              null
          ) {
            return null;
          }


          return {
            x,
            y,
          };
        }
      )
      .filter(
        (
          point
        ): point is
          ChartPoint =>
            point !==
            null
      );


  if (
    points.length <
    2
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Pas assez de points
        pour afficher cette relation.
      </div>
    );
  }


  const width =
    860;

  const height =
    300;


  const padding = {
    top:
      18,

    right:
      24,

    bottom:
      60,

    left:
      92,
  };


  const xMin =
    Math.min(
      ...points.map(
        (
          point
        ) =>
          point.x
      )
    );

  const xMax =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          point.x
      )
    );

  const yMin =
    Math.min(
      ...points.map(
        (
          point
        ) =>
          point.y
      )
    );

  const yMax =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          point.y
      )
    );


  const xRange =
    xMax -
      xMin ||
    1;

  const yRange =
    yMax -
      yMin ||
    1;


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    value:
      number
  ) =>
    padding.left +
    (
      (
        value -
        xMin
      ) /
      xRange
    ) *
      plotWidth;


  const projectY = (
    value:
      number
  ) =>
    padding.top +
    plotHeight -
    (
      (
        value -
        yMin
      ) /
      yRange
    ) *
      plotHeight;


  const tickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const hoveredPoint =
    hoveredIndex !==
      null
      ? points[
          hoveredIndex
        ] ??
        null
      : null;


  const tooltipWidth =
    220;

  const tooltipHeight =
    66;


  const tooltipPosition =
    hoveredPoint
      ? clampChartTooltipPosition(
          projectX(
            hoveredPoint.x
          ),
          projectY(
            hoveredPoint.y
          ),
          tooltipWidth,
          tooltipHeight,
          width,
          height
        )
      : null;


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Nuage de points : ${xLabel} et ${yLabel}`
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          tickRatios.map(
            (
              ratio
            ) => {
              const x =
                padding.left +
                ratio *
                  plotWidth;

              const y =
                padding.top +
                plotHeight -
                ratio *
                  plotHeight;

              const xValue =
                xMin +
                ratio *
                  xRange;

              const yValue =
                yMin +
                ratio *
                  yRange;


              return (
                <g
                  key={
                    ratio
                  }
                >
                  <line
                    x1={
                      padding.left
                    }
                    y1={
                      y
                    }
                    x2={
                      padding.left +
                      plotWidth
                    }
                    y2={
                      y
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <line
                    x1={
                      x
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      x
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      x
                    }
                    y={
                      padding.top +
                      plotHeight +
                      20
                    }
                    textAnchor="middle"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatAxisNumber(
                        xValue
                      )
                    }
                  </text>

                  <text
                    x={
                      padding.left -
                      10
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatAxisNumber(
                        yValue
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <line
          x1={
            padding.left
          }
          y1={
            padding.top
          }
          x2={
            padding.left
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        {
          points
            .slice(
              0,
              4000
            )
            .map(
              (
                point,
                index
              ) => (
                <g
                  key={
                    `${point.x}-${point.y}-${index}`
                  }
                  onMouseEnter={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                  onFocus={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                  onBlur={
                    () =>
                      setHoveredIndex(
                        null
                      )
                  }
                  tabIndex={
                    0
                  }
                  aria-label={
                    `${xLabel}: ${formatDecimal(
                      point.x
                    )}, ${yLabel}: ${formatDecimal(
                      point.y
                    )}`
                  }
                  style={{
                    outline:
                      "none",
                  }}
                >
                  <circle
                    cx={
                      projectX(
                        point.x
                      )
                    }
                    cy={
                      projectY(
                        point.y
                      )
                    }
                    r={
                      hoveredIndex ===
                        index
                        ? 5
                        : 3.1
                    }
                    className={
                      styles.chartPoint
                    }
                  />

                  <circle
                    cx={
                      projectX(
                        point.x
                      )
                    }
                    cy={
                      projectY(
                        point.y
                      )
                    }
                    r="10"
                    fill="transparent"
                  />
                </g>
              )
            )
        }


        {
          hoveredPoint &&
          tooltipPosition
            ? (
                <>
                  <line
                    x1={
                      projectX(
                        hoveredPoint.x
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectX(
                        hoveredPoint.x
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    stroke="currentColor"
                    strokeWidth="1"
                    opacity="0.18"
                    pointerEvents="none"
                  />

                  <SvgChartTooltip
                    x={
                      tooltipPosition.x
                    }
                    y={
                      tooltipPosition.y
                    }
                    width={
                      tooltipWidth
                    }
                    lines={
                      [
                        "Observation",
                        `${xLabel} : ${formatDecimal(
                          hoveredPoint.x
                        )}`,
                        `${yLabel} : ${formatDecimal(
                          hoveredPoint.y
                        )}`,
                      ]
                    }
                  />
                </>
              )
            : null
        }


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            8
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          {
            xLabel
          }
        </text>


        <text
          x="18"
          y={
            padding.top +
            plotHeight /
              2
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          transform={
            `rotate(-90 18 ${
              padding.top +
              plotHeight /
                2
            })`
          }
        >
          {
            yLabel
          }
        </text>
      </svg>
    </div>
  );
}



function lineBandRenderablePoints(
  data:
    ReportChartDatum[]
) {
  return (
    data
      .map(
        (
          datum
        ) => {
          const period =
            datumLabel(
              datum,
              "period"
            );

          const median =
            datumNumber(
              datum,
              "median"
            );

          const q1 =
            datumNumber(
              datum,
              "q1"
            );

          const q3 =
            datumNumber(
              datum,
              "q3"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            period ===
              null ||
            median ===
              null ||
            q1 ===
              null ||
            q3 ===
              null
          ) {
            return null;
          }


          return {
            period,
            median,
            q1,
            q3,
            count,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          period:
            string;

          median:
            number;

          q1:
            number;

          q3:
            number;

          count:
            number |
            null;
        } =>
          point !==
          null
      )
  );
}


function downsampleLineBandPoints(
  points:
    ReturnType<
      typeof lineBandRenderablePoints
    >,

  maxPoints =
    160
) {
  if (
    points.length <=
    maxPoints
  ) {
    return points;
  }


  const step =
    (
      points.length -
      1
    ) /
    (
      maxPoints -
      1
    );


  return (
    Array.from(
      {
        length:
          maxPoints,
      },
      (
        _,
        index
      ) =>
        points[
          Math.round(
            index *
            step
          )
        ]
    )
  );
}


function LineBandChart({
  data,
  yLabel =
    "Valeur",
}: {
  data:
    ReportChartDatum[];

  yLabel?:
    string;
}) {
  const [
    hoveredIndex,
    setHoveredIndex,
  ] = useState<
    number |
    null
  >(
    null
  );


  const allPoints =
    lineBandRenderablePoints(
      data
    );


  if (
    allPoints.length <
    2
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Pas assez de périodes
        pour afficher l’évolution.
      </div>
    );
  }


  const points =
    downsampleLineBandPoints(
      allPoints
    );


  const width =
    860;

  const height =
    420;


  const padding = {
    top:
      28,

    right:
      28,

    bottom:
      58,

    left:
      94,
  };


  const values =
    points.flatMap(
      (
        point
      ) => [
        point.q1,
        point.median,
        point.q3,
      ]
    );


  const rawMin =
    Math.min(
      ...values
    );

  const rawMax =
    Math.max(
      ...values
    );


  const rawRange =
    rawMax -
      rawMin ||
    1;


  const yPadding =
    rawRange *
    0.06;


  const yMin =
    rawMin -
    yPadding;

  const yMax =
    rawMax +
    yPadding;


  const yRange =
    yMax -
      yMin ||
    1;


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    index:
      number
  ) =>
    padding.left +
    (
      points.length ===
      1
        ? 0
        : (
            index /
            (
              points.length -
              1
            )
          ) *
          plotWidth
    );


  const projectY = (
    value:
      number
  ) =>
    padding.top +
    plotHeight -
    (
      (
        value -
        yMin
      ) /
      yRange
    ) *
      plotHeight;


  const upper =
    points.map(
      (
        point,
        index
      ) =>
        `${projectX(
          index
        )},${projectY(
          point.q3
        )}`
    );


  const lower =
    [...points]
      .reverse()
      .map(
        (
          point,
          reverseIndex
        ) => {
          const index =
            points.length -
            1 -
            reverseIndex;


          return (
            `${projectX(
              index
            )},${projectY(
              point.q1
            )}`
          );
        }
      );


  const band =
    [
      ...upper,
      ...lower,
    ].join(
      " "
    );


  const medianLine =
    points
      .map(
        (
          point,
          index
        ) =>
          `${projectX(
            index
          )},${projectY(
            point.median
          )}`
      )
      .join(
        " "
      );


  const yTickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const middleIndex =
    Math.floor(
      (
        points.length -
        1
      ) /
      2
    );


  const xTicks = [
    {
      index:
        0,

      anchor:
        "start" as const,
    },
    {
      index:
        middleIndex,

      anchor:
        "middle" as const,
    },
    {
      index:
        points.length -
        1,

      anchor:
        "end" as const,
    },
  ];


  const hoveredPoint =
    hoveredIndex !==
      null
      ? points[
          hoveredIndex
        ] ??
        null
      : null;


  const tooltipWidth =
    230;

  const tooltipLineCount =
    hoveredPoint?.count !==
      null &&
    hoveredPoint?.count !==
      undefined
      ? 5
      : 4;

  const tooltipHeight =
    24 +
    tooltipLineCount *
      18;


  const tooltipPosition =
    hoveredPoint &&
    hoveredIndex !==
      null
      ? clampChartTooltipPosition(
          projectX(
            hoveredIndex
          ),
          projectY(
            hoveredPoint.median
          ),
          tooltipWidth,
          tooltipHeight,
          width,
          height
        )
      : null;


  const hoverBandWidth =
    plotWidth /
    Math.max(
      points.length -
        1,
      1
    );


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Évolution temporelle de ${yLabel} sur ${
            formatNumber(
              allPoints.length
            )
          } période(s)`
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          yTickRatios.map(
            (
              ratio
            ) => {
              const y =
                padding.top +
                plotHeight -
                ratio *
                  plotHeight;

              const value =
                yMin +
                ratio *
                  yRange;


              return (
                <g
                  key={
                    `y-${ratio}`
                  }
                >
                  <line
                    x1={
                      padding.left
                    }
                    y1={
                      y
                    }
                    x2={
                      padding.left +
                      plotWidth
                    }
                    y2={
                      y
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      padding.left -
                      12
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                    style={{
                      fontSize:
                        "12px",

                      opacity:
                        0.82,
                    }}
                  >
                    {
                      formatAxisNumber(
                        value
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <line
          x1={
            padding.left
          }
          y1={
            padding.top
          }
          x2={
            padding.left
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <polygon
          points={
            band
          }
          fill="currentColor"
          opacity="0.08"
        />


        <polyline
          points={
            medianLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
        />


        {
          points.map(
            (
              point,
              index
            ) => (
              <g
                key={
                  `${
                    point.period
                  }-${
                    index
                  }`
                }
              >
                <rect
                  x={
                    Math.max(
                      padding.left,
                      projectX(
                        index
                      ) -
                      hoverBandWidth /
                        2
                    )
                  }
                  y={
                    padding.top
                  }
                  width={
                    Math.min(
                      hoverBandWidth,
                      padding.left +
                        plotWidth -
                        Math.max(
                          padding.left,
                          projectX(
                            index
                          ) -
                          hoverBandWidth /
                            2
                        )
                    )
                  }
                  height={
                    plotHeight
                  }
                  fill="transparent"
                  onMouseEnter={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                />

                <circle
                  cx={
                    projectX(
                      index
                    )
                  }
                  cy={
                    projectY(
                      point.median
                    )
                  }
                  r={
                    hoveredIndex ===
                      index
                      ? 5
                      : 3.4
                  }
                  className={
                    styles.chartPoint
                  }
                  tabIndex={
                    0
                  }
                  onFocus={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                  onBlur={
                    () =>
                      setHoveredIndex(
                        null
                      )
                  }
                  aria-label={
                    `${
                      formatTemporalDisplayValue(
                        point.period
                      )
                    }, médiane ${formatDecimal(
                      point.median
                    )}, Q1 ${formatDecimal(
                      point.q1
                    )}, Q3 ${formatDecimal(
                      point.q3
                    )}`
                  }
                />
              </g>
            )
          )
        }


        {
          hoveredPoint &&
          tooltipPosition &&
          hoveredIndex !==
            null
            ? (
                <>
                  <line
                    x1={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    stroke="currentColor"
                    strokeWidth="1"
                    opacity="0.24"
                    pointerEvents="none"
                  />

                  <circle
                    cx={
                      projectX(
                        hoveredIndex
                      )
                    }
                    cy={
                      projectY(
                        hoveredPoint.median
                      )
                    }
                    r="7"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    opacity="0.7"
                    pointerEvents="none"
                  />

                  <SvgChartTooltip
                    x={
                      tooltipPosition.x
                    }
                    y={
                      tooltipPosition.y
                    }
                    width={
                      tooltipWidth
                    }
                    lines={
                      [
                        formatTemporalDisplayValue(
                          hoveredPoint.period
                        ),
                        `Médiane : ${formatDecimal(
                          hoveredPoint.median
                        )}`,
                        `Q1 : ${formatDecimal(
                          hoveredPoint.q1
                        )}`,
                        `Q3 : ${formatDecimal(
                          hoveredPoint.q3
                        )}`,
                        ...(
                          hoveredPoint.count !==
                            null
                            ? [
                                `Observations : ${formatNumber(
                                  hoveredPoint.count
                                )}`,
                              ]
                            : []
                        ),
                      ]
                    }
                  />
                </>
              )
            : null
        }


        {
          xTicks.map(
            (
              tick
            ) => (
              <text
                key={
                  `${
                    tick.index
                  }-${
                    tick.anchor
                  }`
                }
                x={
                  projectX(
                    tick.index
                  )
                }
                y={
                  padding.top +
                  plotHeight +
                  24
                }
                textAnchor={
                  tick.anchor
                }
                className={
                  styles.chartTick
                }
                style={{
                  fontSize:
                    "12px",

                  opacity:
                    0.82,
                }}
              >
                {
                  formatTemporalDisplayValue(
                    points[
                      tick.index
                    ].period
                  )
                }
              </text>
            )
          )
        }


        <text
          x="24"
          y={
            padding.top +
            plotHeight /
              2
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          transform={
            `rotate(-90 24 ${
              padding.top +
              plotHeight /
                2
            })`
          }
          style={{
            fontSize:
              "12px",

            opacity:
              0.76,
          }}
        >
          {
            yLabel
          }
        </text>
      </svg>


      {
        allPoints.length >
        points.length
          ? (
              <p
                style={{
                  margin:
                    "8px 0 0",

                  fontSize:
                    "0.69rem",

                  opacity:
                    0.52,
                }}
              >
                {
                  formatNumber(
                    allPoints.length
                  )
                }
                {" périodes · affichage simplifié à "}
                {
                  formatNumber(
                    points.length
                  )
                }
                {" points pour la lisibilité."}
              </p>
            )
          : null
      }
    </div>
  );
}


function GroupedSummaryChart({
  data,
}: {
  data:
    ReportChartDatum[];
}) {
  const points =
    data
      .map(
        (
          datum
        ) => {
          const group =
            datumLabel(
              datum,
              "group"
            );

          const median =
            datumNumber(
              datum,
              "median"
            );


          if (
            !group ||
            median ===
              null
          ) {
            return null;
          }


          return {
            group,
            median,
          };
        }
      )
      .filter(
        (
          point
        ): point is
          GroupSummaryPoint =>
            point !==
            null
      );


  if (
    points.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucun groupe exploitable.
      </div>
    );
  }


  const maxValue =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          point.median
      ),
      1
    );


  return (
    <div
      className={
        styles.technicalReasons
      }
    >
      {
        points.map(
          (
            point
          ) => (
            <div
              key={
                point.group
              }
              className={
                styles.summaryItem
              }
            >
              <span>
                {
                  point.group
                }
              </span>

              <strong>
                {
                  formatDecimal(
                    point.median
                  )
                }
              </strong>

              <div
                style={{
                  width:
                    "100%",

                  height:
                    "8px",

                  borderRadius:
                    "999px",

                  overflow:
                    "hidden",

                  background:
                    "rgba(255,255,255,0.08)",
                }}
              >
                <div
                  style={{
                    width:
                      `${Math.max(
                        2,
                        (
                          point.median /
                          maxValue
                        ) *
                          100
                      )}%`,

                    height:
                      "100%",

                    borderRadius:
                      "999px",

                    background:
                      "currentColor",

                    opacity:
                      0.72,
                  }}
                />
              </div>
            </div>
          )
        )
      }
    </div>
  );
}


function GapSummaryChart({
  finding,
}: {
  finding:
    ReportFinding;
}) {
  const minimum =
    metricNumber(
      finding.metrics,
      "minimum_gap"
    );

  const median =
    metricNumber(
      finding.metrics,
      "median_gap"
    );

  const mean =
    metricNumber(
      finding.metrics,
      "mean_gap"
    );

  const maximum =
    metricNumber(
      finding.metrics,
      "maximum_gap"
    );


  if (
    minimum ===
      null ||
    median ===
      null ||
    maximum ===
      null
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Les statistiques de
        distribution sont disponibles
        dans le résumé.
      </div>
    );
  }


  const range =
    maximum -
      minimum ||
    1;


  const medianPosition =
    (
      (
        median -
        minimum
      ) /
      range
    ) *
    100;


  const meanPosition =
    mean !==
      null
      ? (
          (
            mean -
            minimum
          ) /
          range
        ) *
          100
      : null;


  return (
    <div
      className={
        styles.summaryPanel
      }
    >
      <div
        className={
          styles.summaryItem
        }
      >
        <span>
          Étendue observée
        </span>

        <strong>
          {
            formatDecimal(
              minimum
            )
          }
          {" — "}
          {
            formatDecimal(
              maximum
            )
          }
        </strong>


        <div
          style={{
            position:
              "relative",

            height:
              "18px",

            marginTop:
              "18px",

            borderRadius:
              "999px",

            background:
              "rgba(255,255,255,0.08)",
          }}
        >
          <span
            title={
              `Médiane : ${formatDecimal(
                median
              )}`
            }
            style={{
              position:
                "absolute",

              left:
                `${medianPosition}%`,

              top:
                "-5px",

              width:
                "4px",

              height:
                "28px",

              borderRadius:
                "999px",

              background:
                "currentColor",
            }}
          />


          {
            mean !==
              null &&
            meanPosition !==
              null
              ? (
                  <span
                    title={
                      `Moyenne : ${formatDecimal(
                        mean
                      )}`
                    }
                    style={{
                      position:
                        "absolute",

                      left:
                        `${meanPosition}%`,

                      top:
                        "1px",

                      width:
                        "10px",

                      height:
                        "10px",

                      borderRadius:
                        "50%",

                      border:
                        "2px solid currentColor",
                    }}
                  />
                )
              : null
          }
        </div>


        <p>
          Médiane :
          {" "}
          {
            formatDecimal(
              median
            )
          }

          {
            mean !==
              null
              ? (
                  <>
                    {" · "}
                    Moyenne :
                    {" "}
                    {
                      formatDecimal(
                        mean
                      )
                    }
                  </>
                )
              : null
          }
        </p>
      </div>
    </div>
  );
}


function SimpleLineChart({
  data,
  xLabel =
    "Période",
  yLabel =
    "Valeur",
}: {
  data:
    ReportChartDatum[];

  xLabel?:
    string;

  yLabel?:
    string;
}) {
  const [
    hoveredIndex,
    setHoveredIndex,
  ] = useState<
    number |
    null
  >(
    null
  );


  const points =
    data
      .map(
        (
          datum,
          index
        ) => {
          const label =
            datumLabel(
              datum,
              "period"
            ) ??
            String(
              index + 1
            );

          const value =
            datumNumber(
              datum,
              "value"
            );


          if (
            value ===
            null
          ) {
            return null;
          }


          return {
            label,
            value,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          label:
            string;

          value:
            number;
        } =>
          point !==
          null
      );


  if (
    points.length <
    2
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Pas assez de périodes
        pour afficher l’évolution.
      </div>
    );
  }


  const width =
    860;

  const height =
    330;


  const padding = {
    top:
      20,

    right:
      28,

    bottom:
      58,

    left:
      82,
  };


  const values =
    points.map(
      (
        point
      ) =>
        point.value
    );


  const yMin =
    Math.min(
      ...values
    );

  const yMax =
    Math.max(
      ...values
    );

  const yRange =
    yMax -
      yMin ||
    1;


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    index:
      number
  ) =>
    padding.left +
    (
      points.length ===
      1
        ? 0
        : (
            index /
            (
              points.length -
              1
            )
          ) *
          plotWidth
    );


  const projectY = (
    value:
      number
  ) =>
    padding.top +
    plotHeight -
    (
      (
        value -
        yMin
      ) /
      yRange
    ) *
      plotHeight;


  const line =
    points
      .map(
        (
          point,
          index
        ) =>
          `${projectX(
            index
          )},${projectY(
            point.value
          )}`
      )
      .join(
        " "
      );


  const tickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const xTickIndexes =
    Array.from(
      new Set(
        [
          0,
          Math.floor(
            (
              points.length -
              1
            ) /
            2
          ),
          points.length -
          1,
        ]
      )
    );


  const hoveredPoint =
    hoveredIndex !==
      null
      ? points[
          hoveredIndex
        ] ??
        null
      : null;


  const tooltipWidth =
    220;

  const tooltipHeight =
    66;


  const tooltipPosition =
    hoveredPoint &&
    hoveredIndex !==
      null
      ? clampChartTooltipPosition(
          projectX(
            hoveredIndex
          ),
          projectY(
            hoveredPoint.value
          ),
          tooltipWidth,
          tooltipHeight,
          width,
          height
        )
      : null;


  const hoverBandWidth =
    plotWidth /
    Math.max(
      points.length -
        1,
      1
    );


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Évolution de ${yLabel} selon ${xLabel}`
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          tickRatios.map(
            (
              ratio
            ) => {
              const y =
                padding.top +
                plotHeight -
                ratio *
                  plotHeight;

              const value =
                yMin +
                ratio *
                  yRange;


              return (
                <g
                  key={
                    ratio
                  }
                >
                  <line
                    x1={
                      padding.left
                    }
                    y1={
                      y
                    }
                    x2={
                      padding.left +
                      plotWidth
                    }
                    y2={
                      y
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      padding.left -
                      10
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatAxisNumber(
                        value
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <line
          x1={
            padding.left
          }
          y1={
            padding.top
          }
          x2={
            padding.left
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <polyline
          points={
            line
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
        />


        {
          points.map(
            (
              point,
              index
            ) => (
              <g
                key={
                  `${point.label}-${index}`
                }
              >
                <rect
                  x={
                    Math.max(
                      padding.left,
                      projectX(
                        index
                      ) -
                      hoverBandWidth /
                        2
                    )
                  }
                  y={
                    padding.top
                  }
                  width={
                    Math.min(
                      hoverBandWidth,
                      padding.left +
                        plotWidth -
                        Math.max(
                          padding.left,
                          projectX(
                            index
                          ) -
                          hoverBandWidth /
                            2
                        )
                    )
                  }
                  height={
                    plotHeight
                  }
                  fill="transparent"
                  onMouseEnter={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                />

                <circle
                  cx={
                    projectX(
                      index
                    )
                  }
                  cy={
                    projectY(
                      point.value
                    )
                  }
                  r={
                    hoveredIndex ===
                      index
                      ? 5
                      : 3.4
                  }
                  className={
                    styles.chartPoint
                  }
                  tabIndex={
                    0
                  }
                  onFocus={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                  onBlur={
                    () =>
                      setHoveredIndex(
                        null
                      )
                  }
                  aria-label={
                    `${formatTemporalDisplayValue(
                      point.label
                    )}, ${yLabel}: ${formatDecimal(
                      point.value
                    )}`
                  }
                />
              </g>
            )
          )
        }


        {
          hoveredPoint &&
          tooltipPosition &&
          hoveredIndex !==
            null
            ? (
                <>
                  <line
                    x1={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    stroke="currentColor"
                    strokeWidth="1"
                    opacity="0.24"
                    pointerEvents="none"
                  />

                  <SvgChartTooltip
                    x={
                      tooltipPosition.x
                    }
                    y={
                      tooltipPosition.y
                    }
                    width={
                      tooltipWidth
                    }
                    lines={
                      [
                        formatTemporalDisplayValue(
                          hoveredPoint.label
                        ),
                        `${yLabel} : ${formatDecimal(
                          hoveredPoint.value
                        )}`,
                      ]
                    }
                  />
                </>
              )
            : null
        }


        {
          xTickIndexes.map(
            (
              index
            ) => (
              <text
                key={
                  index
                }
                x={
                  projectX(
                    index
                  )
                }
                y={
                  padding.top +
                  plotHeight +
                  22
                }
                textAnchor={
                  index ===
                    0
                    ? "start"
                    : (
                        index ===
                        points.length -
                          1
                          ? "end"
                          : "middle"
                      )
                }
                className={
                  styles.chartTick
                }
              >
                {
                  formatTemporalDisplayValue(
                    points[
                      index
                    ].label
                  )
                }
              </text>
            )
          )
        }


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            8
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          {
            xLabel
          }
        </text>


        <text
          x="20"
          y={
            padding.top +
            plotHeight /
              2
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          transform={
            `rotate(-90 20 ${
              padding.top +
              plotHeight /
                2
            })`
          }
        >
          {
            yLabel
          }
        </text>
      </svg>
    </div>
  );
}



function SimpleBarChart({
  finding,
}: {
  finding:
    ReportFinding;
}) {
  const groupColumn =
    metricString(
      finding.metrics,
      "group_column"
    );

  const measureColumn =
    metricString(
      finding.metrics,
      "measure_column"
    );


  const points =
    (
      finding.chart_data ??
      []
    )
      .map(
        (
          datum,
          index
        ) => {
          const label =
            datumLabel(
              datum,
              "group"
            ) ??
            (
              groupColumn
                ? datumLabel(
                    datum,
                    groupColumn
                  )
                : null
            ) ??
            datumLabel(
              datum,
              "category"
            ) ??
            datumLabel(
              datum,
              "label"
            ) ??
            `Groupe ${index + 1}`;


          const preferredKeys = [
            "value",
            measureColumn,
            "total",
            "amount",
            "sum",
            "count",
            "y",
          ].filter(
            (
              key
            ): key is string =>
              Boolean(
                key
              )
          );


          let value:
            number |
            null =
              null;


          for (
            const key
            of preferredKeys
          ) {
            value =
              datumNumber(
                datum,
                key
              );


            if (
              value !==
              null
            ) {
              break;
            }
          }


          if (
            value ===
            null
          ) {
            const excludedKeys =
              new Set(
                [
                  "group",
                  groupColumn,
                  "category",
                  "label",
                  "share",
                  "percentage",
                  "percent",
                  "n",
                ].filter(
                  (
                    key
                  ): key is string =>
                    Boolean(
                      key
                    )
                )
              );


            for (
              const [
                key,
                rawValue,
              ]
              of Object.entries(
                datum
              )
            ) {
              if (
                excludedKeys.has(
                  key
                )
              ) {
                continue;
              }


              if (
                typeof rawValue ===
                  "number" &&
                Number.isFinite(
                  rawValue
                )
              ) {
                value =
                  rawValue;

                break;
              }
            }
          }


          if (
            value ===
            null
          ) {
            return null;
          }


          return {
            label,
            value,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          label:
            string;

          value:
            number;
        } =>
          point !==
          null
      );


  if (
    points.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucune catégorie exploitable
        pour ce graphique.
      </div>
    );
  }


  const maxValue =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          Math.abs(
            point.value
          )
      ),
      1
    );


  return (
    <div
      className={
        styles.technicalReasons
      }
    >
      {
        points.map(
          (
            point,
            index
          ) => (
            <div
              key={
                `${point.label}-${index}`
              }
              className={
                styles.summaryItem
              }
              title={
                `${
                  groupColumn
                    ? `${friendlyVariableLabel(
                        groupColumn
                      )}: `
                    : ""
                }${point.label} · ${
                  measureColumn
                    ? `${friendlyVariableLabel(
                        measureColumn
                      )}: `
                    : ""
                }${formatDecimal(
                  point.value
                )}`
              }
            >
              <span>
                {
                  groupColumn
                    ? `${friendlyVariableLabel(
                        groupColumn
                      )} · ${point.label}`
                    : point.label
                }
              </span>

              <strong>
                {
                  formatChartNumber(
                    point.value
                  )
                }
              </strong>

              <div
                style={{
                  width:
                    "100%",

                  height:
                    "9px",

                  borderRadius:
                    "999px",

                  overflow:
                    "hidden",

                  background:
                    "rgba(255,255,255,0.08)",
                }}
                aria-hidden="true"
              >
                <div
                  style={{
                    width:
                      `${Math.max(
                        2,
                        (
                          Math.abs(
                            point.value
                          ) /
                          maxValue
                        ) *
                          100
                      )}%`,

                    height:
                      "100%",

                    borderRadius:
                      "999px",

                    background:
                      "currentColor",

                    opacity:
                      0.72,
                  }}
                />
              </div>
            </div>
          )
        )
      }
    </div>
  );
}




function RequestedTimeSeriesChart({
  data,
  valueLabel,
  showMovingAverage =
    false,
}: {
  data:
    ReportChartDatum[];

  valueLabel:
    string;

  showMovingAverage?:
    boolean;
}) {
  const [
    hoveredIndex,
    setHoveredIndex,
  ] = useState<
    number |
    null
  >(
    null
  );


  const points =
    data
      .map(
        (
          datum,
          index
        ) => {
          const label =
            datumLabel(
              datum,
              "period"
            ) ??
            String(
              index + 1
            );

          const value =
            datumNumber(
              datum,
              "value"
            );

          const movingAverage =
            datumNumber(
              datum,
              "moving_average"
            );


          if (
            value ===
            null
          ) {
            return null;
          }


          return {
            label,
            value,
            movingAverage,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          label:
            string;

          value:
            number;

          movingAverage:
            number |
            null;
        } =>
          point !==
          null
      );


  if (
    points.length <
    2
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Pas assez de périodes
        pour afficher l’évolution.
      </div>
    );
  }


  const values = [
    ...points.map(
      (
        point
      ) =>
        point.value
    ),

    ...points
      .map(
        (
          point
        ) =>
          point.movingAverage
      )
      .filter(
        (
          value
        ): value is number =>
          value !==
          null
      ),
  ];


  const width =
    860;

  const height =
    330;


  const padding = {
    top:
      28,

    right:
      30,

    bottom:
      58,

    left:
      82,
  };


  const yMin =
    Math.min(
      ...values
    );

  const yMax =
    Math.max(
      ...values
    );

  const yRange =
    yMax -
      yMin ||
    1;


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    index:
      number
  ) =>
    padding.left +
    (
      index /
      Math.max(
        points.length -
          1,
        1
      )
    ) *
      plotWidth;


  const projectY = (
    value:
      number
  ) =>
    padding.top +
    plotHeight -
    (
      (
        value -
        yMin
      ) /
      yRange
    ) *
      plotHeight;


  const primaryLine =
    points
      .map(
        (
          point,
          index
        ) =>
          `${projectX(
            index
          )},${projectY(
            point.value
          )}`
      )
      .join(
        " "
      );


  const movingLine =
    points
      .map(
        (
          point,
          index
        ) => {
          if (
            point.movingAverage ===
            null
          ) {
            return null;
          }


          return `${projectX(
            index
          )},${projectY(
            point.movingAverage
          )}`;
        }
      )
      .filter(
        (
          point
        ): point is string =>
          point !==
          null
      )
      .join(
        " "
      );


  const tickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const xTickIndexes =
    Array.from(
      new Set(
        [
          0,
          Math.floor(
            (
              points.length -
              1
            ) /
            2
          ),
          points.length -
            1,
        ]
      )
    );


  const hoveredPoint =
    hoveredIndex !==
      null
      ? points[
          hoveredIndex
        ] ??
        null
      : null;


  const tooltipWidth =
    230;

  const tooltipLineCount =
    hoveredPoint?.movingAverage !==
      null &&
    hoveredPoint?.movingAverage !==
      undefined
      ? 3
      : 2;

  const tooltipHeight =
    24 +
    tooltipLineCount *
      18;


  const tooltipPosition =
    hoveredPoint &&
    hoveredIndex !==
      null
      ? clampChartTooltipPosition(
          projectX(
            hoveredIndex
          ),
          projectY(
            hoveredPoint.value
          ),
          tooltipWidth,
          tooltipHeight,
          width,
          height
        )
      : null;


  const hoverBandWidth =
    plotWidth /
    Math.max(
      points.length -
        1,
      1
    );


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Évolution de ${valueLabel}`
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          tickRatios.map(
            (
              ratio
            ) => {
              const y =
                padding.top +
                plotHeight -
                ratio *
                  plotHeight;

              const value =
                yMin +
                ratio *
                  yRange;


              return (
                <g
                  key={
                    ratio
                  }
                >
                  <line
                    x1={
                      padding.left
                    }
                    y1={
                      y
                    }
                    x2={
                      padding.left +
                      plotWidth
                    }
                    y2={
                      y
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      padding.left -
                      10
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatAxisNumber(
                        value
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <line
          x1={
            padding.left
          }
          y1={
            padding.top
          }
          x2={
            padding.left
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <polyline
          points={
            primaryLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />


        {
          showMovingAverage &&
          movingLine
            ? (
                <polyline
                  points={
                    movingLine
                  }
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeDasharray="9 8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.55"
                />
              )
            : null
        }


        {
          points.map(
            (
              point,
              index
            ) => (
              <g
                key={
                  `${point.label}-${index}`
                }
              >
                <rect
                  x={
                    Math.max(
                      padding.left,
                      projectX(
                        index
                      ) -
                      hoverBandWidth /
                        2
                    )
                  }
                  y={
                    padding.top
                  }
                  width={
                    Math.min(
                      hoverBandWidth,
                      padding.left +
                        plotWidth -
                        Math.max(
                          padding.left,
                          projectX(
                            index
                          ) -
                          hoverBandWidth /
                            2
                        )
                    )
                  }
                  height={
                    plotHeight
                  }
                  fill="transparent"
                  onMouseEnter={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                />

                <circle
                  cx={
                    projectX(
                      index
                    )
                  }
                  cy={
                    projectY(
                      point.value
                    )
                  }
                  r={
                    hoveredIndex ===
                      index
                      ? 5
                      : 3.2
                  }
                  className={
                    styles.chartPoint
                  }
                  tabIndex={
                    0
                  }
                  onFocus={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                  onBlur={
                    () =>
                      setHoveredIndex(
                        null
                      )
                  }
                  aria-label={
                    `${formatTemporalDisplayValue(
                      point.label
                    )}, ${valueLabel}: ${formatDecimal(
                      point.value
                    )}`
                  }
                />
              </g>
            )
          )
        }


        {
          hoveredPoint &&
          tooltipPosition &&
          hoveredIndex !==
            null
            ? (
                <>
                  <line
                    x1={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    stroke="currentColor"
                    strokeWidth="1"
                    opacity="0.24"
                    pointerEvents="none"
                  />

                  <SvgChartTooltip
                    x={
                      tooltipPosition.x
                    }
                    y={
                      tooltipPosition.y
                    }
                    width={
                      tooltipWidth
                    }
                    lines={
                      [
                        formatTemporalDisplayValue(
                          hoveredPoint.label
                        ),
                        `${valueLabel} : ${formatDecimal(
                          hoveredPoint.value
                        )}`,
                        ...(
                          hoveredPoint.movingAverage !==
                            null
                            ? [
                                `Moyenne mobile : ${formatDecimal(
                                  hoveredPoint.movingAverage
                                )}`,
                              ]
                            : []
                        ),
                      ]
                    }
                  />
                </>
              )
            : null
        }


        {
          xTickIndexes.map(
            (
              index
            ) => (
              <text
                key={
                  index
                }
                x={
                  projectX(
                    index
                  )
                }
                y={
                  padding.top +
                  plotHeight +
                  22
                }
                textAnchor={
                  index ===
                    0
                    ? "start"
                    : (
                        index ===
                        points.length -
                          1
                          ? "end"
                          : "middle"
                      )
                }
                className={
                  styles.chartTick
                }
              >
                {
                  formatTemporalDisplayValue(
                    points[
                      index
                    ].label
                  )
                }
              </text>
            )
          )
        }


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            8
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          Période
        </text>


        {
          showMovingAverage
            ? (
                <g
                  transform={
                    `translate(${padding.left + 8} 12)`
                  }
                >
                  <line
                    x1="0"
                    y1="0"
                    x2="28"
                    y2="0"
                    stroke="currentColor"
                    strokeWidth="3"
                  />

                  <text
                    x="36"
                    y="4"
                    className={
                      styles.chartTick
                    }
                  >
                    Valeur
                  </text>

                  <line
                    x1="105"
                    y1="0"
                    x2="133"
                    y2="0"
                    stroke="currentColor"
                    strokeWidth="3"
                    strokeDasharray="7 6"
                    opacity="0.55"
                  />

                  <text
                    x="141"
                    y="4"
                    className={
                      styles.chartTick
                    }
                  >
                    Moyenne mobile
                  </text>
                </g>
              )
            : null
        }
      </svg>
    </div>
  );
}



function RequestedBarChart({
  data,
  categoryLabel,
  valueLabel,
}: {
  data:
    ReportChartDatum[];

  categoryLabel:
    string;

  valueLabel:
    string;
}) {
  const points =
    data
      .map(
        (
          datum,
          index
        ) => {
          const label =
            datumLabel(
              datum,
              "category"
            ) ??
            datumLabel(
              datum,
              "group"
            ) ??
            datumLabel(
              datum,
              "label"
            ) ??
            `Catégorie ${index + 1}`;


          const value =
            datumNumber(
              datum,
              "value"
            );


          if (
            value ===
            null
          ) {
            return null;
          }


          return {
            label,
            value,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          label:
            string;

          value:
            number;
        } =>
          point !==
          null
      );


  if (
    points.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucune catégorie exploitable.
      </div>
    );
  }


  const maxValue =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          Math.abs(
            point.value
          )
      ),
      1
    );


  return (
    <div
      className={
        styles.technicalReasons
      }
    >
      {
        points.map(
          (
            point,
            index
          ) => (
            <div
              key={
                `${point.label}-${index}`
              }
              className={
                styles.summaryItem
              }
              title={
                `${categoryLabel}: ${point.label} · ${valueLabel}: ${formatDecimal(
                  point.value
                )}`
              }
            >
              <span>
                {
                  `${categoryLabel} · ${point.label}`
                }
              </span>

              <strong>
                {
                  formatChartNumber(
                    point.value
                  )
                }
              </strong>

              <div
                style={{
                  width:
                    "100%",

                  height:
                    "9px",

                  borderRadius:
                    "999px",

                  overflow:
                    "hidden",

                  background:
                    "rgba(255,255,255,0.08)",
                }}
                aria-hidden="true"
              >
                <div
                  style={{
                    width:
                      `${Math.max(
                        2,
                        (
                          Math.abs(
                            point.value
                          ) /
                          maxValue
                        ) *
                          100
                      )}%`,

                    height:
                      "100%",

                    borderRadius:
                      "999px",

                    background:
                      "currentColor",

                    opacity:
                      0.72,
                  }}
                />
              </div>
            </div>
          )
        )
      }
    </div>
  );
}



function RequestedLorenzChart({
  data,
}: {
  data:
    ReportChartDatum[];
}) {
  const points =
    data
      .map(
        (
          datum
        ) => {
          const populationShare =
            datumNumber(
              datum,
              "population_share"
            );

          const revenueShare =
            datumNumber(
              datum,
              "revenue_share"
            );

          const equalityShare =
            datumNumber(
              datum,
              "equality_share"
            );


          if (
            populationShare ===
              null ||
            revenueShare ===
              null
          ) {
            return null;
          }


          return {
            populationShare,
            revenueShare,
            equalityShare:
              equalityShare ??
              populationShare,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          populationShare:
            number;

          revenueShare:
            number;

          equalityShare:
            number;
        } =>
          point !==
          null
      );


  if (
    points.length <
    2
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Pas assez de points
        pour afficher la courbe de Lorenz.
      </div>
    );
  }


  const width =
    860;

  const height =
    360;


  const padding = {
    top:
      26,

    right:
      30,

    bottom:
      62,

    left:
      78,
  };


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    share:
      number
  ) =>
    padding.left +
    Math.max(
      0,
      Math.min(
        1,
        share
      )
    ) *
      plotWidth;


  const projectY = (
    share:
      number
  ) =>
    padding.top +
    plotHeight -
    Math.max(
      0,
      Math.min(
        1,
        share
      )
    ) *
      plotHeight;


  const lorenzLine =
    points
      .map(
        (
          point
        ) =>
          `${projectX(
            point.populationShare
          )},${projectY(
            point.revenueShare
          )}`
      )
      .join(
        " "
      );


  const equalityLine =
    points
      .map(
        (
          point
        ) =>
          `${projectX(
            point.populationShare
          )},${projectY(
            point.equalityShare
          )}`
      )
      .join(
        " "
      );


  const ticks = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          "Courbe de Lorenz du chiffre d’affaires client"
        }
      >
        {
          ticks.map(
            (
              share
            ) => {
              const x =
                projectX(
                  share
                );

              const y =
                projectY(
                  share
                );


              return (
                <g
                  key={
                    share
                  }
                >
                  <line
                    x1={
                      padding.left
                    }
                    y1={
                      y
                    }
                    x2={
                      padding.left +
                      plotWidth
                    }
                    y2={
                      y
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <line
                    x1={
                      x
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      x
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      x
                    }
                    y={
                      padding.top +
                      plotHeight +
                      22
                    }
                    textAnchor="middle"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatPercent(
                        share
                      )
                    }
                  </text>

                  <text
                    x={
                      padding.left -
                      10
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatPercent(
                        share
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <line
          x1={
            padding.left
          }
          y1={
            padding.top
          }
          x2={
            padding.left
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <polyline
          points={
            equalityLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeDasharray="8 8"
          opacity="0.42"
        />


        <polyline
          points={
            lorenzLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />


        {
          points
            .filter(
              (
                _,
                index
              ) =>
                (
                  index ===
                    0 ||
                  index ===
                    points.length -
                      1 ||
                  index %
                    Math.max(
                      1,
                      Math.floor(
                        points.length /
                        24
                      )
                    )
                    ===
                    0
                )
            )
            .map(
              (
                point,
                index
              ) => (
                <circle
                  key={
                    `${point.populationShare}-${point.revenueShare}-${index}`
                  }
                  cx={
                    projectX(
                      point.populationShare
                    )
                  }
                  cy={
                    projectY(
                      point.revenueShare
                    )
                  }
                  r="3"
                  className={
                    styles.chartPoint
                  }
                >
                  <title>
                    {
                      `Clients cumulés : ${formatPercent(
                        point.populationShare
                      )} · CA cumulé : ${formatPercent(
                        point.revenueShare
                      )}`
                    }
                  </title>
                </circle>
              )
            )
        }


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            10
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          Part cumulée des clients
        </text>


        <text
          x="18"
          y={
            padding.top +
            plotHeight /
              2
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          transform={
            `rotate(-90 18 ${
              padding.top +
              plotHeight /
                2
            })`
          }
        >
          Part cumulée du chiffre d’affaires
        </text>


        <g
          transform={
            `translate(${padding.left + 10} 13)`
          }
        >
          <line
            x1="0"
            y1="0"
            x2="28"
            y2="0"
            stroke="currentColor"
            strokeWidth="3"
          />

          <text
            x="36"
            y="4"
            className={
              styles.chartTick
            }
          >
            Lorenz
          </text>

          <line
            x1="102"
            y1="0"
            x2="130"
            y2="0"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="7 6"
            opacity="0.42"
          />

          <text
            x="138"
            y="4"
            className={
              styles.chartTick
            }
          >
            Égalité parfaite
          </text>
        </g>
      </svg>
    </div>
  );
}


function RequestedHeatmapChart({
  data,
  xLabel,
  yLabel,
}: {
  data:
    ReportChartDatum[];

  xLabel:
    string;

  yLabel:
    string;
}) {
  const cells =
    data
      .map(
        (
          datum
        ) => {
          const x =
            datumLabel(
              datum,
              "x"
            );

          const y =
            datumLabel(
              datum,
              "y"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            x ===
              null ||
            y ===
              null ||
            count ===
              null
          ) {
            return null;
          }


          return {
            x,
            y,
            count,
          };
        }
      )
      .filter(
        (
          cell
        ): cell is {
          x:
            string;

          y:
            string;

          count:
            number;
        } =>
          cell !==
          null
      );


  if (
    cells.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucune cellule de contingence
        exploitable.
      </div>
    );
  }


  const xValues =
    Array.from(
      new Set(
        cells.map(
          (
            cell
          ) =>
            cell.x
        )
      )
    );


  const yValues =
    Array.from(
      new Set(
        cells.map(
          (
            cell
          ) =>
            cell.y
        )
      )
    );


  const maxCount =
    Math.max(
      ...cells.map(
        (
          cell
        ) =>
          cell.count
      ),
      1
    );


  const width =
    860;

  const height =
    Math.max(
      230,
      120 +
      yValues.length *
        70
    );


  const padding = {
    top:
      54,

    right:
      26,

    bottom:
      58,

    left:
      120,
  };


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const cellWidth =
    plotWidth /
    Math.max(
      xValues.length,
      1
    );

  const cellHeight =
    plotHeight /
    Math.max(
      yValues.length,
      1
    );


  const countMap =
    new Map<
      string,
      number
    >();


  for (
    const cell
    of cells
  ) {
    countMap.set(
      `${cell.x}|||${cell.y}`,
      cell.count
    );
  }


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Table de contingence entre ${xLabel} et ${yLabel}`
        }
      >
        {
          xValues.map(
            (
              value,
              index
            ) => (
              <text
                key={
                  `x-${value}`
                }
                x={
                  padding.left +
                  (
                    index +
                    0.5
                  ) *
                    cellWidth
                }
                y={
                  padding.top -
                  16
                }
                textAnchor="middle"
                className={
                  styles.chartTick
                }
              >
                {
                  value
                }
              </text>
            )
          )
        }


        {
          yValues.map(
            (
              value,
              index
            ) => (
              <text
                key={
                  `y-${value}`
                }
                x={
                  padding.left -
                  12
                }
                y={
                  padding.top +
                  (
                    index +
                    0.5
                  ) *
                    cellHeight +
                  4
                }
                textAnchor="end"
                className={
                  styles.chartTick
                }
              >
                {
                  value
                }
              </text>
            )
          )
        }


        {
          yValues.flatMap(
            (
              yValue,
              yIndex
            ) =>
              xValues.map(
                (
                  xValue,
                  xIndex
                ) => {
                  const count =
                    countMap.get(
                      `${xValue}|||${yValue}`
                    ) ??
                    0;

                  const intensity =
                    count /
                    maxCount;


                  return (
                    <g
                      key={
                        `${xValue}-${yValue}`
                      }
                    >
                      <rect
                        x={
                          padding.left +
                          xIndex *
                            cellWidth +
                          2
                        }
                        y={
                          padding.top +
                          yIndex *
                            cellHeight +
                          2
                        }
                        width={
                          Math.max(
                            0,
                            cellWidth -
                            4
                          )
                        }
                        height={
                          Math.max(
                            0,
                            cellHeight -
                            4
                          )
                        }
                        rx="8"
                        fill="currentColor"
                        opacity={
                          0.10 +
                          intensity *
                            0.78
                        }
                      >
                        <title>
                          {
                            `${xLabel}: ${xValue} · ${yLabel}: ${yValue} · ${formatNumber(
                              count
                            )} observation(s)`
                          }
                        </title>
                      </rect>

                      <text
                        x={
                          padding.left +
                          (
                            xIndex +
                            0.5
                          ) *
                            cellWidth
                        }
                        y={
                          padding.top +
                          (
                            yIndex +
                            0.5
                          ) *
                            cellHeight +
                          4
                        }
                        textAnchor="middle"
                        className={
                          styles.chartTick
                        }
                        style={{
                          pointerEvents:
                            "none",

                          fontWeight:
                            700,
                        }}
                      >
                        {
                          formatChartNumber(
                            count
                          )
                        }
                      </text>
                    </g>
                  );
                }
              )
          )
        }


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            8
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          {
            xLabel
          }
        </text>


        <text
          x="18"
          y={
            padding.top +
            plotHeight /
              2
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          transform={
            `rotate(-90 18 ${
              padding.top +
              plotHeight /
                2
            })`
          }
        >
          {
            yLabel
          }
        </text>
      </svg>
    </div>
  );
}


function RequestedBoxPlotChart({
  data,
  groupLabel,
  valueLabel,
}: {
  data:
    ReportChartDatum[];

  groupLabel:
    string;

  valueLabel:
    string;
}) {
  const groups =
    data
      .map(
        (
          datum
        ) => {
          const group =
            datumLabel(
              datum,
              "group"
            );

          const minimum =
            datumNumber(
              datum,
              "min"
            );

          const q1 =
            datumNumber(
              datum,
              "q1"
            );

          const median =
            datumNumber(
              datum,
              "median"
            );

          const q3 =
            datumNumber(
              datum,
              "q3"
            );

          const maximum =
            datumNumber(
              datum,
              "max"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            group ===
              null ||
            minimum ===
              null ||
            q1 ===
              null ||
            median ===
              null ||
            q3 ===
              null ||
            maximum ===
              null
          ) {
            return null;
          }


          return {
            group,
            minimum,
            q1,
            median,
            q3,
            maximum,
            count,
          };
        }
      )
      .filter(
        (
          group
        ): group is {
          group:
            string;

          minimum:
            number;

          q1:
            number;

          median:
            number;

          q3:
            number;

          maximum:
            number;

          count:
            number |
            null;
        } =>
          group !==
          null
      );


  if (
    groups.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucun résumé de distribution
        exploitable.
      </div>
    );
  }


  const minValue =
    Math.min(
      ...groups.map(
        (
          group
        ) =>
          group.minimum
      )
    );

  const maxValue =
    Math.max(
      ...groups.map(
        (
          group
        ) =>
          group.maximum
      )
    );

  const valueRange =
    maxValue -
      minValue ||
    1;


  const width =
    860;

  const rowHeight =
    72;

  const height =
    Math.max(
      250,
      112 +
      groups.length *
        rowHeight
    );


  const padding = {
    top:
      28,

    right:
      30,

    bottom:
      62,

    left:
      130,
  };


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    value:
      number
  ) =>
    padding.left +
    (
      (
        value -
        minValue
      ) /
      valueRange
    ) *
      plotWidth;


  const tickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Distribution de ${valueLabel} selon ${groupLabel}`
        }
      >
        {
          tickRatios.map(
            (
              ratio
            ) => {
              const x =
                padding.left +
                ratio *
                  plotWidth;

              const value =
                minValue +
                ratio *
                  valueRange;


              return (
                <g
                  key={
                    ratio
                  }
                >
                  <line
                    x1={
                      x
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      x
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      x
                    }
                    y={
                      height -
                      28
                    }
                    textAnchor="middle"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatAxisNumber(
                        value
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        {
          groups.map(
            (
              group,
              index
            ) => {
              const y =
                padding.top +
                36 +
                index *
                  rowHeight;


              return (
                <g
                  key={
                    group.group
                  }
                >
                  <text
                    x={
                      padding.left -
                      14
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      group.group
                    }
                  </text>

                  <line
                    x1={
                      projectX(
                        group.minimum
                      )
                    }
                    y1={
                      y
                    }
                    x2={
                      projectX(
                        group.maximum
                      )
                    }
                    y2={
                      y
                    }
                    stroke="currentColor"
                    strokeWidth="2"
                    opacity="0.55"
                  />

                  <line
                    x1={
                      projectX(
                        group.minimum
                      )
                    }
                    y1={
                      y -
                      9
                    }
                    x2={
                      projectX(
                        group.minimum
                      )
                    }
                    y2={
                      y +
                      9
                    }
                    stroke="currentColor"
                    strokeWidth="2"
                    opacity="0.65"
                  />

                  <line
                    x1={
                      projectX(
                        group.maximum
                      )
                    }
                    y1={
                      y -
                      9
                    }
                    x2={
                      projectX(
                        group.maximum
                      )
                    }
                    y2={
                      y +
                      9
                    }
                    stroke="currentColor"
                    strokeWidth="2"
                    opacity="0.65"
                  />

                  <rect
                    x={
                      projectX(
                        group.q1
                      )
                    }
                    y={
                      y -
                      14
                    }
                    width={
                      Math.max(
                        2,
                        projectX(
                          group.q3
                        ) -
                        projectX(
                          group.q1
                        )
                      )
                    }
                    height="28"
                    rx="6"
                    fill="currentColor"
                    opacity="0.22"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <title>
                      {
                        `${groupLabel}: ${group.group} · ` +
                        `min ${formatDecimal(
                          group.minimum
                        )} · Q1 ${formatDecimal(
                          group.q1
                        )} · médiane ${formatDecimal(
                          group.median
                        )} · Q3 ${formatDecimal(
                          group.q3
                        )} · max ${formatDecimal(
                          group.maximum
                        )}` +
                        (
                          group.count !==
                            null
                            ? ` · n=${formatNumber(
                                group.count
                              )}`
                            : ""
                        )
                      }
                    </title>
                  </rect>

                  <line
                    x1={
                      projectX(
                        group.median
                      )
                    }
                    y1={
                      y -
                      17
                    }
                    x2={
                      projectX(
                        group.median
                      )
                    }
                    y2={
                      y +
                      17
                    }
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            8
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          {
            valueLabel
          }
        </text>
      </svg>
    </div>
  );
}



function FindingChart({
  finding,
}: {
  finding:
    ReportFinding;
}) {
  const xColumn =
    metricString(
      finding.metrics,
      "x_column"
    );

  const yColumn =
    metricString(
      finding.metrics,
      "y_column"
    );

  const groupColumn =
    metricString(
      finding.metrics,
      "group_column"
    );

  const measureColumn =
    metricString(
      finding.metrics,
      "measure_column"
    ) ??
    metricString(
      finding.metrics,
      "value_column"
    );

  const valueColumn =
    metricString(
      finding.metrics,
      "value_column"
    ) ??
    measureColumn;


  switch (
    finding.chart_type
  ) {
    case "line":
      return (
        <SimpleLineChart
          data={
            finding.chart_data ??
            []
          }
          xLabel={
            friendlyVariableLabel(
              metricString(
                finding.metrics,
                "time_column"
              ) ??
              "Période"
            )
          }
          yLabel={
            friendlyVariableLabel(
              measureColumn ??
              "Valeur"
            )
          }
        />
      );


    case "bar":
      return (
        <SimpleBarChart
          finding={
            finding
          }
        />
      );


    case "line_band":
      return (
        <LineBandChart
          data={
            finding.chart_data ??
            []
          }
          yLabel={
            friendlyVariableLabel(
              measureColumn ??
              "Valeur"
            )
          }
        />
      );


    case "grouped_summary":
      return (
        <GroupedSummaryChart
          data={
            finding.chart_data ??
            []
          }
        />
      );


    case "scatter":
      return (
        <ScatterPlot
          data={
            finding.chart_data ??
            []
          }
          xLabel={
            friendlyVariableLabel(
              xColumn ??
              "Variable X"
            )
          }
          yLabel={
            friendlyVariableLabel(
              yColumn ??
              "Variable Y"
            )
          }
        />
      );


    case "heatmap":
      return (
        <RequestedHeatmapChart
          data={
            finding.chart_data ??
            []
          }
          xLabel={
            friendlyVariableLabel(
              xColumn ??
              "Variable X"
            )
          }
          yLabel={
            friendlyVariableLabel(
              yColumn ??
              "Variable Y"
            )
          }
        />
      );


    case "boxplot":
      return (
        <RequestedBoxPlotChart
          data={
            finding.chart_data ??
            []
          }
          groupLabel={
            friendlyVariableLabel(
              groupColumn ??
              "Groupe"
            )
          }
          valueLabel={
            friendlyVariableLabel(
              measureColumn ??
              "Valeur"
            )
          }
        />
      );


    case "histogram":
      return (
        <NativeHistogramChart
          data={
            finding.chart_data ??
            []
          }
          valueLabel={
            friendlyVariableLabel(
              valueColumn ??
              "Valeur"
            )
          }
        />
      );


    case "lorenz":
      return (
        <RequestedLorenzChart
          data={
            finding.chart_data ??
            []
          }
        />
      );


    case "distribution":
      return (
        <GapSummaryChart
          finding={
            finding
          }
        />
      );


    default:
      return (
        <div
          className={
            styles.chartEmpty
          }
        >
          Aucune visualisation
          détaillée disponible
          pour cette analyse.
        </div>
      );
  }
}

function RagContextBlock({
  context,
}: {
  context:
    FindingRagContext |
    null;
}) {
  if (
    !context
  ) {
    return null;
  }


  if (
    context.explanation.status ===
      "ready" &&
    context.explanation.claims.length >
      0
  ) {
    return (
      <details
        className={
          styles.technicalPanel
        }
      >
        <summary>
          Contexte documentaire
          {" · "}
          vérifié
        </summary>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            {
              context
                .explanation
                .explanation
            }
          </p>
        </div>


        <div
          className={
            styles.evidenceFlow
          }
        >
          {
            context
              .explanation
              .claims
              .map(
                (
                  claim,
                  index
                ) => (
                  <article
                    className={
                      styles.evidenceItem
                    }
                    key={
                      `${
                        claim
                          .citation
                          .chunk_id
                      }-${index}`
                    }
                  >
                    <span>
                      Preuve documentaire
                    </span>

                    <strong>
                      {
                        claim
                          .citation
                          .filename
                      }
                    </strong>

                    <small>
                      {
                        claim
                          .evidence_quote
                      }
                    </small>

                    <small>
                      {
                        claim
                          .citation
                          .source_locator
                      }
                    </small>
                  </article>
                )
              )
          }
        </div>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            Relation documentaire :
            {" "}
            {
              context
                .documentary_context
                .relation_type ??
              "non spécifiée"
            }
          </p>

          <p>
            Force :
            {" "}
            {
              context
                .documentary_context
                .strength ??
              "non spécifiée"
            }
          </p>
        </div>
      </details>
    );
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        Contexte documentaire
        {" · "}
        non retenu
      </summary>

      <div
        className={
          styles.technicalReasons
        }
      >
        <p>
          {
            context
              .explanation
              .abstention_reason ??
            context
              .abstention_reason ??
            "Aucune preuve documentaire suffisamment directe n’a été retenue."
          }
        </p>
      </div>
    </details>
  );
}

function requestedAnalysisLabel(
  finding:
    ReportRequestedFinding
): string {
  switch (
    finding.kind
  ) {
    case "age_total_amount_association":
      return "Âge × montant total des achats";

    case "age_frequency_association":
      return "Âge × fréquence d’achat";

    case "age_average_basket_association":
      return "Âge × panier moyen";

    case "gender_category_association":
      return "Genre × catégorie de produits";

    case "age_category_association":
      return "Âge × catégorie de produits";

    case "revenue_moving_average":
      return "Évolution du chiffre d’affaires";

    case "revenue_by_category":
      return "Chiffre d’affaires par catégorie";

    case "customers_by_period":
      return "Clients par période";

    case "transaction_count":
      return "Nombre de transactions";

    case "products_sold_count":
      return "Produits vendus";

    case "top_products":
      return "Meilleures ventes";

    case "flop_products":
      return "Produits les moins vendus";

    case "product_category_distribution":
      return "Répartition des catégories";

    case "b2b_revenue_distribution":
      return "Répartition du chiffre d’affaires BtoB";

    case "lorenz_curve":
      return "Concentration du chiffre d’affaires";

    default:
      return finding.title;
  }
}


function analyticalGrainLabel(
  value:
    string
): string {
  switch (
    value
  ) {
    case "customer":
      return "client";

    case "transaction":
      return "transaction";

    case "product":
      return "produit";

    case "session":
      return "session";

    default:
      return friendlyVariableLabel(
        value
      );
  }
}


function descriptiveAssociationMagnitude(
  value:
    number
): string {
  const absolute =
    Math.abs(
      value
    );


  if (
    absolute <
    0.10
  ) {
    return "très faible";
  }


  if (
    absolute <
    0.30
  ) {
    return "faible";
  }


  if (
    absolute <
    0.50
  ) {
    return "modérée";
  }


  if (
    absolute <
    0.70
  ) {
    return "forte";
  }


  return "très forte";
}


function descriptiveAssociationReading(
  coefficient:
    number,

  xLabel:
    string,

  yLabel:
    string
): string {
  const magnitude =
    descriptiveAssociationMagnitude(
      coefficient
    );


  if (
    Math.abs(
      coefficient
    ) <
    0.10
  ) {
    return (
      `Dans les observations, aucune tendance monotone marquée ` +
      `n’apparaît entre ${xLabel.toLowerCase()} et ` +
      `${yLabel.toLowerCase()} (ρ = ${formatDecimal(
        coefficient
      )}).`
    );
  }


  const direction =
    coefficient >
    0
      ? "augmenter"
      : "diminuer";


  return (
    `Dans les observations, lorsque ${xLabel.toLowerCase()} augmente, ` +
    `${yLabel.toLowerCase()} tend à ${direction}. ` +
    `L’association monotone observée est ${magnitude} ` +
    `(ρ = ${formatDecimal(
      coefficient
    )}).`
  );
}


function requestedExecutionLabel(
  finding:
    ReportRequestedFinding
): string {
  switch (
    finding.execution_status
  ) {
    case "complete":
      return "Analyse complète";

    case "descriptive_only":
      return "Analyse descriptive";

    case "needs_information":
      return "Informations nécessaires";

    case "needs_specialized_method":
      return "Méthode spécialisée requise";

    default:
      return finding.execution_status
        .replace(
          /_/g,
          " "
        );
  }
}


function RequestedRagContextBlock({
  context,
}: {
  context:
    FindingRagContext |
    null;
}) {
  if (
    !context
  ) {
    return null;
  }


  if (
    context.explanation.status ===
      "ready" &&
    context.explanation.claims.length >
      0
  ) {
    return (
      <details
        className={
          styles.technicalPanel
        }
      >
        <summary>
          Éclairage métier
          {" · "}
          vérifié
        </summary>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            {
              context
                .explanation
                .explanation
            }
          </p>
        </div>


        <div
          className={
            styles.evidenceFlow
          }
        >
          {
            context
              .explanation
              .claims
              .map(
                (
                  claim,
                  index
                ) => (
                  <article
                    className={
                      styles.evidenceItem
                    }
                    key={
                      `${
                        claim
                          .citation
                          .chunk_id
                      }-${index}`
                    }
                  >
                    <span>
                      Preuve complémentaire
                    </span>

                    <strong>
                      {
                        claim
                          .citation
                          .filename
                      }
                    </strong>

                    <small>
                      {
                        claim
                          .evidence_quote
                      }
                    </small>

                    <small>
                      {
                        claim
                          .citation
                          .source_locator
                      }
                    </small>
                  </article>
                )
              )
          }
        </div>
      </details>
    );
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        Éclairage métier
        {" · "}
        aucun contexte supplémentaire
      </summary>

      <div
        className={
          styles.technicalReasons
        }
      >
        <p>
          Aucun contexte documentaire
          supplémentaire suffisamment direct
          n’a été retenu pour cette analyse.
        </p>

        <p>
          La demande source reste néanmoins
          vérifiée et traçable indépendamment
          du RAG.
        </p>
      </div>
    </details>
  );
}


function RequestedFindingCard({
  finding,
  index,
  ragContext,
}: {
  finding:
    ReportRequestedFinding;

  index:
    number;

  ragContext:
    FindingRagContext |
    null;
}) {
  const pearson =
    metricNumber(
      finding.metrics,
      "pearson_r"
    );

  const spearman =
    metricNumber(
      finding.metrics,
      "spearman_rho"
    );

  const xColumn =
    metricString(
      finding.metrics,
      "x_column"
    );

  const yColumn =
    metricString(
      finding.metrics,
      "y_column"
    );


  const xLabel =
    xColumn
      ? friendlyVariableLabel(
          xColumn
        )
      : "Variable X";


  const yLabel =
    yColumn
      ? friendlyVariableLabel(
          yColumn
        )
      : "Variable Y";


  const descriptiveReading =
    spearman !==
      null
      ? descriptiveAssociationReading(
          spearman,
          xLabel,
          yLabel
        )
      : null;


  const requestedXColumn =
    finding.variables.x ??
    xColumn ??
    null;


  const requestedYColumn =
    finding.variables.y ??
    yColumn ??
    null;


  const requestedGroupColumn =
    finding.variables.group ??
    null;


  const requestedValueColumn =
    finding.variables.value ??
    null;


  const requestedXLabel =
    requestedXColumn
      ? friendlyVariableLabel(
          requestedXColumn
        )
      : "Variable X";


  const requestedYLabel =
    requestedYColumn
      ? friendlyVariableLabel(
          requestedYColumn
        )
      : "Variable Y";


  const requestedGroupLabel =
    requestedGroupColumn
      ? friendlyVariableLabel(
          requestedGroupColumn
        )
      : "Groupe";


  const requestedValueLabel =
    requestedValueColumn
      ? friendlyVariableLabel(
          requestedValueColumn
        )
      : "Valeur";


  const transactionCount =
    metricNumber(
      finding.metrics,
      "transaction_count"
    );


  const productsSoldCount =
    metricNumber(
      finding.metrics,
      "products_sold_count"
    );


  const distinctProductsSold =
    metricNumber(
      finding.metrics,
      "distinct_products_sold"
    );


  const totalRevenue =
    metricNumber(
      finding.metrics,
      "total_revenue"
    );


  const distinctCustomersTotal =
    metricNumber(
      finding.metrics,
      "distinct_customers_total"
    );


  const periodCount =
    metricNumber(
      finding.metrics,
      "period_count"
    );


  const categoryCount =
    metricNumber(
      finding.metrics,
      "category_count"
    );


  const movingAverageWindow =
    metricNumber(
      finding.metrics,
      "moving_average_window"
    );


  const giniCoefficient =
    metricNumber(
      finding.metrics,
      "gini_coefficient"
    );


  const referenceCount =
    metricNumber(
      finding.metrics,
      "reference_count"
    );


  const rankingLimit =
    metricNumber(
      finding.metrics,
      "ranking_limit"
    );


  const rankedProductCount =
    metricNumber(
      finding.metrics,
      "ranked_product_count"
    );


  const customerCount =
    metricNumber(
      finding.metrics,
      "customer_count"
    );


  const requestedTimeSeriesValueLabel =
    finding.kind ===
      "revenue_moving_average"
      ? "Chiffre d’affaires"
      : (
          finding.kind ===
            "customers_by_period"
            ? "Clients distincts"
            : requestedValueLabel
        );


  const requestedBarValueLabel =
    (
      finding.kind ===
        "revenue_by_category" ||
      finding.kind ===
        "top_products" ||
      finding.kind ===
        "flop_products"
    )
      ? "Chiffre d’affaires"
      : (
          finding.kind ===
            "product_category_distribution"
            ? "Références distinctes"
            : requestedValueLabel
        );


  const requestedBarCategoryLabel =
    (
      finding.kind ===
        "top_products" ||
      finding.kind ===
        "flop_products"
    )
      ? "Produit"
      : (
          finding.kind ===
            "revenue_by_category" ||
          finding.kind ===
            "product_category_distribution"
            ? "Catégorie"
            : requestedGroupLabel
        );


  const requestedBarTitle =
    finding.kind ===
      "top_products"
      ? "Top produits"
      : (
          finding.kind ===
            "flop_products"
            ? "Flop produits"
            : (
                finding.kind ===
                  "product_category_distribution"
                  ? "Références par catégorie"
                  : "Chiffre d’affaires par catégorie"
              )
        );


  const requestedBarDescription =
    finding.kind ===
      "top_products"
      ? "Références classées par chiffre d’affaires décroissant."
      : (
          finding.kind ===
            "flop_products"
            ? "Références classées par chiffre d’affaires croissant."
            : (
                finding.kind ===
                  "product_category_distribution"
                  ? "Nombre de références distinctes observées dans chaque catégorie."
                  : "Comparaison descriptive du chiffre d’affaires agrégé par catégorie."
              )
        );


  return (
    <article
      className={
        styles.explanationCard
      }
      style={{
        gridColumn:
          "1 / -1",
      }}
    >
      <div
        className={
          styles.chartHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Demande
            {" "}
            {
              String(
                index + 1
              ).padStart(
                2,
                "0"
              )
            }
            {" · "}
            {
              requestedExecutionLabel(
                finding
              )
            }
          </span>


          <h3
            className={
              styles.explanationTitle
            }
          >
            {
              requestedAnalysisLabel(
                finding
              )
            }
          </h3>


          <p
            className={
              styles.resultSubtitle
            }
          >
            {
              familyLabel(
                finding.family
              )
            }

            {
              finding.analytical_grain
                ? (
                    <>
                      {" · "}
                      grain
                      {" "}
                      {
                        analyticalGrainLabel(
                          finding.analytical_grain
                        )
                      }
                    </>
                  )
                : null
            }
          </p>


          {
            xColumn &&
            yColumn
              ? (
                  <p
                    className={
                      styles.resultSubtitle
                    }
                  >
                    {
                      friendlyVariableLabel(
                        xColumn
                      )
                    }
                    {" × "}
                    {
                      friendlyVariableLabel(
                        yColumn
                      )
                    }
                  </p>
                )
              : null
          }
        </div>
      </div>


      <div
        className={
          styles.metricGrid
        }
      >
        {
          spearman !==
            null
            ? (
                <article
                  className={
                    styles.metricCard
                  }
                >
                  <span>
                    Spearman ρ
                  </span>

                  <strong>
                    {
                      formatDecimal(
                        spearman
                      )
                    }
                  </strong>
                </article>
              )
            : null
        }


        {
          pearson !==
            null
            ? (
                <article
                  className={
                    styles.metricCard
                  }
                >
                  <span>
                    Pearson r
                  </span>

                  <strong>
                    {
                      formatDecimal(
                        pearson
                      )
                    }
                  </strong>
                </article>
              )
            : null
        }


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Observations
          </span>

          <strong>
            {
              formatNumber(
                finding.sample_size
              )
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Inférence
          </span>

          <strong
            className={
              finding.inferential_status ===
                "executed"
                ? styles.statusGood
                : styles.statusNeutral
            }
          >
            {
              finding.inferential_status ===
                "executed"
                ? "Exécutée"
                : (
                    finding.inferential_status ===
                      "not_applicable"
                      ? "Non applicable"
                      : "Non sélectionnée"
                  )
            }
          </strong>
        </article>
      </div>


      {
        (
          transactionCount !==
            null ||
          productsSoldCount !==
            null ||
          distinctProductsSold !==
            null ||
          totalRevenue !==
            null ||
          distinctCustomersTotal !==
            null ||
          periodCount !==
            null ||
          categoryCount !==
            null ||
          movingAverageWindow !==
            null ||
          giniCoefficient !==
            null ||
          referenceCount !==
            null ||
          rankingLimit !==
            null ||
          rankedProductCount !==
            null ||
          customerCount !==
            null
        )
          ? (
              <div
                className={
                  styles.metricGrid
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                {
                  totalRevenue !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Chiffre d’affaires total
                          </span>

                          <strong>
                            {
                              formatChartNumber(
                                totalRevenue
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  transactionCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Transactions
                          </span>

                          <strong>
                            {
                              formatNumber(
                                transactionCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  productsSoldCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Produits vendus
                          </span>

                          <strong>
                            {
                              formatNumber(
                                productsSoldCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  distinctProductsSold !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références distinctes
                          </span>

                          <strong>
                            {
                              formatNumber(
                                distinctProductsSold
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  distinctCustomersTotal !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Clients distincts
                          </span>

                          <strong>
                            {
                              formatNumber(
                                distinctCustomersTotal
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  periodCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Périodes
                          </span>

                          <strong>
                            {
                              formatNumber(
                                periodCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  categoryCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Catégories
                          </span>

                          <strong>
                            {
                              formatNumber(
                                categoryCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  movingAverageWindow !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Fenêtre moyenne mobile
                          </span>

                          <strong>
                            {
                              `${formatNumber(
                                movingAverageWindow
                              )} périodes`
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  giniCoefficient !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Coefficient de Gini
                          </span>

                          <strong>
                            {
                              formatDecimal(
                                giniCoefficient
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  referenceCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références distinctes
                          </span>

                          <strong>
                            {
                              formatNumber(
                                referenceCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  rankingLimit !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références affichées
                          </span>

                          <strong>
                            {
                              formatNumber(
                                rankingLimit
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  rankedProductCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références classées
                          </span>

                          <strong>
                            {
                              formatNumber(
                                rankedProductCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  customerCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Clients dans Lorenz
                          </span>

                          <strong>
                            {
                              formatNumber(
                                customerCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }
              </div>
            )
          : null
      }


      {
        descriptiveReading
          ? (
              <section
                className={
                  styles.summaryPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.summaryItem
                  }
                >
                  <span>
                    Lecture descriptive
                  </span>

                  <p>
                    {
                      descriptiveReading
                    }
                  </p>

                  <small>
                    Repère générique fondé sur
                    la valeur absolue de Spearman ρ.
                    Cette lecture ne constitue ni
                    une preuve de causalité ni une
                    conclusion de significativité
                    statistique.
                  </small>
                </div>
              </section>
            )
          : null
      }


      {
        finding.summary.length >
        0
          ? (
              <details
                className={
                  styles.technicalPanel
                }
              >
                <summary>
                  Détail du résultat
                </summary>

                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  {
                    finding.summary.map(
                      (
                        item
                      ) => (
                        <p
                          key={
                            item
                          }
                        >
                          {
                            item
                          }
                        </p>
                      )
                    )
                  }
                </div>
              </details>
            )
          : null
      }


      {
        finding.chart_type ===
          "line" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Évolution temporelle
                  </h3>

                  <p>
                    {
                      finding.kind ===
                        "revenue_moving_average"
                        ? (
                            "La ligne continue montre la valeur par période ; " +
                            "la ligne pointillée montre la moyenne mobile."
                          )
                        : (
                            "Évolution du nombre de clients distincts par période."
                          )
                    }
                  </p>
                </div>

                <ExpandableChart
                  title={
                    requestedAnalysisLabel(
                      finding
                    )
                  }
                >
                  <RequestedTimeSeriesChart
                    data={
                      finding.chart_data
                    }
                    valueLabel={
                      requestedTimeSeriesValueLabel
                    }
                    showMovingAverage={
                      finding.kind ===
                        "revenue_moving_average"
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "bar" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    {
                      requestedBarTitle
                    }
                  </h3>

                  <p>
                    {
                      requestedBarDescription
                    }
                  </p>
                </div>

                <ExpandableChart
                  title={
                    requestedBarTitle
                  }
                >
                  <RequestedBarChart
                    data={
                      finding.chart_data
                    }
                    categoryLabel={
                      requestedBarCategoryLabel
                    }
                    valueLabel={
                      requestedBarValueLabel
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "scatter" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Relation observée
                  </h3>

                  <p>
                    Visualisation descriptive
                    des observations utilisées.
                  </p>
                </div>

                <ExpandableChart
                  title={
                    requestedAnalysisLabel(
                      finding
                    )
                  }
                >
                  <ScatterPlot
                    data={
                      finding.chart_data
                    }
                    xLabel={
                      xLabel
                    }
                    yLabel={
                      yLabel
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "lorenz" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Courbe de Lorenz
                  </h3>

                  <p>
                    La ligne pointillée représente
                    l’égalité parfaite. Plus la
                    courbe observée s’en éloigne,
                    plus le chiffre d’affaires est
                    concentré entre certains clients.
                  </p>
                </div>

                <ExpandableChart
                  title="Courbe de Lorenz"
                >
                  <RequestedLorenzChart
                    data={
                      finding.chart_data
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "heatmap" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Table de contingence
                  </h3>

                  <p>
                    Plus une cellule est intense,
                    plus le nombre d’observations
                    est élevé.
                  </p>
                </div>

                <ExpandableChart
                  title="Table de contingence"
                >
                  <RequestedHeatmapChart
                    data={
                      finding.chart_data
                    }
                    xLabel={
                      requestedXLabel
                    }
                    yLabel={
                      requestedYLabel
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "boxplot" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Distribution par groupe
                  </h3>

                  <p>
                    Min, quartiles, médiane et max
                    sont affichés pour chaque groupe.
                  </p>
                </div>

                <ExpandableChart
                  title="Distribution par groupe"
                >
                  <RequestedBoxPlotChart
                    data={
                      finding.chart_data
                    }
                    groupLabel={
                      requestedGroupLabel
                    }
                    valueLabel={
                      requestedValueLabel
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      <details
        className={
          styles.technicalPanel
        }
      >
        <summary>
          Demande source
          {" · "}
          vérifiée
        </summary>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            «
            {
              finding.evidence_quote
                .replace(
                  /,\s*$/,
                  ""
                )
            }
            »
          </p>
        </div>


        <div
          className={
            styles.evidenceFlow
          }
        >
          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Document
            </span>

            <strong>
              {
                finding.source_filename
              }
            </strong>

            <small>
              {
                finding.source_locator
              }
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Page
            </span>

            <strong>
              {
                finding.page_number ??
                "—"
              }
            </strong>

            <small>
              Localisation vérifiée
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Unité de preuve
            </span>

            <strong>
              {
                finding.evidence_unit_id ??
                "—"
              }
            </strong>

            <small>
              Élément documentaire
              ayant déclenché l’analyse
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Provenance
            </span>

            <strong>
              Vérifiée
            </strong>

            <small>
              {
                finding.adapter_rule_version
              }
            </small>
          </article>
        </div>
      </details>


      <RequestedRagContextBlock
        context={
          ragContext
        }
      />


      {
        finding.caveats.length >
        0
          ? (
              <details
                className={
                  styles.technicalPanel
                }
              >
                <summary>
                  Limites et précautions
                </summary>

                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  {
                    finding.caveats.map(
                      (
                        caveat
                      ) => (
                        <p
                          key={
                            caveat
                          }
                        >
                          {
                            caveat
                          }
                        </p>
                      )
                    )
                  }
                </div>
              </details>
            )
          : null
      }
    </article>
  );
}

function NativeHistogramChart({
  data,
  valueLabel,
  lowerBound =
    null,
  upperBound =
    null,
  highlightOutliers =
    false,
}: {
  data:
    ReportChartDatum[];

  valueLabel:
    string;

  lowerBound?:
    number |
    null;

  upperBound?:
    number |
    null;

  highlightOutliers?:
    boolean;
}) {
  const [
    hoveredIndex,
    setHoveredIndex,
  ] = useState<
    number |
    null
  >(
    null
  );


  const bins =
    data
      .map(
        (
          datum,
          index
        ) => {
          const start =
            datumNumber(
              datum,
              "bin_start"
            );

          const end =
            datumNumber(
              datum,
              "bin_end"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            start ===
              null ||
            end ===
              null ||
            count ===
              null
          ) {
            return null;
          }


          return {
            index,
            start,
            end,
            count,
          };
        }
      )
      .filter(
        (
          bin
        ): bin is {
          index:
            number;

          start:
            number;

          end:
            number;

          count:
            number;
        } =>
          bin !==
          null
      );


  if (
    bins.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucun intervalle exploitable
        pour l’histogramme.
      </div>
    );
  }


  const width =
    860;

  const height =
    350;


  const padding = {
    top:
      32,

    right:
      28,

    bottom:
      68,

    left:
      78,
  };


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const maxCount =
    Math.max(
      ...bins.map(
        (
          bin
        ) =>
          bin.count
      ),
      1
    );


  const barGap =
    4;


  const barWidth =
    plotWidth /
    bins.length;


  const xMin =
    bins[
      0
    ].start;


  const xMax =
    bins[
      bins.length -
      1
    ].end;


  const xRange =
    xMax -
      xMin ||
    1;


  const projectXValue = (
    value:
      number
  ) =>
    padding.left +
    (
      (
        value -
        xMin
      ) /
      xRange
    ) *
      plotWidth;


  const lowerBoundVisible =
    lowerBound !==
      null &&
    Number.isFinite(
      lowerBound
    ) &&
    lowerBound >
      xMin &&
    lowerBound <
      xMax;


  const upperBoundVisible =
    upperBound !==
      null &&
    Number.isFinite(
      upperBound
    ) &&
    upperBound >
      xMin &&
    upperBound <
      xMax;


  const projectY = (
    count:
      number
  ) =>
    padding.top +
    plotHeight -
    (
      count /
      maxCount
    ) *
      plotHeight;


  const yTickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const middleIndex =
    Math.floor(
      (
        bins.length -
        1
      ) /
      2
    );


  const xTicks = [
    {
      index:
        0,

      value:
        bins[
          0
        ].start,

      anchor:
        "start" as const,
    },
    {
      index:
        middleIndex,

      value:
        (
          bins[
            middleIndex
          ].start +
          bins[
            middleIndex
          ].end
        ) /
        2,

      anchor:
        "middle" as const,
    },
    {
      index:
        bins.length -
        1,

      value:
        bins[
          bins.length -
          1
        ].end,

      anchor:
        "end" as const,
    },
  ];


  const hoveredBin =
    hoveredIndex !==
      null
      ? bins[
          hoveredIndex
        ] ??
        null
      : null;


  const tooltipWidth =
    230;

  const tooltipHeight =
    84;


  const tooltipPosition =
    hoveredBin &&
    hoveredIndex !==
      null
      ? clampChartTooltipPosition(
          padding.left +
            (
              hoveredIndex +
              0.5
            ) *
              barWidth,
          projectY(
            hoveredBin.count
          ),
          tooltipWidth,
          tooltipHeight,
          width,
          height
        )
      : null;


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Histogramme de ${valueLabel}`
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          yTickRatios.map(
            (
              ratio
            ) => {
              const count =
                maxCount *
                ratio;

              const y =
                projectY(
                  count
                );


              return (
                <g
                  key={
                    `hist-y-${ratio}`
                  }
                >
                  <line
                    x1={
                      padding.left
                    }
                    y1={
                      y
                    }
                    x2={
                      padding.left +
                      plotWidth
                    }
                    y2={
                      y
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      padding.left -
                      10
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                    style={{
                      fontSize:
                        "12px",

                      opacity:
                        0.82,
                    }}
                  >
                    {
                      formatNumber(
                        Math.round(
                          count
                        )
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <line
          x1={
            padding.left
          }
          y1={
            padding.top
          }
          x2={
            padding.left
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        {
          highlightOutliers &&
          lowerBoundVisible &&
          lowerBound !==
            null
            ? (
                <>
                  <rect
                    x={
                      padding.left
                    }
                    y={
                      padding.top
                    }
                    width={
                      Math.max(
                        0,
                        projectXValue(
                          lowerBound
                        ) -
                        padding.left
                      )
                    }
                    height={
                      plotHeight
                    }
                    className={
                      styles.outlierZone
                    }
                  />

                  <line
                    x1={
                      projectXValue(
                        lowerBound
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectXValue(
                        lowerBound
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.outlierThreshold
                    }
                  />
                </>
              )
            : null
        }


        {
          highlightOutliers &&
          upperBoundVisible &&
          upperBound !==
            null
            ? (
                <>
                  <rect
                    x={
                      projectXValue(
                        upperBound
                      )
                    }
                    y={
                      padding.top
                    }
                    width={
                      Math.max(
                        0,
                        padding.left +
                        plotWidth -
                        projectXValue(
                          upperBound
                        )
                      )
                    }
                    height={
                      plotHeight
                    }
                    className={
                      styles.outlierZone
                    }
                  />

                  <line
                    x1={
                      projectXValue(
                        upperBound
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectXValue(
                        upperBound
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.outlierThreshold
                    }
                  />
                </>
              )
            : null
        }


        {
          bins.map(
            (
              bin,
              index
            ) => {
              const x =
                padding.left +
                index *
                  barWidth +
                barGap /
                  2;

              const y =
                projectY(
                  bin.count
                );

              const heightValue =
                padding.top +
                plotHeight -
                y;


              return (
                <g
                  key={
                    `${
                      bin.index
                    }-${
                      bin.start
                    }-${
                      bin.end
                    }`
                  }
                >
                  <rect
                    x={
                      x
                    }
                    y={
                      y
                    }
                    width={
                      Math.max(
                        1,
                        barWidth -
                        barGap
                      )
                    }
                    height={
                      Math.max(
                        1,
                        heightValue
                      )
                    }
                    rx="4"
                    fill="currentColor"
                    opacity={
                      hoveredIndex ===
                        index
                        ? 0.92
                        : 0.68
                    }
                    onMouseEnter={
                      () =>
                        setHoveredIndex(
                          index
                        )
                    }
                    onFocus={
                      () =>
                        setHoveredIndex(
                          index
                        )
                    }
                    onBlur={
                      () =>
                        setHoveredIndex(
                          null
                        )
                    }
                    tabIndex={
                      0
                    }
                    aria-label={
                      `${formatAxisNumber(
                        bin.start
                      )} à ${formatAxisNumber(
                        bin.end
                      )}, ${formatNumber(
                        bin.count
                      )} observations`
                    }
                  >
                    <title>
                      {
                        `${
                          formatAxisNumber(
                            bin.start
                          )
                        } — ${
                          formatAxisNumber(
                            bin.end
                          )
                        } · ${
                          formatNumber(
                            bin.count
                          )
                        } observation(s)`
                      }
                    </title>
                  </rect>


                  {
                    bins.length <=
                    20
                      ? (
                          <text
                            x={
                              x +
                              Math.max(
                                1,
                                barWidth -
                                barGap
                              ) /
                              2
                            }
                            y={
                              Math.max(
                                14,
                                y -
                                7
                              )
                            }
                            textAnchor="middle"
                            className={
                              styles.chartTick
                            }
                            style={{
                              fontSize:
                                "10px",

                              opacity:
                                0.75,
                            }}
                          >
                            {
                              formatNumber(
                                bin.count
                              )
                            }
                          </text>
                        )
                      : null
                  }
                </g>
              );
            }
          )
        }


        {
          hoveredBin &&
          tooltipPosition &&
          hoveredIndex !==
            null
            ? (
                <SvgChartTooltip
                  x={
                    tooltipPosition.x
                  }
                  y={
                    tooltipPosition.y
                  }
                  width={
                    tooltipWidth
                  }
                  lines={
                    [
                      `${valueLabel}`,
                      `Intervalle : ${formatAxisNumber(
                        hoveredBin.start
                      )} — ${formatAxisNumber(
                        hoveredBin.end
                      )}`,
                      `Observations : ${formatNumber(
                        hoveredBin.count
                      )}`,
                    ]
                  }
                />
              )
            : null
        }


        {
          xTicks.map(
            (
              tick
            ) => {
              const x =
                tick.index ===
                  bins.length -
                    1
                  ? padding.left +
                    plotWidth
                  : (
                      tick.index ===
                        0
                        ? padding.left
                        : padding.left +
                          (
                            tick.index +
                            0.5
                          ) *
                            barWidth
                    );


              return (
                <text
                  key={
                    `hist-x-${tick.index}`
                  }
                  x={
                    x
                  }
                  y={
                    padding.top +
                    plotHeight +
                    24
                  }
                  textAnchor={
                    tick.anchor
                  }
                  className={
                    styles.chartTick
                  }
                  style={{
                    fontSize:
                      "12px",

                    opacity:
                      0.82,
                  }}
                >
                  {
                    formatAxisNumber(
                      tick.value
                    )
                  }
                </text>
              );
            }
          )
        }


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            10
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          style={{
            fontSize:
              "12px",

            opacity:
              0.76,
          }}
        >
          {
            valueLabel
          }
        </text>


        <text
          x="20"
          y={
            padding.top +
            plotHeight /
              2
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          transform={
            `rotate(-90 20 ${
              padding.top +
              plotHeight /
                2
            })`
          }
          style={{
            fontSize:
              "12px",

            opacity:
              0.76,
          }}
        >
          Observations
        </text>
      </svg>
    </div>
  );
}



function entityEvidenceDirectionLabel(
  direction:
    string
): string {
  switch (
    direction
  ) {
    case "high":
      return "Valeur élevée";

    case "low":
      return "Valeur faible";

    default:
      return direction
        .replace(
          /_/g,
          " "
        );
  }
}


function formatEntityEvidenceValue(
  evidence:
    EntityOutlierFindingEvidenceView
): string {
  if (
    evidence.metric ===
      "purchase_sessions" ||
    evidence.metric ===
      "total_items"
  ) {
    return formatNumber(
      Math.round(
        evidence.value
      )
    );
  }


  return formatDecimal(
    evidence.value
  );
}


function EntityOutlierRequestedAnswer({
  finding,
  objective,
}: {
  finding:
    EntityOutlierFindingView;

  objective:
    string;
}) {
  const titleId =
    `entity-outlier-answer-${finding.analysis_id.replace(
      /[^a-zA-Z0-9_-]/g,
      "-"
    )}`;


  if (
    finding.status ===
    "blocked"
  ) {
    return (
      <section
        className={
          `${styles.requestedAnswerGroup} ${styles.entityAnswerGroup}`
        }
        aria-labelledby={
          titleId
        }
      >
        <div
          className={
            styles.requestedAnswerGroupHeader
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Votre demande
            </span>

            <h2
              id={
                titleId
              }
            >
              Analyse d’entités non exécutable
            </h2>

            <p
              className={
                styles.requestedQuestion
              }
            >
              «
              {
                objective.trim()
              }
              »
            </p>
          </div>

          <span
            className={
              styles.requestedAnswerCount
            }
          >
            Analyse bloquée
          </span>
        </div>


        <section
          className={
            styles.entityAnswerCard
          }
        >
          <div
            className={
              styles.technicalReasons
            }
          >
            <strong>
              DataLens n’invente pas le grain client
            </strong>

            {
              finding.blockers.length >
              0
                ? finding.blockers.map(
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
                : (
                    <p>
                      Aucun contexte analytique client
                      suffisamment sûr n’a pu être
                      construit pour cette demande.
                    </p>
                  )
            }
          </div>
        </section>
      </section>
    );
  }


  const priorityProfiles =
    finding.priority_profiles
      .slice(
        0,
        8
      );


  return (
    <section
      className={
        `${styles.requestedAnswerGroup} ${styles.entityAnswerGroup}`
      }
      aria-labelledby={
        titleId
      }
    >
      <div
        className={
          styles.requestedAnswerGroupHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Votre demande
          </span>

          <h2
            id={
              titleId
            }
          >
            Réponse à votre demande
          </h2>

          <p
            className={
              styles.requestedQuestion
            }
          >
            «
            {
              objective.trim()
            }
            »
          </p>
        </div>


        <span
          className={
            styles.requestedAnswerCount
          }
        >
          {
            finding.priority_profile_count
          }
          {" profil"}
          {
            finding.priority_profile_count >
            1
              ? "s"
              : ""
          }
          {" prioritaire"}
          {
            finding.priority_profile_count >
            1
              ? "s"
              : ""
          }
        </span>
      </div>


      <section
        className={
          styles.entityAnswerCard
        }
      >
        <div
          className={
            styles.entityAnswerLead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Résultat vérifié · grain client
            </span>

            <h3
              className={
                styles.entityAnswerTitle
              }
            >
              {
                finding.priority_profile_count ===
                  0
                  ? "Aucun profil extrêmement atypique détecté"
                  : finding.priority_profile_count ===
                      1
                    ? "1 profil client nécessite une revue prioritaire"
                    : `${formatNumber(
                        finding.priority_profile_count
                      )} profils clients nécessitent une revue prioritaire`
              }
            </h3>

            <p
              className={
                styles.entityAnswerDescription
              }
            >
              {
                formatNumber(
                  finding.entity_count
                )
              }
              {" clients analysés · "}
              {
                finding.dataset_filename ??
                "vue analytique client"
              }
            </p>
          </div>


          <div
            className={
              styles.entityAnswerStats
            }
          >
            <article>
              <span>
                Clients analysés
              </span>

              <strong>
                {
                  formatNumber(
                    finding.entity_count
                  )
                }
              </strong>
            </article>

            <article>
              <span>
                Profils prioritaires
              </span>

              <strong>
                {
                  formatNumber(
                    finding.priority_profile_count
                  )
                }
              </strong>
            </article>

            <article>
              <span>
                Signaux secondaires
              </span>

              <strong>
                {
                  formatNumber(
                    finding.behavioral_signal_count
                  )
                }
              </strong>
            </article>
          </div>
        </div>


        {
          priorityProfiles.length >
          0
            ? (
                <div
                  className={
                    styles.entityProfileGrid
                  }
                >
                  {
                    priorityProfiles.map(
                      (
                        profile,
                        profileIndex
                      ) => (
                        <article
                          className={
                            styles.entityProfileCard
                          }
                          key={
                            profile.entity
                          }
                        >
                          <header
                            className={
                              styles.entityProfileHeader
                            }
                          >
                            <div>
                              <span
                                className={
                                  styles.entityProfileIndex
                                }
                              >
                                Profil
                                {" "}
                                {
                                  String(
                                    profileIndex +
                                    1
                                  ).padStart(
                                    2,
                                    "0"
                                  )
                                }
                              </span>

                              <h3>
                                {
                                  profile.entity
                                }
                              </h3>

                              <p>
                                {
                                  profile.dominant_family_label
                                }
                              </p>
                            </div>

                            <span
                              className={
                                styles.entitySignalBadge
                              }
                            >
                              {
                                profile.signal_count
                              }
                              {" signal"}
                              {
                                profile.signal_count >
                                1
                                  ? "s"
                                  : ""
                              }
                            </span>
                          </header>


                          <div
                            className={
                              styles.entityMetricGrid
                            }
                          >
                            {
                              profile.evidence
                                .slice(
                                  0,
                                  3
                                )
                                .map(
                                  (
                                    evidence
                                  ) => (
                                    <div
                                      className={
                                        styles.entityMetric
                                      }
                                      key={
                                        `${profile.entity}-${evidence.metric}`
                                      }
                                    >
                                      <span>
                                        {
                                          evidence.metric_label
                                        }
                                      </span>

                                      <strong>
                                        {
                                          formatEntityEvidenceValue(
                                            evidence
                                          )
                                        }
                                      </strong>
                                    </div>
                                  )
                                )
                            }
                          </div>


                          <details
                            className={
                              styles.entityProfileDetails
                            }
                          >
                            <summary>
                              Voir pourquoi ce profil ressort
                            </summary>

                            <div
                              className={
                                styles.entityProfileEvidence
                              }
                            >
                              <p>
                                {
                                  profile.explanation
                                }
                              </p>

                              {
                                profile.evidence.map(
                                  (
                                    evidence
                                  ) => (
                                    <div
                                      key={
                                        `${profile.entity}-${evidence.metric}-detail`
                                      }
                                    >
                                      <span>
                                        {
                                          evidence.metric_label
                                        }
                                      </span>

                                      <strong>
                                        {
                                          entityEvidenceDirectionLabel(
                                            evidence.direction
                                          )
                                        }
                                        {" · "}
                                        {
                                          formatDecimal(
                                            evidence.distance_iqr
                                          )
                                        }
                                        {" IQR au-delà du seuil"}
                                      </strong>
                                    </div>
                                  )
                                )
                              }
                            </div>
                          </details>
                        </article>
                      )
                    )
                  }
                </div>
              )
            : null
        }


        <div
          className={
            styles.entityAnswerFooter
          }
        >
          {
            finding.behavioral_signal_count >
            0
              ? (
                  <details
                    className={
                      styles.entitySecondaryDetails
                    }
                  >
                    <summary>
                      <span>
                        Autres signaux comportementaux
                      </span>

                      <strong>
                        {
                          formatNumber(
                            finding.behavioral_signal_count
                          )
                        }
                      </strong>
                    </summary>

                    <div>
                      <p>
                        Ces clients franchissent au moins une
                        borne statistique, mais ne présentent pas
                        le même niveau d’atypicité globale que les
                        profils prioritaires affichés ci-dessus.
                      </p>

                      <p>
                        Ils restent des pistes d’exploration.
                        DataLens ne les classe pas automatiquement
                        comme profils extrêmes.
                      </p>
                    </div>
                  </details>
                )
              : null
          }


          <details
            className={
              styles.entityMethodDetails
            }
          >
            <summary>
              Preuves & méthodologie
            </summary>

            <div
              className={
                styles.entityMethodBody
              }
            >
              {
                finding.methodology.length >
                0
                  ? (
                      <section>
                        <strong>
                          Méthodologie
                        </strong>

                        {
                          finding.methodology.map(
                            (
                              item
                            ) => (
                              <p
                                key={
                                  item
                                }
                              >
                                {
                                  item
                                }
                              </p>
                            )
                          )
                        }
                      </section>
                    )
                  : null
              }


              {
                finding.caveats.length >
                0
                  ? (
                      <section>
                        <strong>
                          Précautions
                        </strong>

                        {
                          finding.caveats.map(
                            (
                              caveat
                            ) => (
                              <p
                                key={
                                  caveat
                                }
                              >
                                {
                                  caveat
                                }
                              </p>
                            )
                          )
                        }
                      </section>
                    )
                  : null
              }


              <div
                className={
                  styles.entityTraceGrid
                }
              >
                <article>
                  <span>
                    Grain
                  </span>

                  <strong>
                    Client
                  </strong>

                  <small>
                    {
                      finding.entity_column ??
                      "Entité dérivée"
                    }
                  </small>
                </article>

                <article>
                  <span>
                    Signaux IQR bruts
                  </span>

                  <strong>
                    {
                      formatNumber(
                        finding.raw_flagged_entity_count
                      )
                    }
                  </strong>

                  <small>
                    Au moins une borne franchie
                  </small>
                </article>

                <article>
                  <span>
                    Adaptation produit
                  </span>

                  <strong>
                    Vérifiée
                  </strong>

                  <small>
                    {
                      finding.adapter_rule_version
                    }
                  </small>
                </article>
              </div>
            </div>
          </details>
        </div>
      </section>
    </section>
  );
}


function PlannerBlockedAnalysisCard({
  planner,
  objective,
}: {
  planner:
    AIPlannerReportView;

  objective:
    string;
}) {
  const blockedItem =
    planner.items.find(
      (
        candidate
      ) =>
        candidate.validation_status ===
          "blocked" ||
        candidate.validation_status ===
          "rejected" ||
        candidate.validation_status ===
          "ambiguous"
    ) ??
    null;


  if (
    !blockedItem
  ) {
    return null;
  }


  const contract =
    blockedItem.contract;


  const statusLabel =
    blockedItem.validation_status ===
      "blocked"
      ? "Analyse bloquée"
      : blockedItem.validation_status ===
          "rejected"
        ? "Plan rejeté"
        : "Plan ambigu";


  const reasons =
    [
      ...(
        contract
          ?.blockers ??
        []
      ),
      ...blockedItem.errors,
      ...(
        blockedItem.proposal
          .blockers ??
        []
      ),
    ].filter(
      (
        reason,
        index,
        values
      ) =>
        Boolean(
          reason
        ) &&
        values.indexOf(
          reason
        ) ===
          index
    );


  return (
    <section
      aria-labelledby="requested-ai-planner-blocked-title"
      style={{
        marginBottom:
          "26px",

        padding:
          "18px",

        border:
          "1px solid rgba(255, 178, 92, 0.24)",

        borderRadius:
          "16px",

        background:
          "linear-gradient(180deg, rgba(139, 83, 22, 0.10), rgba(10, 18, 32, 0.20))",
      }}
    >
      <div
        className={
          styles.sectionHead
        }
        style={{
          marginBottom:
            "14px",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Analyse demandée
          </span>

          <h2
            id="requested-ai-planner-blocked-title"
          >
            {
              blockedItem.proposal.title ||
              objective.trim() ||
              "Analyse demandée"
            }
          </h2>

          <p
            className={
              styles.resultSubtitle
            }
          >
            {
              objective.trim() ||
              planner.objective
            }
          </p>
        </div>


        <div
          style={{
            display:
              "flex",

            gap:
              "7px",

            flexWrap:
              "wrap",

            justifyContent:
              "flex-end",
          }}
        >
          <span
            style={{
              padding:
                "6px 9px",

              border:
                "1px solid rgba(255, 178, 92, 0.28)",

              borderRadius:
                "999px",

              fontSize:
                "0.69rem",

              fontWeight:
                700,
            }}
          >
            {
              familyLabel(
                blockedItem.proposal.family
              )
            }
          </span>

          <span
            style={{
              padding:
                "6px 9px",

              border:
                "1px solid rgba(255, 178, 92, 0.32)",

              borderRadius:
                "999px",

              fontSize:
                "0.69rem",

              fontWeight:
                700,
            }}
          >
            {
              statusLabel
            }
          </span>
        </div>
      </div>


      <div
        className={
          styles.technicalReasons
        }
        style={{
          marginBottom:
            "12px",
        }}
      >
        <strong>
          DataLens ne substitue pas silencieusement
          une variable demandée
        </strong>

        <p>
          Le plan analytique n’a pas franchi la validation
          déterministe. L’exploration automatique peut
          continuer séparément, mais elle ne remplace pas
          cette demande utilisateur.
        </p>

        {
          reasons.length >
          0
            ? (
                reasons.map(
                  (
                    reason,
                    reasonIndex
                  ) => (
                    <p
                      key={
                        `${reasonIndex}-${reason}`
                      }
                    >
                      {
                        reason
                      }
                    </p>
                  )
              )
            )
          : (
              <p>
                Le planner n’a pas produit de contrat
                exécutable pour cet objectif.
              </p>
            )
        }
      </div>


      <div
        className={
          styles.evidenceFlow
        }
      >
        <article
          className={
            styles.evidenceItem
          }
        >
          <span>
            Planner
          </span>

          <strong>
            {
              planner.model
            }
          </strong>

          <small>
            {
              planner.attempt_count ??
              1
            }
            {" tentative(s)"}
          </small>
        </article>


        <article
          className={
            styles.evidenceItem
          }
        >
          <span>
            Validation
          </span>

          <strong>
            Python
          </strong>

          <small>
            {
              statusLabel
            }
          </small>
        </article>


        <article
          className={
            styles.evidenceItem
          }
        >
          <span>
            Tool calling
          </span>

          <strong>
            Non exécuté
          </strong>

          <small>
            Aucun contrat validé
          </small>
        </article>


        <article
          className={
            styles.evidenceItem
          }
        >
          <span>
            Calcul demandé
          </span>

          <strong>
            Non exécuté
          </strong>

          <small>
            Aucun résultat statistique produit
          </small>
        </article>
      </div>


      {
        planner.retry_triggered
          ? (
              <div
                className={
                  styles.technicalReasons
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <p>
                  Un retry contrôlé du planner a été effectué.
                  Python a conservé le garde-fou et la demande
                  est restée non exécutable.
                </p>
              </div>
            )
          : null
      }
    </section>
  );
}


function NativeRequestedAnalysisSingleCard({
  report,
  objective,
}: {
  report:
    AINativePipelineReportView;

  objective:
    string;
}) {
  const item =
    report.items.find(
      (
        candidate
      ) =>
        candidate.pipeline_status ===
        "executed" &&
        candidate.native_tool
          ?.execution
          ?.result
          !==
          null
    ) ??
    null;


  const nativeTool =
    item
      ?.native_tool ??
    null;


  const execution =
    nativeTool
      ?.execution ??
    null;


  const result =
    execution
      ?.result ??
    null;


  if (
    !item ||
    !nativeTool ||
    !execution ||
    !result
  ) {
    const blockedItem =
      report.planner.items.find(
        (
          candidate
        ) =>
          candidate.validation_status ===
            "blocked" ||
          candidate.validation_status ===
            "rejected" ||
          candidate.validation_status ===
            "ambiguous"
      ) ??
      null;


    if (
      !blockedItem
    ) {
      return null;
    }


    const contract =
      blockedItem.contract;


    const statusLabel =
      blockedItem.validation_status ===
        "blocked"
        ? "Analyse bloquée"
        : blockedItem.validation_status ===
            "rejected"
          ? "Plan rejeté"
          : "Plan ambigu";


    const reasons =
      [
        ...(
          contract
            ?.blockers ??
          []
        ),
        ...blockedItem.errors,
        ...(
          blockedItem.proposal
            .blockers ??
          []
        ),
      ].filter(
        (
          reason,
          index,
          values
        ) =>
          Boolean(
            reason
          ) &&
          values.indexOf(
            reason
          ) ===
            index
      );


    return (
      <section
        aria-labelledby="requested-ai-analysis-blocked-title"
        style={{
          marginBottom:
            "26px",

          padding:
            "18px",

          border:
            "1px solid rgba(255, 178, 92, 0.24)",

          borderRadius:
            "16px",

          background:
            "linear-gradient(180deg, rgba(139, 83, 22, 0.10), rgba(10, 18, 32, 0.20))",
        }}
      >
        <div
          className={
            styles.sectionHead
          }
          style={{
            marginBottom:
              "14px",
          }}
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Analyse demandée
            </span>

            <h2
              id="requested-ai-analysis-blocked-title"
            >
              {
                blockedItem.proposal.title ||
                objective.trim() ||
                "Analyse demandée"
              }
            </h2>

            <p
              className={
                styles.resultSubtitle
              }
            >
              {
                objective.trim() ||
                report.planner.objective
              }
            </p>
          </div>


          <div
            style={{
              display:
                "flex",

              gap:
                "7px",

              flexWrap:
                "wrap",

              justifyContent:
                "flex-end",
            }}
          >
            <span
              style={{
                padding:
                  "6px 9px",

                border:
                  "1px solid rgba(255, 178, 92, 0.28)",

                borderRadius:
                  "999px",

                fontSize:
                  "0.69rem",

                fontWeight:
                  700,
              }}
            >
              {
                familyLabel(
                  blockedItem.proposal.family
                )
              }
            </span>

            <span
              style={{
                padding:
                  "6px 9px",

                border:
                  "1px solid rgba(255, 178, 92, 0.32)",

                borderRadius:
                  "999px",

                fontSize:
                  "0.69rem",

                fontWeight:
                  700,
              }}
            >
              {
                statusLabel
              }
            </span>
          </div>
        </div>


        <div
          className={
            styles.technicalReasons
          }
          style={{
            marginBottom:
              "12px",
          }}
        >
          <strong>
            DataLens n’exécute pas cette analyse
          </strong>

          <p>
            Le plan n’a pas franchi la validation
            déterministe. Aucun outil natif ni calcul
            statistique n’a été lancé pour cette demande.
          </p>

          {
            reasons.length >
            0
              ? (
                  reasons.map(
                    (
                      reason,
                      reasonIndex
                    ) => (
                      <p
                        key={
                          `${reasonIndex}-${reason}`
                        }
                      >
                        {
                          reason
                        }
                      </p>
                    )
                )
              )
            : (
                <p>
                  Le planner n’a pas produit de contrat
                  exécutable pour cet objectif.
                </p>
              )
          }
        </div>


        <div
          className={
            styles.evidenceFlow
          }
        >
          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Planner
            </span>

            <strong>
              {
                plannerEngineLabel(
                  report.planner_model
                )
              }
            </strong>

            <small>
              {
                report.planner_model
              }
              {" · "}
              {
                report.planner
                  .attempt_count ??
                1
              }
              {" tentative(s)"}
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Validation
            </span>

            <strong>
              Python
            </strong>

            <small>
              {
                statusLabel
              }
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Tool calling
            </span>

            <strong>
              Non exécuté
            </strong>

            <small>
              Qwen n’est pas autorisé à poursuivre
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Calcul
            </span>

            <strong>
              Non exécuté
            </strong>

            <small>
              Aucun résultat statistique produit
            </small>
          </article>
        </div>


        {
          report.planner.retry_triggered
            ? (
                <div
                  className={
                    styles.technicalReasons
                  }
                  style={{
                    marginTop:
                      "12px",
                  }}
                >
                  <p>
                    Le planner a reçu un retry contrôlé après
                    le rejet initial. Python a conservé le
                    garde-fou et aucune substitution silencieuse
                    n’a été exécutée.
                  </p>
                </div>
              )
            : null
        }
      </section>
    );
  }


  const variables =
    execution
      .arguments
      .variables ??
    {};


  const chartData =
    result.chart_data ??
    [];


  const xColumn =
    variables.x ??
    null;


  const yColumn =
    variables.y ??
    null;


  const groupColumn =
    variables.group ??
    null;


  const valueColumn =
    variables.value ??
    null;


  const timeColumn =
    variables.time ??
    null;


  const statisticalResult =
    result.statistical_result ??
    {};


  const count =
    metricNumber(
      result.metrics,
      "count"
    ) ??
    metricNumber(
      result.metrics,
      "valid_observations"
    ) ??
    metricNumber(
      result.metrics,
      "valid_pairs"
    ) ??
    (
      typeof statisticalResult.n ===
        "number"
        ? statisticalResult.n
        : null
    );


  const mean =
    metricNumber(
      result.metrics,
      "mean"
    );


  const median =
    metricNumber(
      result.metrics,
      "median"
    );


  const standardDeviation =
    metricNumber(
      result.metrics,
      "std"
    );


  const coefficient =
    metricNumber(
      result.metrics,
      "coefficient"
    ) ??
    (
      typeof statisticalResult.coefficient ===
        "number"
        ? statisticalResult.coefficient
        : null
    );


  const pValue =
    metricNumber(
      result.metrics,
      "p_value"
    ) ??
    (
      typeof statisticalResult.p_value ===
        "number"
        ? statisticalResult.p_value
        : null
    );


  const cramersV =
    typeof statisticalResult.cramers_v ===
      "number"
      ? statisticalResult.cramers_v
      : null;


  const groupCount =
    metricNumber(
      result.metrics,
      "group_count"
    );


  const periodCount =
    metricNumber(
      result.metrics,
      "period_count"
    );


  const timeStart =
    result.metrics[
      "time_start"
    ];


  const timeEnd =
    result.metrics[
      "time_end"
    ];


  const periodMedianMin =
    metricNumber(
      result.metrics,
      "period_median_min"
    );


  const periodMedianMax =
    metricNumber(
      result.metrics,
      "period_median_max"
    );


  const q1 =
    metricNumber(
      result.metrics,
      "q1"
    );


  const q3 =
    metricNumber(
      result.metrics,
      "q3"
    );


  const iqr =
    metricNumber(
      result.metrics,
      "iqr"
    );


  const outlierCountIqr =
    metricNumber(
      result.metrics,
      "outlier_count_iqr"
    );


  const outlierRatioIqr =
    metricNumber(
      result.metrics,
      "outlier_ratio_iqr"
    );


  const isOutlierRequest =
    result.family ===
      "distribution" &&
    /outlier|atypiqu|aberrant/i.test(
      objective
    );


  const lowerIqrBound =
    q1 !==
      null &&
    iqr !==
      null
      ? q1 -
        1.5 *
        iqr
      : null;


  const upperIqrBound =
    q3 !==
      null &&
    iqr !==
      null
      ? q3 +
        1.5 *
        iqr
      : null;


  const requestedVariableLabel =
    friendlyVariableLabel(
      valueColumn ??
      metricString(
        result.metrics,
        "column"
      ) ??
      "Valeur"
    );


  const requestedResultTitle =
    isOutlierRequest
      ? `Valeurs atypiques · ${requestedVariableLabel}`
      : result.title;


  const requestedAnalysisTitleId =
    `requested-ai-analysis-${item.contract_id.replace(
      /[^a-zA-Z0-9_-]/g,
      "-"
    )}`;


  const kpis:
    {
      label:
        string;

      value:
        string;
    }[] = [];


  if (
    isOutlierRequest
  ) {
    if (
      count !==
      null
    ) {
      kpis.push(
        {
          label:
            "Valeurs analysées",

          value:
            formatNumber(
              count
            ),
        }
      );
    }


    if (
      outlierCountIqr !==
      null
    ) {
      kpis.push(
        {
          label:
            "Outliers détectés",

          value:
            formatNumber(
              outlierCountIqr
            ),
        }
      );
    }


    if (
      outlierRatioIqr !==
      null
    ) {
      kpis.push(
        {
          label:
            "Part des observations",

          value:
            formatPercent(
              outlierRatioIqr
            ),
        }
      );
    }


    kpis.push(
      {
        label:
          "Méthode",

        value:
          "IQR · 1,5×",
      }
    );
  }


  else if (
    count !==
    null
  ) {
    kpis.push(
      {
        label:
          "Observations",

        value:
          formatNumber(
            count
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    periodCount !==
      null
  ) {
    kpis.push(
      {
        label:
          "Périodes",

        value:
          formatNumber(
            periodCount
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    timeStart !==
      undefined &&
    timeStart !==
      null
  ) {
    kpis.push(
      {
        label:
          "Début",

        value:
          formatTemporalDisplayValue(
            timeStart
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    timeEnd !==
      undefined &&
    timeEnd !==
      null
  ) {
    kpis.push(
      {
        label:
          "Fin",

        value:
          formatTemporalDisplayValue(
            timeEnd
          ),
      }
    );
  }


  if (
    mean !==
    null
  ) {
    kpis.push(
      {
        label:
          "Moyenne",

        value:
          formatDecimal(
            mean
          ),
      }
    );
  }


  if (
    median !==
    null
  ) {
    kpis.push(
      {
        label:
          "Médiane",

        value:
          formatDecimal(
            median
          ),
      }
    );
  }


  if (
    standardDeviation !==
    null
  ) {
    kpis.push(
      {
        label:
          "Écart-type",

        value:
          formatDecimal(
            standardDeviation
          ),
      }
    );
  }


  if (
    coefficient !==
    null
  ) {
    kpis.push(
      {
        label:
          "Coefficient",

        value:
          formatDecimal(
            coefficient
          ),
      }
    );
  }


  if (
    cramersV !==
    null
  ) {
    kpis.push(
      {
        label:
          "V de Cramér",

        value:
          formatDecimal(
            cramersV
          ),
      }
    );
  }


  if (
    pValue !==
    null
  ) {
    kpis.push(
      {
        label:
          "p-value",

        value:
          formatDecimal(
            pValue
          ),
      }
    );
  }


  if (
    groupCount !==
    null
  ) {
    kpis.push(
      {
        label:
          "Groupes",

        value:
          formatNumber(
            groupCount
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    periodMedianMin !==
      null
  ) {
    kpis.push(
      {
        label:
          "Médiane min.",

        value:
          formatDecimal(
            periodMedianMin
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    periodMedianMax !==
      null
  ) {
    kpis.push(
      {
        label:
          "Médiane max.",

        value:
          formatDecimal(
            periodMedianMax
          ),
      }
    );
  }


  return (
    <section
      aria-labelledby={
        requestedAnalysisTitleId
      }
      className={
        styles.requestedAnswerCard
      }
    >
      <div
        className={
          styles.sectionHead
        }
        style={{
          marginBottom:
            "14px",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Résultat vérifié
          </span>

          <h2
            id={
              requestedAnalysisTitleId
            }
          >
            {
              requestedResultTitle
            }
          </h2>

          <p
            className={
              styles.resultSubtitle
            }
          >
            {
              familyLabel(
                result.family
              )
            }
            {" · "}
            {
              result.dataset_filename ??
              "Dataset analysé"
            }
          </p>
        </div>


        <div
          style={{
            display:
              "flex",

            gap:
              "7px",

            flexWrap:
              "wrap",

            justifyContent:
              "flex-end",
          }}
        >
          <span
            style={{
              padding:
                "6px 9px",

              border:
                "1px solid rgba(126, 177, 255, 0.18)",

              borderRadius:
                "999px",

              fontSize:
                "0.69rem",

              fontWeight:
                700,
            }}
          >
            {
              familyLabel(
                result.family
              )
            }
          </span>

          <span
            style={{
              padding:
                "6px 9px",

              border:
                "1px solid rgba(122, 203, 160, 0.25)",

              borderRadius:
                "999px",

              fontSize:
                "0.69rem",

              fontWeight:
                700,
            }}
          >
            Exécution vérifiée
          </span>
        </div>
      </div>


      {
        !isOutlierRequest &&
        Object.keys(
          variables
        ).length >
        0
          ? (
              <div
                style={{
                  display:
                    "flex",

                  gap:
                    "7px",

                  flexWrap:
                    "wrap",

                  marginBottom:
                    "12px",
                }}
              >
                {
                  Object.entries(
                    variables
                  ).map(
                    (
                      [
                        role,
                        column,
                      ]
                    ) => (
                      <span
                        key={
                          `${role}-${column}`
                        }
                        style={{
                          padding:
                            "6px 8px",

                          border:
                            "1px solid rgba(255,255,255,0.07)",

                          borderRadius:
                            "8px",

                          fontSize:
                            "0.72rem",
                        }}
                      >
                        <strong>
                          {
                            role
                          }
                        </strong>

                        {" · "}

                        {
                          friendlyVariableLabel(
                            column
                          )
                        }
                      </span>
                    )
                  )
                }
              </div>
            )
          : null
      }


      {
        !isOutlierRequest &&
        result.summary.length >
        0
          ? (
              <div
                className={
                  styles.technicalReasons
                }
                style={{
                  marginBottom:
                    "12px",
                }}
              >
                {
                  result.summary
                    .slice(
                      0,
                      3
                    )
                    .map(
                      (
                        summary
                      ) => (
                        <p
                          key={
                            summary
                          }
                        >
                          {
                            summary
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
        kpis.length >
        0
          ? (
              <div
                className={
                  styles.metricsGrid
                }
                style={{
                  marginBottom:
                    "14px",
                }}
              >
                {
                  kpis
                    .slice(
                      0,
                      4
                    )
                    .map(
                      (
                        kpi
                      ) => (
                        <article
                          className={
                            styles.metricCard
                          }
                          key={
                            kpi.label
                          }
                        >
                          <span>
                            {
                              kpi.label
                            }
                          </span>

                          <strong>
                            {
                              kpi.value
                            }
                          </strong>
                        </article>
                      )
                    )
                }
              </div>
            )
          : null
      }


      {
        isOutlierRequest
          ? (
              <section
                className={
                  styles.requestedAnswerStatement
                }
              >
                <span>
                  Conclusion
                </span>

                <strong>
                  {
                    outlierCountIqr ===
                      null
                      ? "Le calcul des valeurs atypiques est disponible dans les preuves."
                      : outlierCountIqr ===
                          0
                        ? "Aucune valeur atypique détectée avec la règle IQR 1,5×."
                        : `${formatNumber(
                            outlierCountIqr
                          )} valeur${
                            outlierCountIqr >
                            1
                              ? "s"
                              : ""
                          } atypique${
                            outlierCountIqr >
                            1
                              ? "s"
                              : ""
                          } détectée${
                            outlierCountIqr >
                            1
                              ? "s"
                              : ""
                          } sur ${requestedVariableLabel.toLowerCase()}.`
                  }
                </strong>

                <p>
                  Les valeurs sont signalées, pas supprimées.
                  DataLens conserve les données et laisse la décision
                  de traitement à l’analyste.
                </p>
              </section>
            )
          : null
      }


      {
        isOutlierRequest &&
        q1 !==
          null &&
        q3 !==
          null &&
        iqr !==
          null
          ? (
              <div
                className={
                  styles.outlierEvidenceGrid
                }
              >
                {[
                  {
                    label:
                      "Q1",

                    value:
                      formatDecimal(
                        q1
                      ),
                  },

                  {
                    label:
                      "Q3",

                    value:
                      formatDecimal(
                        q3
                      ),
                  },

                  {
                    label:
                      "IQR",

                    value:
                      formatDecimal(
                        iqr
                      ),
                  },

                  {
                    label:
                      "Borne basse",

                    value:
                      lowerIqrBound !==
                        null
                        ? formatDecimal(
                            lowerIqrBound
                          )
                        : "—",
                  },

                  {
                    label:
                      "Borne haute",

                    value:
                      upperIqrBound !==
                        null
                        ? formatDecimal(
                            upperIqrBound
                          )
                        : "—",
                  },
                ].map(
                  (
                    evidence
                  ) => (
                    <article
                      className={
                        styles.outlierEvidenceItem
                      }
                      key={
                        evidence.label
                      }
                    >
                      <span>
                        {
                          evidence.label
                        }
                      </span>

                      <strong>
                        {
                          evidence.value
                        }
                      </strong>
                    </article>
                  )
                )}
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "line_band" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <div>
                    <h3>
                      {
                        `Évolution de ${
                          friendlyVariableLabel(
                            valueColumn ??
                            "Valeur"
                          )
                        } selon ${
                          friendlyVariableLabel(
                            timeColumn ??
                            "Temps"
                          )
                        }`
                      }
                    </h3>

                    <p>
                      Médiane par période · bande
                      interquartile Q1–Q3 · calcul
                      déterministe.
                    </p>
                  </div>
                </div>

                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <LineBandChart
                    data={
                      chartData
                    }
                    yLabel={
                      friendlyVariableLabel(
                        valueColumn ??
                        "Valeur"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "histogram" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <div>
                    <h3>
                      {
                        isOutlierRequest
                          ? "Distribution et bornes IQR"
                          : "Distribution observée"
                      }
                    </h3>

                    <p>
                      {
                        isOutlierRequest
                          ? (
                              "Les zones extérieures aux bornes IQR 1,5× " +
                              "correspondent au périmètre des valeurs atypiques."
                            )
                          : "Histogramme calculé par le moteur déterministe."
                      }
                    </p>
                  </div>
                </div>

                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <NativeHistogramChart
                    data={
                      chartData
                    }
                    valueLabel={
                      requestedVariableLabel
                    }
                    lowerBound={
                      lowerIqrBound
                    }
                    upperBound={
                      upperIqrBound
                    }
                    highlightOutliers={
                      isOutlierRequest
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "scatter" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <ScatterPlot
                    data={
                      chartData
                    }
                    xLabel={
                      friendlyVariableLabel(
                        xColumn ??
                        "Variable X"
                      )
                    }
                    yLabel={
                      friendlyVariableLabel(
                        yColumn ??
                        "Variable Y"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "heatmap" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <RequestedHeatmapChart
                    data={
                      chartData
                    }
                    xLabel={
                      friendlyVariableLabel(
                        xColumn ??
                        "Variable X"
                      )
                    }
                    yLabel={
                      friendlyVariableLabel(
                        yColumn ??
                        "Variable Y"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "boxplot" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <RequestedBoxPlotChart
                    data={
                      chartData
                    }
                    groupLabel={
                      friendlyVariableLabel(
                        groupColumn ??
                        "Groupe"
                      )
                    }
                    valueLabel={
                      friendlyVariableLabel(
                        valueColumn ??
                        "Valeur"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      <details
        className={
          styles.technicalPanel
        }
        style={{
          marginTop:
            "12px",
        }}
      >
        <summary>
          Traçabilité de l’analyse demandée
        </summary>

        <div
          className={
            styles.evidenceFlow
          }
        >
          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Planner
            </span>

            <strong>
              {
                report.planner_model
              }
            </strong>

            <small>
              {
                report.planner
                  .attempt_count ??
                1
              }
              {" tentative(s)"}
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Validation
            </span>

            <strong>
              Python
            </strong>

            <small>
              Contrat analytique validé
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Tool calling
            </span>

            <strong>
              {
                nativeTool
                  .requested_tool ??
                "—"
              }
            </strong>

            <small>
              {
                report.tool_model
              }
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Exécution
            </span>

            <strong>
              Python
            </strong>

            <small>
              {
                nativeTool.attempt_count
              }
              {" tentative(s) · "}
              {
                nativeTool.retry_count
              }
              {" retry"}
            </small>
          </article>
        </div>


        {
          result.family ===
            "time_series"
            ? (
                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  <p>
                    Exécution temporelle déterministe :
                    médiane de
                    {" "}
                    <strong>
                      {
                        friendlyVariableLabel(
                          valueColumn ??
                          "Valeur"
                        )
                      }
                    </strong>
                    {" "}
                    regroupée par
                    {" "}
                    <strong>
                      {
                        friendlyVariableLabel(
                          timeColumn ??
                          "Temps"
                        )
                      }
                    </strong>
                    , avec Q1 et Q3 calculés pour chaque
                    période.
                  </p>
                </div>
              )
            : null
        }


        {
          (
            report
              .planner
              .normalization_count ??
            0
          ) >
          0
            ? (
                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  <p>
                    Python a appliqué
                    {" "}
                    {
                      report
                        .planner
                        .normalization_count
                    }
                    {" "}
                    normalisation(s) de protocole
                    avant validation.
                  </p>
                </div>
              )
            : null
        }
      </details>
    </section>
  );
}


function NativeRequestedAnalysisCard({
  report,
  objective,
}: {
  report:
    AINativePipelineReportView;

  objective:
    string;
}) {
  const executedItems =
    report.items.filter(
      (
        candidate
      ) =>
        candidate.pipeline_status ===
          "executed" &&
        candidate.native_tool
          ?.execution
          ?.result
          !==
          null
    );


  if (
    executedItems.length ===
    0
  ) {
    return (
      <NativeRequestedAnalysisSingleCard
        report={
          report
        }
        objective={
          objective
        }
      />
    );
  }


  return (
    <section
      className={
        styles.requestedAnswerGroup
      }
      aria-labelledby="requested-answer-group-title"
    >
      <div
        className={
          styles.requestedAnswerGroupHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Votre demande
          </span>

          <h2
            id="requested-answer-group-title"
          >
            Réponse à votre demande
          </h2>

          <p
            className={
              styles.requestedQuestion
            }
          >
            «
            {
              objective.trim() ||
              report.planner.objective
            }
            »
          </p>
        </div>

        <span
          className={
            styles.requestedAnswerCount
          }
        >
          {
            executedItems.length
          }
          {" résultat"}
          {
            executedItems.length >
            1
              ? "s"
              : ""
          }
          {" vérifié"}
          {
            executedItems.length >
            1
              ? "s"
              : ""
          }
        </span>
      </div>


      <div
        className={
          styles.requestedAnswerStack
        }
      >
        {
          executedItems.map(
            (
              executedItem
            ) => (
              <NativeRequestedAnalysisSingleCard
                report={{
                  ...report,
                  items: [
                    executedItem,
                  ],
                }}
                objective={
                  objective
                }
                key={
                  executedItem.contract_id
                }
              />
            )
          )
        }
      </div>
    </section>
  );
}


function findingHasDetailedVisualization(
  finding:
    ReportFinding
): boolean {
  const data =
    finding.chart_data ??
    [];


  switch (
    finding.chart_type
  ) {
    case "line":
      return (
        data.filter(
          (
            datum
          ) =>
            datumNumber(
              datum,
              "value"
            ) !==
            null
        ).length >
        1
      );


    case "line_band":
      return (
        lineBandRenderablePoints(
          data
        ).length >
        1
      );


    case "scatter":
      return (
        data.filter(
          (
            datum
          ) =>
            datumNumber(
              datum,
              "x"
            ) !==
              null &&
            datumNumber(
              datum,
              "y"
            ) !==
              null
        ).length >
        1
      );


    case "lorenz":
      return (
        data.filter(
          (
            datum
          ) =>
            datumNumber(
              datum,
              "population_share"
            ) !==
              null &&
            datumNumber(
              datum,
              "revenue_share"
            ) !==
              null
        ).length >
        1
      );


    case "bar":
    case "grouped_summary":
    case "heatmap":
    case "boxplot":
    case "histogram":
      return (
        data.length >
        0
      );


    case "distribution":
      return true;


    default:
      return false;
  }
}

function FindingCard({
  finding,
  index,
  ragContext,
}: {
  finding:
    ReportFinding;

  index:
    number;

  ragContext:
    FindingRagContext |
    null;
}) {
  return (
    <article
      className={
        styles.explanationCard
      }
    >
      <div
        className={
          styles.chartHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Analyse
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

          <h3
            className={
              styles.explanationTitle
            }
          >
            {
              finding.title
            }
          </h3>

          <p
            className={
              styles.resultSubtitle
            }
          >
            {
              familyLabel(
                finding.family
              )
            }

            {
              finding.datasets.length >
              0
                ? (
                    <>
                      {" · "}
                      {
                        finding
                          .datasets
                          .join(
                            " · "
                          )
                      }
                    </>
                  )
                : null
            }
          </p>
        </div>
      </div>


      {
        finding.summary.length >
        0
          ? (
              <div
                className={
                  styles.technicalReasons
                }
              >
                {
                  finding
                    .summary
                    .slice(
                      0,
                      3
                    )
                    .map(
                      (
                        item
                      ) => (
                        <p
                          key={
                            item
                          }
                        >
                          {
                            item
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
        findingHasDetailedVisualization(
          finding
        )
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    finding.title
                  }
                >
                  <FindingChart
                    finding={
                      finding
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      <RagContextBlock
        context={
          ragContext
        }
      />
    </article>
  );
}


function CompactFindingList({
  title,
  findings,
}: {
  title:
    string;

  findings:
    ReportFinding[];
}) {
  if (
    findings.length ===
    0
  ) {
    return null;
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        {
          title
        }
        {" · "}
        {
          findings.length
        }
      </summary>


      <div
        className={
          styles.explanationGrid
        }
      >
        {
          findings.map(
            (
              finding,
              findingIndex
            ) => (
              <article
                className={
                  styles.explanationCard
                }
                key={
                  `${
                    finding.analysis_id ??
                    `${finding.family}-${finding.title}`
                  }-${findingIndex}`
                }
              >
                <span
                  className={
                    styles.eyebrow
                  }
                >
                  {
                    familyLabel(
                      finding.family
                    )
                  }
                </span>

                <h3
                  className={
                    styles.explanationTitle
                  }
                >
                  {
                    finding.title
                  }
                </h3>

                {
                  finding
                    .summary
                    .slice(
                      0,
                      2
                    )
                    .map(
                      (
                        item,
                        summaryIndex
                      ) => (
                        <p
                          className={
                            styles.explanationText
                          }
                          key={
                            `${summaryIndex}-${item}`
                          }
                        >
                          {
                            item
                          }
                        </p>
                      )
                    )
                }
              </article>
            )
          )
        }
      </div>
    </details>
  );
}


function QualityList({
  items,
}: {
  items:
    ReportQualityItem[];
}) {
  if (
    items.length ===
    0
  ) {
    return null;
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        Qualité des données
        {" · "}
        {
          items.length
        }
      </summary>


      <div
        className={
          styles.explanationGrid
        }
      >
        {
          items.map(
            (
              item
            ) => (
              <article
                className={
                  styles.explanationCard
                }
                key={
                  item.analysis_id
                }
              >
                <span
                  className={
                    styles.eyebrow
                  }
                >
                  Contrôle qualité
                </span>

                <h3
                  className={
                    styles.explanationTitle
                  }
                >
                  {
                    item.dataset
                  }
                </h3>


                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  <p>
                    {
                      formatNumber(
                        item.row_count
                      )
                    } lignes
                    {" · "}
                    {
                      formatNumber(
                        item.column_count
                      )
                    } colonnes
                  </p>

                  <p>
                    Valeurs manquantes :
                    {" "}
                    {
                      formatNumber(
                        item.missing_cells
                      )
                    }
                    {" · "}
                    {
                      formatPercent(
                        item.missing_ratio
                      )
                    }
                  </p>

                  <p>
                    Doublons stricts :
                    {" "}
                    {
                      formatNumber(
                        item.duplicate_rows
                      )
                    }
                    {" · "}
                    {
                      formatPercent(
                        item.duplicate_ratio
                      )
                    }
                  </p>


                  {
                    item
                      .completely_missing_columns
                      .length >
                    0
                      ? (
                          <p>
                            Colonnes entièrement
                            manquantes :
                            {" "}
                            {
                              item
                                .completely_missing_columns
                                .join(
                                  ", "
                                )
                            }
                          </p>
                        )
                      : null
                  }


                  {
                    item
                      .constant_columns
                      .length >
                    0
                      ? (
                          <p>
                            Colonnes constantes :
                            {" "}
                            {
                              item
                                .constant_columns
                                .join(
                                  ", "
                                )
                            }
                          </p>
                        )
                      : null
                  }
                </div>


                {
                  item
                    .summary
                    .map(
                      (
                        line
                      ) => (
                        <p
                          className={
                            styles.explanationText
                          }
                          key={
                            line
                          }
                        >
                          {
                            line
                          }
                        </p>
                      )
                    )
                }
              </article>
            )
          )
        }
      </div>
    </details>
  );
}


function BlockedAnalysisList({
  items,
}: {
  items:
    ReportBlockedAnalysis[];
}) {
  if (
    items.length ===
    0
  ) {
    return null;
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        Analyses non exécutées
        {" · "}
        {
          items.length
        }
      </summary>


      <div
        className={
          styles.explanationGrid
        }
      >
        {
          items.map(
            (
              item
            ) => (
              <article
                className={
                  styles.explanationCard
                }
                key={
                  item.analysis_id
                }
              >
                <span
                  className={
                    styles.eyebrow
                  }
                >
                  {
                    familyLabel(
                      item.family
                    )
                  }
                </span>

                <h3
                  className={
                    styles.explanationTitle
                  }
                >
                  {
                    item.title
                  }
                </h3>


                <p
                  className={
                    styles.resultSubtitle
                  }
                >
                  {
                    item
                      .datasets
                      .join(
                        " · "
                      )
                  }
                </p>


                <p
                  className={
                    styles.explanationText
                  }
                >
                  {
                    item.reason
                  }
                </p>


                {
                  item
                    .caveats
                    .length >
                  0
                    ? (
                        <div
                          className={
                            styles.technicalReasons
                          }
                        >
                          {
                            item
                              .caveats
                              .map(
                                (
                                  caveat
                                ) => (
                                  <p
                                    key={
                                      caveat
                                    }
                                  >
                                    {
                                      caveat
                                    }
                                  </p>
                                )
                              )
                          }
                        </div>
                      )
                    : null
                }


                <p
                  className={
                    styles.explanationText
                  }
                >
                  Priorité de découverte :
                  {" "}
                  {
                    formatDecimal(
                      item.discovery_priority_score
                    )
                  }
                </p>
              </article>
            )
          )
        }
      </div>
    </details>
  );
}


function requestPlanningLabel(
  status:
    RequestedPlanItemView[
      "status"
    ]
): string {
  switch (
    status
  ) {
    case "ready":
      return "Prête";

    case "blocked":
      return "Bloquée";

    case "ambiguous":
      return "À clarifier";

    default:
      return status;
  }
}



function requestResolutionGuidance(
  request:
    RequestedPlanItemView
): {
  title:
    string;

  action:
    string;

  protection:
    string;
} {
  if (
    request.kind ===
    "b2b_revenue_share"
  ) {
    return {
      title:
        "Identifier explicitement les clients BtoB",

      action:
        (
          "Ajoutez ou indiquez une colonne qui décrit explicitement " +
          "le type de client, par exemple segment, customer_type, " +
          "account_type ou b2b_flag."
        ),

      protection:
        (
          "DataLens ne déduira pas qu’un client est BtoB à partir " +
          "d’un chiffre d’affaires élevé, d’un panier atypique ou " +
          "d’une fréquence d’achat importante."
        ),
    };
  }


  if (
    request.status ===
    "ambiguous"
  ) {
    return {
      title:
        "Préciser la règle d’analyse",

      action:
        (
          "Précisez la métrique, le périmètre ou la règle attendue " +
          "afin que DataLens puisse construire un plan déterministe."
        ),

      protection:
        (
          "DataLens préfère demander une clarification plutôt que " +
          "choisir arbitrairement une définition qui pourrait changer " +
          "le résultat."
        ),
    };
  }


  return {
    title:
      "Fournir l’information manquante",

    action:
      (
        "Ajoutez la variable explicitement requise dans les données " +
        "ou complétez la documentation afin que la demande puisse " +
        "être résolue sans hypothèse cachée."
      ),

    protection:
      (
        "Aucune substitution approximative n’est exécutée lorsque " +
        "la preuve nécessaire à l’analyse manque."
      ),
  };
}


function RequestResolutionPanel({
  plan,
}: {
  plan:
    RequestedPlanView |
    null;
}) {
  const unresolved =
    plan?.requests.filter(
      (
        request
      ) =>
        request.status !==
        "ready"
    ) ??
    [];


  if (
    unresolved.length ===
    0
  ) {
    return null;
  }


  return (
    <section
      className={
        styles.summaryPanel
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
            Action requise
          </span>

          <h3>
            {
              unresolved.length ===
                1
                ? "Une demande ne peut pas encore être exécutée"
                : `${unresolved.length} demandes nécessitent votre intervention`
            }
          </h3>

          <p
            className={
              styles.resultSubtitle
            }
          >
            DataLens s’arrête lorsqu’une information
            indispensable manque ou lorsqu’une règle
            reste ambiguë. Le moteur n’invente pas
            la donnée manquante.
          </p>
        </div>
      </div>


      <div
        className={
          styles.explanationGrid
        }
      >
        {
          unresolved.map(
            (
              request
            ) => {
              const guidance =
                requestResolutionGuidance(
                  request
                );


              return (
                <article
                  className={
                    styles.explanationCard
                  }
                  key={
                    request.request_id
                  }
                >
                  <span
                    className={
                      styles.eyebrow
                    }
                  >
                    {
                      request.status ===
                        "blocked"
                        ? "Bloquée"
                        : "À clarifier"
                    }
                  </span>

                  <h3
                    className={
                      styles.explanationTitle
                    }
                  >
                    {
                      request.request_text
                    }
                  </h3>


                  <p
                    className={
                      styles.resultSubtitle
                    }
                  >
                    {
                      request.source_filename
                    }
                    {" · "}
                    {
                      request.source_locator
                    }
                  </p>


                  {
                    request.blockers.length >
                    0
                      ? (
                          <div
                            className={
                              styles.technicalReasons
                            }
                          >
                            <strong>
                              Pourquoi DataLens s’arrête ici
                            </strong>

                            {
                              request.blockers.map(
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


                  <div
                    className={
                      styles.technicalReasons
                    }
                  >
                    <strong>
                      {
                        guidance.title
                      }
                    </strong>

                    <p>
                      {
                        guidance.action
                      }
                    </p>
                  </div>


                  <p
                    className={
                      styles.explanationText
                    }
                  >
                    {
                      guidance.protection
                    }
                  </p>
                </article>
              );
            }
          )
        }
      </div>
    </section>
  );
}


function DocumentRequestsSummary({
  summary,
  plan,
}: {
  summary:
    DocumentSummaryView |
    null;

  plan:
    RequestedPlanView |
    null;
}) {
  if (
    !summary ||
    summary.status !==
      "ready"
  ) {
    return null;
  }


  const summaryPoints =
    summary.summary_points ??
    [];

  const documents =
    summary.documents ??
    [];

  const requestCount =
    plan?.request_count ??
    summary.analytical_request_count;

  const readyCount =
    plan?.ready_count ??
    0;

  const blockedCount =
    plan?.blocked_count ??
    0;

  const ambiguousCount =
    plan?.ambiguous_count ??
    0;

  const clarificationCount =
    blockedCount +
    ambiguousCount;


  return (
    <>
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
            Documentation métier
          </span>

          <h2>
            Ce que demandent
            vos documents
          </h2>

          <p
            className={
              styles.resultSubtitle
            }
          >
            DataLens distingue le cadrage
            documentaire des résultats calculés.
            Une demande détectée n’est pas
            considérée comme exécutée tant que
            le moteur Python ne l’a pas réellement
            analysée.
          </p>
        </div>
      </div>


      <div
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
            Documents
          </span>

          <strong>
            {
              summary.document_count
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Demandes détectées
          </span>

          <strong>
            {
              requestCount
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Prêtes à analyser
          </span>

          <strong
            className={
              styles.statusGood
            }
          >
            {
              readyCount
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Intervention requise
          </span>

          <strong
            className={
              clarificationCount >
              0
                ? styles.statusNeutral
                : styles.statusNeutral
            }
          >
            {
              clarificationCount
            }
          </strong>
        </article>
      </div>


      <RequestResolutionPanel
        plan={
          plan
        }
      />


      {
        summaryPoints.length >
        0
          ? (
              <section
                className={
                  styles.summaryPanel
                }
              >
                <div
                  className={
                    styles.summaryItem
                  }
                >
                  <span>
                    Points de cadrage
                  </span>

                  {
                    summaryPoints
                      .slice(
                        0,
                        5
                      )
                      .map(
                        (
                          point
                        ) => (
                          <p
                            key={
                              `${
                                point.citation.chunk_id
                              }-${
                                point.evidence_unit_id
                              }`
                            }
                          >
                            {
                              point.statement
                            }
                          </p>
                        )
                      )
                  }
                </div>
              </section>
            )
          : null
      }


      {
        documents.length >
        0
          ? (
              <div
                className={
                  styles.explanationGrid
                }
              >
                {
                  documents.map(
                    (
                      document
                    ) => {
                      const visiblePoints =
                        document
                          .summary_points
                          .length >
                        0
                          ? document
                              .summary_points
                              .slice(
                                0,
                                2
                              )
                          : document
                              .analytical_requests
                              .slice(
                                0,
                                2
                              );


                      return (
                        <article
                          className={
                            styles.explanationCard
                          }
                          key={
                            document.document_id
                          }
                        >
                          <span
                            className={
                              styles.eyebrow
                            }
                          >
                            Document vérifié
                          </span>

                          <h3
                            className={
                              styles.explanationTitle
                            }
                          >
                            {
                              document.filename
                            }
                          </h3>

                          <p
                            className={
                              styles.resultSubtitle
                            }
                          >
                            {
                              document
                                .analytical_requests
                                .length
                            }
                            {" "}
                            demande
                            {
                              document
                                .analytical_requests
                                .length >
                              1
                                ? "s"
                                : ""
                            }
                            {" · "}
                            {
                              document
                                .verified_claim_count
                            }
                            {" "}
                            élément
                            {
                              document
                                .verified_claim_count >
                              1
                                ? "s"
                                : ""
                            }
                            {" "}
                            vérifié
                            {
                              document
                                .verified_claim_count >
                              1
                                ? "s"
                                : ""
                            }
                          </p>


                          {
                            visiblePoints.map(
                              (
                                point
                              ) => (
                                <p
                                  className={
                                    styles.explanationText
                                  }
                                  key={
                                    `${
                                      document.document_id
                                    }-${
                                      point.evidence_unit_id
                                    }`
                                  }
                                >
                                  {
                                    point.statement
                                  }
                                </p>
                              )
                            )
                          }
                        </article>
                      );
                    }
                  )
                }
              </div>
            )
          : null
      }


      {
        plan &&
        plan.requests.length >
        0
          ? (
              <details
                className={
                  styles.technicalPanel
                }
              >
                <summary>
                  Voir les
                  {" "}
                  {
                    plan.request_count
                  }
                  {" "}
                  demandes détectées
                </summary>

                <div
                  className={
                    styles.explanationGrid
                  }
                >
                  {
                    plan.requests.map(
                      (
                        request,
                        index
                      ) => (
                        <article
                          className={
                            styles.explanationCard
                          }
                          key={
                            request.request_id
                          }
                        >
                          <span
                            className={
                              styles.eyebrow
                            }
                          >
                            Demande
                            {" "}
                            {
                              String(
                                index + 1
                              ).padStart(
                                2,
                                "0"
                              )
                            }
                            {" · "}
                            {
                              requestPlanningLabel(
                                request.status
                              )
                            }
                          </span>

                          <h3
                            className={
                              styles.explanationTitle
                            }
                          >
                            {
                              request.request_text
                            }
                          </h3>

                          <p
                            className={
                              styles.resultSubtitle
                            }
                          >
                            {
                              request.source_filename
                            }
                            {" · "}
                            {
                              request.source_locator
                            }
                          </p>


                          {
                            request.blockers.length >
                            0
                              ? (
                                  <div
                                    className={
                                      styles.technicalReasons
                                    }
                                  >
                                    {
                                      request.blockers.map(
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
                        </article>
                      )
                    )
                  }
                </div>
              </details>
            )
          : null
      }
    </>
  );
}


function RagReportSummary({
  rag,
}: {
  rag:
    RagContextReport;
}) {
  return (
    <>
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
            Documentation locale
          </span>

          <h2>
            Contexte documentaire
          </h2>
        </div>
      </div>


      <div
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
            Documents
          </span>

          <strong>
            {
              rag.document_count
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Passages acceptés
          </span>

          <strong>
            {
              rag.accepted_hit_count
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Findings contextualisés
          </span>

          <strong>
            {
              rag.accepted_finding_count
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Explications vérifiées
          </span>

          <strong>
            {
              rag.explanation_ready_count
            }
          </strong>
        </article>
      </div>


      <details
        className={
          styles.technicalPanel
        }
      >
        <summary>
          Traçabilité RAG
        </summary>

        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            Chunks indexés :
            {" "}
            {
              rag.chunk_count
            }
          </p>

          <p>
            Candidats validés :
            {" "}
            {
              rag.validated_candidate_count
            }
          </p>

          <p>
            Findings sans contexte :
            {" "}
            {
              rag.abstained_finding_count
            }
          </p>

          <p>
            Explications en abstention :
            {" "}
            {
              rag.explanation_abstained_count
            }
          </p>

          <p>
            Erreurs d’explication :
            {" "}
            {
              rag.explanation_error_count
            }
          </p>

          <p>
            Relevance gate :
            {" "}
            {
              rag.relevance_rule_version
            }
          </p>

          <p>
            Explication :
            {" "}
            {
              rag.explanation_rule_version
            }
          </p>

          <p>
            Contextualisation :
            {" "}
            {
              rag.context_rule_version
            }
          </p>
        </div>
      </details>
    </>
  );
}



type PreparationIssueSeverity =
  | "important"
  | "moderate"
  | "minor";


type QualityIssueEvidenceView = {
  observed_count:
    number;

  affected_ratio:
    number;

  examples:
    string[];

  details:
    Record<
      string,
      unknown
    >;
};


type CleaningProposalView = {
  operation:
    string;

  automatic_safe:
    boolean;

  description:
    string;

  requires_user_confirmation:
    boolean;

  parameters:
    Record<
      string,
      unknown
    >;
};


type DataQualityIssueView = {
  issue_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  column:
    string |
    null;

  kind:
    string;

  severity:
    PreparationIssueSeverity;

  title:
    string;

  explanation:
    string;

  evidence:
    QualityIssueEvidenceView;

  proposal:
    CleaningProposalView;

  semantic_review_recommended:
    boolean;
};


type DatasetQualitySummaryView = {
  dataset_id:
    string;

  dataset_filename:
    string;

  row_count:
    number;

  column_count:
    number;

  missing_cell_count:
    number;

  missing_cell_ratio:
    number;

  duplicate_row_count:
    number;

  issue_count:
    number;

  important_count:
    number;

  moderate_count:
    number;

  minor_count:
    number;
};


type DataQualityReportView = {
  status:
    string;

  dataset_count:
    number;

  total_rows:
    number;

  total_columns:
    number;

  issue_count:
    number;

  important_count:
    number;

  moderate_count:
    number;

  minor_count:
    number;

  semantic_review_count:
    number;

  datasets:
    DatasetQualitySummaryView[];

  issues:
    DataQualityIssueView[];

  notes:
    string[];

  rule_version:
    string;
};


type CleaningActionView = {
  action_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  kind:
    string;

  column:
    string |
    null;

  title:
    string;

  rationale:
    string;

  safe_candidate:
    boolean;

  requires_user_confirmation:
    boolean;

  affected_rows_estimate:
    number;

  before_examples:
    string[];

  after_examples:
    string[];

  parameters:
    Record<
      string,
      unknown
    >;
};


type CleaningPlanView = {
  status:
    string;

  dataset_count:
    number;

  action_count:
    number;

  safe_candidate_count:
    number;

  confirmation_required_count:
    number;

  protected_issue_count:
    number;

  actions:
    CleaningActionView[];

  notes:
    string[];

  rule_version:
    string;
};


type CleaningActionResultView = {
  action_id:
    string;

  status:
    string;

  affected_rows_actual:
    number;

  rows_before:
    number;

  rows_after:
    number;

  details:
    Record<
      string,
      unknown
    >;
};


type DatasetCleaningProvenanceView = {
  dataset_id:
    string;

  dataset_filename:
    string;

  rows_before:
    number;

  rows_after:
    number;

  columns_before:
    number;

  columns_after:
    number;

  source_fingerprint:
    string;

  derived_fingerprint:
    string;

  applied_action_ids:
    string[];

  skipped_action_ids:
    string[];
};


type CleaningExecutionView = {
  status:
    string;

  dataset_count:
    number;

  applied_action_count:
    number;

  skipped_action_count:
    number;

  blocked_action_count:
    number;

  action_results:
    CleaningActionResultView[];

  provenance:
    DatasetCleaningProvenanceView[];

  notes:
    string[];

  rule_version:
    string;
};


type CleaningApplyResponseView = {
  status:
    string;

  quality_report:
    DataQualityReportView;

  cleaning_plan:
    CleaningPlanView;

  execution:
    CleaningExecutionView;

  derived_datasets:
    Array<{
      dataset_id:
        string;

      dataset_filename:
        string;

      rows_before:
        number;

      rows_after:
        number;

      columns_before:
        number;

      columns_after:
        number;

      preview_rows:
        Array<
          Record<
            string,
            unknown
          >
        >;
    }>;

  notes:
    string[];
};



type SemanticVerdictView =
  | "merge_values"
  | "keep_separate"
  | "flag_for_review"
  | "contextualize"
  | "no_change"
  | "abstain";


type SemanticDecisionView = {
  issue_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  column:
    string |
    null;

  kind:
    string;

  verdict:
    SemanticVerdictView;

  confidence:
    number;

  rationale:
    string;

  source_values:
    string[];

  canonical_value:
    string |
    null;

  user_message:
    string;

  python_validated:
    boolean;

  executable:
    boolean;

  requires_user_confirmation:
    boolean;

  validation_notes:
    string[];
};


type SemanticReviewReportView = {
  status:
    string;

  model:
    string;

  candidate_count:
    number;

  decision_count:
    number;

  merge_proposal_count:
    number;

  abstention_count:
    number;

  decisions:
    SemanticDecisionView[];

  notes:
    string[];

  rule_version:
    string;
};


type SemanticCleaningActionView = {
  action_id:
    string;

  issue_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  column:
    string;

  source_values:
    string[];

  suggested_canonical_value:
    string;

  allowed_canonical_values:
    string[];

  confidence:
    number;

  rationale:
    string;

  requires_user_confirmation:
    boolean;

  python_validated:
    boolean;
};


type SemanticCleaningPlanView = {
  status:
    string;

  action_count:
    number;

  actions:
    SemanticCleaningActionView[];

  notes:
    string[];

  rule_version:
    string;
};


type SemanticCleaningChoiceView = {
  action_id:
    string;

  canonical_value:
    string;
};


type SemanticCleaningActionResultView = {
  action_id:
    string;

  status:
    "applied" |
    "skipped";

  dataset_id:
    string;

  column:
    string;

  source_values:
    string[];

  canonical_value:
    string |
    null;

  affected_rows_actual:
    number;

  details:
    Record<
      string,
      unknown
    >;
};


type SemanticDatasetProvenanceView = {
  dataset_id:
    string;

  dataset_filename:
    string;

  rows_before:
    number;

  rows_after:
    number;

  source_fingerprint:
    string;

  derived_fingerprint:
    string;

  applied_action_ids:
    string[];

  changed_cell_count:
    number;
};


type SemanticCleaningExecutionView = {
  status:
    string;

  dataset_count:
    number;

  applied_action_count:
    number;

  skipped_action_count:
    number;

  changed_cell_count:
    number;

  action_results:
    SemanticCleaningActionResultView[];

  provenance:
    SemanticDatasetProvenanceView[];

  notes:
    string[];

  rule_version:
    string;
};


type SemanticCleaningApplyResponseView = {
  plan:
    SemanticCleaningPlanView;

  execution:
    SemanticCleaningExecutionView;
};


function preparationSeverityLabel(
  severity:
    PreparationIssueSeverity
): string {
  switch (
    severity
  ) {
    case "important":
      return "Important";

    case "moderate":
      return "Modéré";

    case "minor":
      return "Mineur";

    default:
      return "À examiner";
  }
}


function preparationSeverityBorder(
  severity:
    PreparationIssueSeverity
): string {
  switch (
    severity
  ) {
    case "important":
      return "rgba(255, 142, 117, 0.24)";

    case "moderate":
      return "rgba(255, 187, 112, 0.22)";

    case "minor":
      return "rgba(126, 177, 255, 0.16)";

    default:
      return "rgba(255,255,255,0.08)";
  }
}


function preparationQualityLabel(
  report:
    DataQualityReportView
): string {
  if (
    report.important_count >
    0
  ) {
    return "Attention requise";
  }


  if (
    report.moderate_count >
    0
  ) {
    return "À contrôler";
  }


  return "Satisfaisante";
}


function DataPreparationStudio({
  ingestion,
  qualityReport,
  qualityLoading,
  qualityError,
}: {
  ingestion:
    MultiDatasetIngestion |
    null;

  qualityReport:
    DataQualityReportView |
    null;

  qualityLoading:
    boolean;

  qualityError:
    string |
    null;
}) {
  if (
    !ingestion
  ) {
    return null;
  }


  const qualityLabel =
    qualityReport
      ? preparationQualityLabel(
          qualityReport
        )
      : (
          qualityLoading
            ? "Diagnostic en cours"
            : "Diagnostic indisponible"
        );


  const safeProposalCount =
    qualityReport?.issues.filter(
      (
        issue
      ) =>
        issue.proposal
          .automatic_safe
    ).length ??
    0;


  const confirmationCount =
    qualityReport?.issues.filter(
      (
        issue
      ) =>
        issue.proposal
          .requires_user_confirmation
    ).length ??
    0;


  const missingCells =
    qualityReport?.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        dataset.missing_cell_count,
      0
    ) ??
    0;


  const totalCells =
    qualityReport?.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        (
          dataset.row_count *
          dataset.column_count
        ),
      0
    ) ??
    0;


  const missingRatio =
    totalCells >
      0
      ? missingCells /
        totalCells
      : 0;


  const duplicateRows =
    qualityReport?.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        dataset.duplicate_row_count,
      0
    ) ??
    0;


  return (
    <section
      style={{
        marginTop:
          "18px",

        padding:
          "18px",

        border:
          "1px solid rgba(126, 177, 255, 0.12)",

        borderRadius:
          "16px",

        background:
          "linear-gradient(180deg, rgba(126,177,255,0.032), rgba(255,255,255,0.012))",
      }}
    >
      <div
        style={{
          display:
            "flex",

          justifyContent:
            "space-between",

          alignItems:
            "flex-start",

          gap:
            "18px",

          flexWrap:
            "wrap",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Préparation automatique
          </span>

          <h3
            style={{
              margin:
                "7px 0 0",

              fontSize:
                "1.02rem",
            }}
          >
            Qualité et nettoyage des données
          </h3>

          <p
            style={{
              margin:
                "7px 0 0",

              maxWidth:
                "800px",

              opacity:
                0.66,

              fontSize:
                "0.79rem",

              lineHeight:
                1.58,
            }}
          >
            Python inspecte réellement les fichiers
            chargés et retourne les anomalies avec
            leurs preuves. DataLens ne modifie aucune
            valeur à cette étape.
          </p>
        </div>

        <span
          style={{
            padding:
              "6px 9px",

            border:
              qualityReport?.important_count
                ? "1px solid rgba(255,142,117,0.24)"
                : (
                    qualityReport?.moderate_count
                      ? "1px solid rgba(255,187,112,0.22)"
                      : "1px solid rgba(122,203,160,0.20)"
                  ),

            borderRadius:
              "999px",

            fontSize:
              "0.67rem",

            fontWeight:
              700,
          }}
        >
          {
            qualityLabel
          }
        </span>
      </div>


      {
        qualityLoading
          ? (
              <div
                style={{
                  marginTop:
                    "16px",

                  padding:
                    "14px",

                  border:
                    "1px solid rgba(126,177,255,0.12)",

                  borderRadius:
                    "10px",

                  fontSize:
                    "0.7rem",

                  lineHeight:
                    1.5,

                  opacity:
                    0.7,
                }}
              >
                Diagnostic déterministe en cours…
              </div>
            )
          : null
      }


      {
        qualityError
          ? (
              <div
                style={{
                  marginTop:
                    "16px",

                  padding:
                    "14px",

                  border:
                    "1px solid rgba(255,142,117,0.18)",

                  borderRadius:
                    "10px",

                  fontSize:
                    "0.7rem",

                  lineHeight:
                    1.5,
                }}
              >
                <strong>
                  Diagnostic qualité indisponible.
                </strong>
                {" "}
                {
                  qualityError
                }
              </div>
            )
          : null
      }


      {
        qualityReport
          ? (
              <>
                <div
                  style={{
                    display:
                      "grid",

                    gridTemplateColumns:
                      "repeat(auto-fit, minmax(150px, 1fr))",

                    gap:
                      "8px",

                    marginTop:
                      "16px",
                  }}
                >
                  <article
                    style={{
                      padding:
                        "11px",

                      border:
                        "1px solid rgba(255,255,255,0.055)",

                      borderRadius:
                        "10px",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        opacity:
                          0.46,

                        fontSize:
                          "0.59rem",
                      }}
                    >
                      Datasets analysés
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "5px",

                        fontSize:
                          "0.8rem",
                      }}
                    >
                      {
                        qualityReport.dataset_count
                      }
                    </strong>
                  </article>


                  <article
                    style={{
                      padding:
                        "11px",

                      border:
                        "1px solid rgba(255,255,255,0.055)",

                      borderRadius:
                        "10px",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        opacity:
                          0.46,

                        fontSize:
                          "0.59rem",
                      }}
                    >
                      Lignes
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "5px",

                        fontSize:
                          "0.8rem",
                      }}
                    >
                      {
                        formatNumber(
                          qualityReport.total_rows
                        )
                      }
                    </strong>
                  </article>


                  <article
                    style={{
                      padding:
                        "11px",

                      border:
                        "1px solid rgba(255,255,255,0.055)",

                      borderRadius:
                        "10px",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        opacity:
                          0.46,

                        fontSize:
                          "0.59rem",
                      }}
                    >
                      Colonnes
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "5px",

                        fontSize:
                          "0.8rem",
                      }}
                    >
                      {
                        formatNumber(
                          qualityReport.total_columns
                        )
                      }
                    </strong>
                  </article>


                  <article
                    style={{
                      padding:
                        "11px",

                      border:
                        qualityReport.issue_count >
                          0
                          ? "1px solid rgba(255,187,112,0.18)"
                          : "1px solid rgba(122,203,160,0.16)",

                      borderRadius:
                        "10px",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        opacity:
                          0.46,

                        fontSize:
                          "0.59rem",
                      }}
                    >
                      Problèmes détectés
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "5px",

                        fontSize:
                          "0.8rem",
                      }}
                    >
                      {
                        formatNumber(
                          qualityReport.issue_count
                        )
                      }
                    </strong>
                  </article>
                </div>


                <div
                  style={{
                    display:
                      "grid",

                    gridTemplateColumns:
                      "repeat(3, minmax(0, 1fr))",

                    gap:
                      "10px",

                    marginTop:
                      "12px",
                  }}
                >
                  <article
                    style={{
                      padding:
                        "14px",

                      border:
                        "1px solid rgba(255,255,255,0.06)",

                      borderRadius:
                        "11px",

                      background:
                        "rgba(255,255,255,0.012)",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        fontSize:
                          "0.59rem",

                        letterSpacing:
                          "0.07em",

                        textTransform:
                          "uppercase",

                        opacity:
                          0.45,
                      }}
                    >
                      Qualité
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "7px",

                        fontSize:
                          "0.84rem",
                      }}
                    >
                      {
                        qualityReport.issue_count ===
                          0
                          ? "Aucun signal critique"
                          : `${qualityReport.issue_count} problème(s) détecté(s)`
                      }
                    </strong>

                    <p
                      style={{
                        margin:
                          "6px 0 0",

                        fontSize:
                          "0.67rem",

                        lineHeight:
                          1.5,

                        opacity:
                          0.56,
                      }}
                    >
                      {
                        `${qualityReport.important_count} important · ` +
                        `${qualityReport.moderate_count} modéré · ` +
                        `${qualityReport.minor_count} mineur`
                      }
                    </p>
                  </article>


                  <article
                    style={{
                      padding:
                        "14px",

                      border:
                        "1px solid rgba(255,255,255,0.06)",

                      borderRadius:
                        "11px",

                      background:
                        "rgba(255,255,255,0.012)",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        fontSize:
                          "0.59rem",

                        letterSpacing:
                          "0.07em",

                        textTransform:
                          "uppercase",

                        opacity:
                          0.45,
                      }}
                    >
                      Nettoyage
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "7px",

                        fontSize:
                          "0.84rem",
                      }}
                    >
                      {
                        `${safeProposalCount} correction(s) déterministe(s)`
                      }
                    </strong>

                    <p
                      style={{
                        margin:
                          "6px 0 0",

                        fontSize:
                          "0.67rem",

                        lineHeight:
                          1.5,

                        opacity:
                          0.56,
                      }}
                    >
                      {
                        `${confirmationCount} proposition(s) demandent une validation avant exécution.`
                      }
                    </p>
                  </article>


                  <article
                    style={{
                      padding:
                        "14px",

                      border:
                        "1px solid rgba(255,255,255,0.06)",

                      borderRadius:
                        "11px",

                      background:
                        "rgba(255,255,255,0.012)",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        fontSize:
                          "0.59rem",

                        letterSpacing:
                          "0.07em",

                        textTransform:
                          "uppercase",

                        opacity:
                          0.45,
                      }}
                    >
                      Lecture sémantique
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "7px",

                        fontSize:
                          "0.84rem",
                      }}
                    >
                      {
                        `${qualityReport.semantic_review_count} signal(s) candidat(s)`
                      }
                    </strong>

                    <p
                      style={{
                        margin:
                          "6px 0 0",

                        fontSize:
                          "0.67rem",

                        lineHeight:
                          1.5,

                        opacity:
                          0.56,
                      }}
                    >
                      Gemma n’est pas encore appelé :
                      ces cas sont seulement préparés
                      pour la prochaine couche.
                    </p>
                  </article>
                </div>


                <div
                  style={{
                    display:
                      "grid",

                    gridTemplateColumns:
                      "repeat(auto-fit, minmax(210px, 1fr))",

                    gap:
                      "8px",

                    marginTop:
                      "12px",
                  }}
                >
                  <article
                    style={{
                      padding:
                        "11px",

                      border:
                        "1px solid rgba(255,255,255,0.055)",

                      borderRadius:
                        "10px",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        opacity:
                          0.46,

                        fontSize:
                          "0.59rem",
                      }}
                    >
                      Cellules manquantes
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "5px",

                        fontSize:
                          "0.76rem",
                      }}
                    >
                      {
                        formatNumber(
                          missingCells
                        )
                      }
                      {" · "}
                      {
                        formatPercent(
                          missingRatio
                        )
                      }
                    </strong>
                  </article>


                  <article
                    style={{
                      padding:
                        "11px",

                      border:
                        duplicateRows >
                          0
                          ? "1px solid rgba(255,187,112,0.18)"
                          : "1px solid rgba(122,203,160,0.14)",

                      borderRadius:
                        "10px",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        opacity:
                          0.46,

                        fontSize:
                          "0.59rem",
                      }}
                    >
                      Doublons stricts
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "5px",

                        fontSize:
                          "0.76rem",
                      }}
                    >
                      {
                        formatNumber(
                          duplicateRows
                        )
                      }
                    </strong>
                  </article>


                  <article
                    style={{
                      padding:
                        "11px",

                      border:
                        "1px solid rgba(255,255,255,0.055)",

                      borderRadius:
                        "10px",
                    }}
                  >
                    <span
                      style={{
                        display:
                          "block",

                        opacity:
                          0.46,

                        fontSize:
                          "0.59rem",
                      }}
                    >
                      Moteur qualité
                    </span>

                    <strong
                      style={{
                        display:
                          "block",

                        marginTop:
                          "5px",

                        fontSize:
                          "0.76rem",
                      }}
                    >
                      {
                        qualityReport.rule_version
                      }
                    </strong>
                  </article>
                </div>


                {
                  qualityReport.issues.length >
                  0
                    ? (
                        <div
                          style={{
                            marginTop:
                              "16px",
                          }}
                        >
                          <span
                            style={{
                              display:
                                "block",

                              marginBottom:
                                "9px",

                              fontSize:
                                "0.61rem",

                              letterSpacing:
                                "0.08em",

                              textTransform:
                                "uppercase",

                              opacity:
                                0.46,
                            }}
                          >
                            Ce qui nécessite votre attention
                          </span>

                          <div
                            style={{
                              display:
                                "grid",

                              gap:
                                "8px",
                            }}
                          >
                            {
                              qualityReport.issues
                                .slice(
                                  0,
                                  12
                                )
                                .map(
                                  (
                                    issue
                                  ) => (
                                    <article
                                      key={
                                        issue.issue_id
                                      }
                                      style={{
                                        padding:
                                          "12px",

                                        border:
                                          `1px solid ${preparationSeverityBorder(
                                            issue.severity
                                          )}`,

                                        borderRadius:
                                          "10px",

                                        background:
                                          "rgba(255,255,255,0.012)",
                                      }}
                                    >
                                      <div
                                        style={{
                                          display:
                                            "flex",

                                          justifyContent:
                                            "space-between",

                                          gap:
                                            "10px",

                                          flexWrap:
                                            "wrap",

                                          alignItems:
                                            "flex-start",
                                        }}
                                      >
                                        <div
                                          style={{
                                            minWidth:
                                              0,
                                          }}
                                        >
                                          <span
                                            style={{
                                              display:
                                                "block",

                                              fontSize:
                                                "0.58rem",

                                              opacity:
                                                0.45,
                                            }}
                                          >
                                            {
                                              issue.dataset_filename
                                            }
                                            {
                                              issue.column
                                                ? ` · ${issue.column}`
                                                : ""
                                            }
                                          </span>

                                          <strong
                                            style={{
                                              display:
                                                "block",

                                              marginTop:
                                                "4px",

                                              fontSize:
                                                "0.73rem",
                                            }}
                                          >
                                            {
                                              issue.title
                                            }
                                          </strong>
                                        </div>

                                        <span
                                          style={{
                                            padding:
                                              "4px 7px",

                                            border:
                                              `1px solid ${preparationSeverityBorder(
                                                issue.severity
                                              )}`,

                                            borderRadius:
                                              "999px",

                                            fontSize:
                                              "0.57rem",

                                            fontWeight:
                                              700,
                                          }}
                                        >
                                          {
                                            preparationSeverityLabel(
                                              issue.severity
                                            )
                                          }
                                        </span>
                                      </div>


                                      <p
                                        style={{
                                          margin:
                                            "8px 0 0",

                                          fontSize:
                                            "0.67rem",

                                          lineHeight:
                                            1.5,

                                          opacity:
                                            0.66,
                                        }}
                                      >
                                        {
                                          issue.explanation
                                        }
                                      </p>


                                      <p
                                        style={{
                                          margin:
                                            "5px 0 0",

                                          fontSize:
                                            "0.63rem",

                                          lineHeight:
                                            1.5,

                                          opacity:
                                            0.52,
                                        }}
                                      >
                                        {
                                          `${formatNumber(
                                            issue.evidence.observed_count
                                          )} observation(s) · ${formatPercent(
                                            issue.evidence.affected_ratio
                                          )}`
                                        }
                                      </p>


                                      {
                                        issue.evidence.examples.length >
                                        0
                                          ? (
                                              <div
                                                style={{
                                                  display:
                                                    "flex",

                                                  gap:
                                                    "5px",

                                                  flexWrap:
                                                    "wrap",

                                                  marginTop:
                                                    "7px",
                                                }}
                                              >
                                                {
                                                  issue.evidence.examples
                                                    .slice(
                                                      0,
                                                      5
                                                    )
                                                    .map(
                                                      (
                                                        example,
                                                        exampleIndex
                                                      ) => (
                                                        <code
                                                          key={
                                                            `${issue.issue_id}:${exampleIndex}`
                                                          }
                                                          style={{
                                                            padding:
                                                              "3px 5px",

                                                            border:
                                                              "1px solid rgba(255,255,255,0.055)",

                                                            borderRadius:
                                                              "5px",

                                                            fontSize:
                                                              "0.58rem",

                                                            opacity:
                                                              0.7,

                                                            overflowWrap:
                                                              "anywhere",
                                                          }}
                                                        >
                                                          {
                                                            example
                                                          }
                                                        </code>
                                                      )
                                                    )
                                                }
                                              </div>
                                            )
                                          : null
                                      }


                                      <div
                                        style={{
                                          marginTop:
                                            "8px",

                                          paddingTop:
                                            "8px",

                                          borderTop:
                                            "1px solid rgba(255,255,255,0.045)",
                                        }}
                                      >
                                        <p
                                          style={{
                                            margin:
                                              0,

                                            fontSize:
                                              "0.64rem",

                                            lineHeight:
                                              1.5,

                                            opacity:
                                              0.56,
                                          }}
                                        >
                                          <strong>
                                            Proposition Python :
                                          </strong>
                                          {" "}
                                          {
                                            issue.proposal.description
                                          }
                                        </p>

                                        <p
                                          style={{
                                            margin:
                                              "4px 0 0",

                                            fontSize:
                                              "0.59rem",

                                            opacity:
                                              0.46,
                                          }}
                                        >
                                          {
                                            issue.proposal.automatic_safe
                                              ? "Transformation déterministe possible"
                                              : "Décision automatique interdite"
                                          }
                                          {
                                            issue.semantic_review_recommended
                                              ? " · lecture sémantique recommandée"
                                              : ""
                                          }
                                        </p>
                                      </div>
                                    </article>
                                  )
                                )
                            }
                          </div>


                          {
                            qualityReport.issues.length >
                            12
                              ? (
                                  <p
                                    style={{
                                      margin:
                                        "8px 0 0",

                                      fontSize:
                                        "0.63rem",

                                      opacity:
                                        0.48,
                                    }}
                                  >
                                    {
                                      qualityReport.issues.length -
                                      12
                                    }
                                    {" autre(s) problème(s) masqué(s) dans cette vue compacte."}
                                  </p>
                                )
                              : null
                          }
                        </div>
                      )
                    : (
                        <div
                          style={{
                            marginTop:
                              "16px",

                            padding:
                              "13px",

                            border:
                              "1px solid rgba(122,203,160,0.16)",

                            borderRadius:
                              "10px",

                            background:
                              "rgba(74,143,103,0.025)",

                            fontSize:
                              "0.7rem",

                            lineHeight:
                              1.5,
                          }}
                        >
                          Aucun problème structurel évident
                          n’a été détecté par le moteur qualité.
                        </div>
                      )
                }


                <details
                  style={{
                    marginTop:
                      "12px",

                    border:
                      "1px solid rgba(255,255,255,0.055)",

                    borderRadius:
                      "10px",

                    background:
                      "rgba(255,255,255,0.01)",
                  }}
                >
                  <summary
                    style={{
                      padding:
                        "11px 12px",

                      cursor:
                        "pointer",

                      fontSize:
                        "0.69rem",

                      fontWeight:
                        700,
                    }}
                  >
                    Voir le rôle de l’IA et les contrôles avancés
                  </summary>

                  <div
                    style={{
                      padding:
                        "0 12px 12px",
                    }}
                  >
                    <div
                      style={{
                        display:
                          "grid",

                        gridTemplateColumns:
                          "repeat(3, minmax(0, 1fr))",

                        gap:
                          "8px",
                      }}
                    >
                      <article
                        style={{
                          padding:
                            "10px",

                          border:
                            "1px solid rgba(122,203,160,0.14)",

                          borderRadius:
                            "9px",
                        }}
                      >
                        <span
                          style={{
                            display:
                              "block",

                            opacity:
                              0.44,

                            fontSize:
                              "0.58rem",
                          }}
                        >
                          Python
                        </span>

                        <strong
                          style={{
                            display:
                              "block",

                            marginTop:
                              "4px",

                            fontSize:
                              "0.69rem",
                          }}
                        >
                          Diagnostic exécuté
                        </strong>

                        <p
                          style={{
                            margin:
                              "5px 0 0",

                            fontSize:
                              "0.61rem",

                            lineHeight:
                              1.45,

                            opacity:
                              0.5,
                          }}
                        >
                          Les anomalies visibles ci-dessus
                          proviennent du endpoint
                          /preparation/quality.
                        </p>
                      </article>


                      <article
                        style={{
                          padding:
                            "10px",

                          border:
                            "1px solid rgba(126,177,255,0.14)",

                          borderRadius:
                            "9px",
                        }}
                      >
                        <span
                          style={{
                            display:
                              "block",

                            opacity:
                              0.44,

                            fontSize:
                              "0.58rem",
                          }}
                        >
                          Gemma
                        </span>

                        <strong
                          style={{
                            display:
                              "block",

                            marginTop:
                              "4px",

                            fontSize:
                              "0.69rem",
                          }}
                        >
                          Interprétation à connecter
                        </strong>

                        <p
                          style={{
                            margin:
                              "5px 0 0",

                            fontSize:
                              "0.61rem",

                            lineHeight:
                              1.45,

                            opacity:
                              0.5,
                          }}
                        >
                          Le modèle interprétera seulement
                          les candidats sémantiques et devra
                          retourner une proposition structurée.
                        </p>
                      </article>


                      <article
                        style={{
                          padding:
                            "10px",

                          border:
                            "1px solid rgba(255,255,255,0.055)",

                          borderRadius:
                            "9px",
                        }}
                      >
                        <span
                          style={{
                            display:
                              "block",

                            opacity:
                              0.44,

                            fontSize:
                              "0.58rem",
                          }}
                        >
                          Exécution
                        </span>

                        <strong
                          style={{
                            display:
                              "block",

                            marginTop:
                              "4px",

                            fontSize:
                              "0.69rem",
                          }}
                        >
                          Toujours inactive
                        </strong>

                        <p
                          style={{
                            margin:
                              "5px 0 0",

                            fontSize:
                              "0.61rem",

                            lineHeight:
                              1.45,

                            opacity:
                              0.5,
                          }}
                        >
                          Aucun nettoyage n’est appliqué.
                          La prochaine étape ajoutera la
                          validation puis l’exécution Python.
                        </p>
                      </article>
                    </div>


                    {
                      qualityReport.notes.length >
                      0
                        ? (
                            <div
                              style={{
                                display:
                                  "grid",

                                gap:
                                  "5px",

                                marginTop:
                                  "10px",
                              }}
                            >
                              {
                                qualityReport.notes.map(
                                  (
                                    note,
                                    index
                                  ) => (
                                    <p
                                      key={
                                        `${qualityReport.rule_version}:note:${index}`
                                      }
                                      style={{
                                        margin:
                                          0,

                                        padding:
                                          "8px 9px",

                                        border:
                                          "1px solid rgba(255,255,255,0.045)",

                                        borderRadius:
                                          "8px",

                                        fontSize:
                                          "0.61rem",

                                        lineHeight:
                                          1.45,

                                        opacity:
                                          0.5,
                                      }}
                                    >
                                      {
                                        note
                                      }
                                    </p>
                                  )
                                )
                              }
                            </div>
                          )
                        : null
                    }
                  </div>
                </details>


                <div
                  style={{
                    marginTop:
                      "14px",

                    paddingTop:
                      "14px",

                    borderTop:
                      "1px solid rgba(255,255,255,0.06)",

                    fontSize:
                      "0.67rem",

                    lineHeight:
                      1.5,

                    opacity:
                      0.62,
                  }}
                >
                  <strong
                    style={{
                      color:
                        "rgba(151,218,180,0.86)",
                    }}
                  >
                    Diagnostic généré par Python · aucune donnée brute modifiée
                  </strong>

                  <span>
                    {" · "}
                    {
                      qualityReport.important_count >
                        0
                        ? "Des points importants restent à traiter avant une analyse automatique fiable."
                        : "Aucun blocage important n’est visible."
                    }
                  </span>
                </div>
              </>
            )
          : null
      }
    </section>
  );
}


function CleaningPlanPanel({
  plan,
  loading,
  error,
  selectedActionIds,
  execution,
  applyLoading,
  applyError,
  exportLoading,
  exportError,
  onToggleAction,
  onApply,
  onExportPrepared,
  onContinueSemantic,
}: {
  plan:
    CleaningPlanView |
    null;

  loading:
    boolean;

  error:
    string |
    null;

  selectedActionIds:
    string[];

  execution:
    CleaningExecutionView |
    null;

  applyLoading:
    boolean;

  applyError:
    string |
    null;

  exportLoading:
    boolean;

  exportError:
    string |
    null;

  onToggleAction:
    (
      actionId:
        string
    ) => void;

  onApply:
    () => void;

  onExportPrepared:
    () => void;

  onContinueSemantic:
    () => void;
}) {
  const rowsBefore =
    execution?.provenance.reduce(
      (
        total,
        item
      ) =>
        total +
        item.rows_before,
      0
    ) ??
    0;


  const rowsAfter =
    execution?.provenance.reduce(
      (
        total,
        item
      ) =>
        total +
        item.rows_after,
      0
    ) ??
    0;


  if (
    loading
  ) {
    return (
      <section
        style={{
          marginTop:
            "12px",

          padding:
            "14px",

          border:
            "1px solid rgba(126,177,255,0.12)",

          borderRadius:
            "12px",
        }}
      >
        <strong
          style={{
            fontSize:
              "0.72rem",
          }}
        >
          Construction du plan de nettoyage…
        </strong>
      </section>
    );
  }


  if (
    error
  ) {
    return (
      <section
        style={{
          marginTop:
            "12px",

          padding:
            "14px",

          border:
            "1px solid rgba(255,142,117,0.18)",

          borderRadius:
            "12px",
        }}
      >
        <strong
          style={{
            fontSize:
              "0.7rem",
          }}
        >
          Plan de nettoyage indisponible.
        </strong>

        <p
          style={{
            margin:
              "5px 0 0",

            fontSize:
              "0.65rem",

            lineHeight:
              1.5,

            opacity:
              0.58,
          }}
        >
          {
            error
          }
        </p>
      </section>
    );
  }


  if (
    plan ===
    null
  ) {
    return null;
  }


  const selectedCount =
    selectedActionIds.length;


  return (
    <section
      style={{
        marginTop:
          "12px",

        padding:
          "16px",

        border:
          execution
            ? "1px solid rgba(122,203,160,0.18)"
            : "1px solid rgba(126,177,255,0.12)",

        borderRadius:
          "14px",

        background:
          execution
            ? "rgba(122,203,160,0.018)"
            : "rgba(255,255,255,0.01)",
      }}
    >
      <div
        style={{
          display:
            "flex",

          justifyContent:
            "space-between",

          alignItems:
            "flex-start",

          gap:
            "14px",

          flexWrap:
            "wrap",
        }}
      >
        <div>
          <span
            style={{
              display:
                "block",

              fontSize:
                "0.59rem",

              textTransform:
                "uppercase",

              letterSpacing:
                "0.07em",

              opacity:
                0.46,
            }}
          >
            Plan de nettoyage
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "5px",

              fontSize:
                "0.86rem",
            }}
          >
            {
              execution
                ? `${execution.applied_action_count} correction(s) appliquée(s)`
                : `${plan.safe_candidate_count} correction(s) sûre(s) proposée(s)`
            }
          </strong>

          <p
            style={{
              margin:
                "6px 0 0",

              maxWidth:
                "760px",

              fontSize:
                "0.67rem",

              lineHeight:
                1.5,

              opacity:
                0.56,
            }}
          >
            {
              execution
                ? (
                    `${formatNumber(
                      rowsBefore
                    )} ligne(s) avant préparation · ` +
                    `${formatNumber(
                      rowsAfter
                    )} ligne(s) dans le dataset dérivé. ` +
                    "Le fichier source reste intact."
                  )
                : (
                    "DataLens a présélectionné uniquement les transformations déterministes. " +
                    "Les anomalies ambiguës, valeurs manquantes et outliers restent protégés."
                  )
            }
          </p>
        </div>

        <span
          style={{
            padding:
              "5px 8px",

            border:
              execution
                ? "1px solid rgba(122,203,160,0.18)"
                : "1px solid rgba(126,177,255,0.14)",

            borderRadius:
              "999px",

            fontSize:
              "0.58rem",

            fontWeight:
              700,
          }}
        >
          {
            execution
              ? "NETTOYAGE APPLIQUÉ"
              : `${selectedCount} / ${plan.safe_candidate_count} SÉLECTIONNÉES`
          }
        </span>
      </div>


      {
        plan.actions.length >
        0
          ? (
              <details
                style={{
                  marginTop:
                    "12px",

                  border:
                    "1px solid rgba(255,255,255,0.055)",

                  borderRadius:
                    "10px",
                }}
              >
                <summary
                  style={{
                    padding:
                      "10px 11px",

                    cursor:
                      execution
                        ? "default"
                        : "pointer",

                    fontSize:
                      "0.66rem",

                    fontWeight:
                      700,
                  }}
                >
                  Voir les corrections proposées
                </summary>

                <div
                  style={{
                    display:
                      "grid",

                    gap:
                      "7px",

                    padding:
                      "0 10px 10px",
                  }}
                >
                  {
                    plan.actions.map(
                      (
                        action
                      ) => {
                        const checked =
                          selectedActionIds.includes(
                            action.action_id
                          );


                        const applied =
                          execution?.action_results.some(
                            (
                              result
                            ) =>
                              result.action_id ===
                                action.action_id &&
                              result.status ===
                                "applied"
                          ) ??
                          false;


                        return (
                          <label
                            key={
                              action.action_id
                            }
                            style={{
                              display:
                                "grid",

                              gridTemplateColumns:
                                "auto minmax(0, 1fr) auto",

                              alignItems:
                                "start",

                              gap:
                                "9px",

                              padding:
                                "10px",

                              border:
                                applied
                                  ? "1px solid rgba(122,203,160,0.15)"
                                  : "1px solid rgba(255,255,255,0.045)",

                              borderRadius:
                                "9px",

                              cursor:
                                execution
                                  ? "default"
                                  : "pointer",
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={
                                checked
                              }
                              disabled={
                                Boolean(
                                  execution ||
                                  applyLoading ||
                                  !action.safe_candidate
                                )
                              }
                              onChange={
                                () =>
                                  onToggleAction(
                                    action.action_id
                                  )
                              }
                              aria-label={
                                `Sélectionner ${action.title}`
                              }
                            />

                            <span
                              style={{
                                minWidth:
                                  0,
                              }}
                            >
                              <strong
                                style={{
                                  display:
                                    "block",

                                  fontSize:
                                    "0.67rem",
                                }}
                              >
                                {
                                  action.title
                                }
                              </strong>

                              <span
                                style={{
                                  display:
                                    "block",

                                  marginTop:
                                    "3px",

                                  fontSize:
                                    "0.6rem",

                                  lineHeight:
                                    1.45,

                                  opacity:
                                    0.5,
                                }}
                              >
                                {
                                  action.dataset_filename
                                }
                                {
                                  action.column
                                    ? ` · ${action.column}`
                                    : ""
                                }
                                {" · "}
                                {
                                  formatNumber(
                                    action.affected_rows_estimate
                                  )
                                }
                                {" ligne(s) estimée(s)"}
                              </span>

                              {
                                action.before_examples.length >
                                0
                                  ? (
                                      <span
                                        style={{
                                          display:
                                            "block",

                                          marginTop:
                                            "4px",

                                          fontSize:
                                            "0.59rem",

                                          lineHeight:
                                            1.42,

                                          opacity:
                                            0.48,

                                          overflowWrap:
                                            "anywhere",
                                        }}
                                      >
                                        {
                                          action.before_examples
                                            .slice(
                                              0,
                                              3
                                            )
                                            .join(
                                              " · "
                                            )
                                        }
                                      </span>
                                    )
                                  : null
                              }
                            </span>

                            <span
                              style={{
                                padding:
                                  "3px 6px",

                                border:
                                  "1px solid rgba(122,203,160,0.14)",

                                borderRadius:
                                  "999px",

                                fontSize:
                                  "0.54rem",

                                fontWeight:
                                  700,

                                whiteSpace:
                                  "nowrap",
                              }}
                            >
                              {
                                applied
                                  ? "APPLIQUÉE"
                                  : "SÛRE"
                              }
                            </span>
                          </label>
                        );
                      }
                    )
                  }
                </div>
              </details>
            )
          : (
              <div
                style={{
                  marginTop:
                    "12px",

                  padding:
                    "11px",

                  border:
                    "1px solid rgba(122,203,160,0.14)",

                  borderRadius:
                    "9px",

                  fontSize:
                    "0.65rem",

                  lineHeight:
                    1.45,
                }}
              >
                Aucune correction déterministe n’est nécessaire.
              </div>
            )
      }


      {
        applyError
          ? (
              <p
                style={{
                  margin:
                    "10px 0 0",

                  padding:
                    "9px",

                  border:
                    "1px solid rgba(255,142,117,0.16)",

                  borderRadius:
                    "8px",

                  fontSize:
                    "0.63rem",

                  lineHeight:
                    1.45,
                }}
              >
                {
                  applyError
                }
              </p>
            )
          : null
      }


      {
        exportError
          ? (
              <p
                style={{
                  margin:
                    "8px 0 0",

                  padding:
                    "9px",

                  border:
                    "1px solid rgba(255,142,117,0.16)",

                  borderRadius:
                    "8px",

                  fontSize:
                    "0.63rem",

                  lineHeight:
                    1.45,
                }}
              >
                {
                  exportError
                }
              </p>
            )
          : null
      }


      <div
        style={{
          display:
            "flex",

          justifyContent:
            "space-between",

          alignItems:
            "center",

          gap:
            "12px",

          flexWrap:
            "wrap",

          marginTop:
            "12px",

          paddingTop:
            "12px",

          borderTop:
            "1px solid rgba(255,255,255,0.05)",
        }}
      >
        <p
          style={{
            margin:
              0,

            fontSize:
              "0.62rem",

            lineHeight:
              1.45,

            opacity:
              0.5,
          }}
        >
          {
            execution
              ? (
                  "Les prochaines analyses utiliseront le dataset dérivé nettoyé."
                )
              : (
                  `${plan.protected_issue_count} problème(s) restent protégés et ne seront pas corrigés automatiquement.`
                )
          }
        </p>

        <div
          style={{
            display:
              "flex",

            gap:
              "8px",

            flexWrap:
              "wrap",

            justifyContent:
              "flex-end",
          }}
        >
          {
            execution
              ? (
                  <button
                    type="button"
                    className={
                      styles.secondaryButton
                    }
                    disabled={
                      exportLoading
                    }
                    onClick={
                      onExportPrepared
                    }
                    style={{
                      minWidth:
                        "210px",
                    }}
                  >
                    {
                      exportLoading
                        ? "Préparation du fichier…"
                        : "Télécharger après corrections sûres"
                    }
                  </button>
                )
              : null
          }


          {
            plan.action_count ===
              0
              ? (
                  <div
                    style={{
                      display:
                        "flex",

                      alignItems:
                        "center",

                      justifyContent:
                        "flex-end",

                      gap:
                        "12px",

                      flexWrap:
                        "wrap",
                    }}
                  >
                    <span
                      style={{
                        color:
                          "rgba(151,218,180,0.92)",

                        fontSize:
                          "0.72rem",

                        fontWeight:
                          700,
                      }}
                    >
                      {
                        plan.protected_issue_count >
                          0
                          ? (
                              "Aucune correction déterministe à appliquer"
                            )
                          : (
                              "Aucun nettoyage nécessaire"
                            )
                      }
                    </span>

                    <button
                      type="button"
                      className={
                        styles.submitButton
                      }
                      onClick={
                        onContinueSemantic
                      }
                      style={{
                        minWidth:
                          "210px",
                      }}
                    >
                      {
                        plan.protected_issue_count >
                          0
                          ? (
                              "Continuer vers la revue sémantique"
                            )
                          : (
                              "Continuer sans modification"
                            )
                      }
                    </button>
                  </div>
                )
              : (
                  <button
                    type="button"
                    className={
                      styles.submitButton
                    }
                    disabled={
                      Boolean(
                        execution ||
                        applyLoading ||
                        selectedCount ===
                          0
                      )
                    }
                    onClick={
                      onApply
                    }
                    style={{
                      minWidth:
                        "210px",
                    }}
                  >
                    {
                      execution
                        ? "Nettoyage appliqué"
                        : (
                            applyLoading
                              ? "Application…"
                              : "Appliquer les corrections sûres"
                          )
                    }
                  </button>
                )
          }
        </div>
      </div>
    </section>
  );
}


function SemanticReviewPanel({
  deterministicCleaningReady,
  review,
  reviewLoading,
  reviewError,
  plan,
  planLoading,
  planError,
  selectedActionIds,
  canonicalValues,
  execution,
  applyLoading,
  applyError,
  onRunReview,
  onSetDecision,
  onCanonicalChange,
  onApply,
}: {
  deterministicCleaningReady:
    boolean;

  review:
    SemanticReviewReportView |
    null;

  reviewLoading:
    boolean;

  reviewError:
    string |
    null;

  plan:
    SemanticCleaningPlanView |
    null;

  planLoading:
    boolean;

  planError:
    string |
    null;

  selectedActionIds:
    string[];

  canonicalValues:
    Record<
      string,
      string
    >;

  execution:
    SemanticCleaningExecutionView |
    null;

  applyLoading:
    boolean;

  applyError:
    string |
    null;

  onRunReview:
    () => void;

  onSetDecision:
    (
      actionId:
        string,
      shouldMerge:
        boolean
    ) => void;

  onCanonicalChange:
    (
      actionId:
        string,
      canonicalValue:
        string
    ) => void;

  onApply:
    () => void;
}) {
  if (
    !deterministicCleaningReady
  ) {
    return (
      <section
        style={{
          marginTop:
            "12px",

          padding:
            "14px",

          border:
            "1px solid rgba(255,255,255,0.055)",

          borderRadius:
            "12px",

          fontSize:
            "0.64rem",

          lineHeight:
            1.5,

          opacity:
            0.56,
        }}
      >
        La revue sémantique devient disponible après
        l’application des corrections déterministes sélectionnées.
      </section>
    );
  }


  const flaggedCount =
    review?.decisions.filter(
      (
        decision
      ) =>
        decision.verdict ===
        "flag_for_review"
    ).length ??
    0;


  const selectedCount =
    selectedActionIds.length;


  return (
    <section
      style={{
        marginTop:
          "12px",

        padding:
          "16px",

        border:
          execution
            ? "1px solid rgba(122,203,160,0.18)"
            : "1px solid rgba(160,133,255,0.16)",

        borderRadius:
          "14px",

        background:
          execution
            ? "rgba(122,203,160,0.018)"
            : "rgba(160,133,255,0.018)",
      }}
    >
      <div
        style={{
          display:
            "flex",

          justifyContent:
            "space-between",

          alignItems:
            "flex-start",

          gap:
            "14px",

          flexWrap:
            "wrap",
        }}
      >
        <div
          style={{
            maxWidth:
              "720px",
          }}
        >
          <span
            style={{
              display:
                "block",

              fontSize:
                "0.59rem",

              textTransform:
                "uppercase",

              letterSpacing:
                "0.07em",

              opacity:
                0.46,
            }}
          >
            Revue sémantique locale
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "5px",

              fontSize:
                "0.86rem",
            }}
          >
            {
              execution
                ? `${execution.applied_action_count} fusion(s) confirmée(s)`
                : review
                  ? `${review.merge_proposal_count} fusion(s) proposée(s) par Gemma`
                  : "Interpréter uniquement les ambiguïtés restantes"
            }
          </strong>

          <p
            style={{
              margin:
                "6px 0 0",

              fontSize:
                "0.64rem",

              lineHeight:
                1.5,

              opacity:
                0.56,
            }}
          >
            Gemma propose. Python revalide les valeurs exactes.
            Aucune fusion n’est exécutée sans votre confirmation.
          </p>
        </div>


        <span
          style={{
            padding:
              "4px 7px",

            border:
              "1px solid rgba(160,133,255,0.16)",

            borderRadius:
              "999px",

            fontSize:
              "0.54rem",

            fontWeight:
              700,

            letterSpacing:
              "0.04em",
          }}
        >
          {
            review
              ? `${review.model} · ${review.rule_version}`
              : "GEMMA · LOCAL"
          }
        </span>
      </div>


      {
        !review &&
        !reviewLoading
          ? (
              <div
                style={{
                  marginTop:
                    "13px",

                  display:
                    "flex",

                  justifyContent:
                    "space-between",

                  alignItems:
                    "center",

                  gap:
                    "12px",

                  flexWrap:
                    "wrap",
                }}
              >
                <p
                  style={{
                    margin:
                      0,

                    fontSize:
                      "0.62rem",

                    lineHeight:
                      1.5,

                    opacity:
                      0.5,
                  }}
                >
                  Seuls les signaux marqués pour revue sémantique
                  et un contexte minimal sont transmis au modèle local.
                </p>

                <button
                  type="button"
                  className={
                    styles.submitButton
                  }
                  onClick={
                    onRunReview
                  }
                  style={{
                    minWidth:
                      "210px",
                  }}
                >
                  Lancer la revue sémantique
                </button>
              </div>
            )
          : null
      }


      {
        reviewLoading
          ? (
              <div
                style={{
                  marginTop:
                    "13px",

                  padding:
                    "11px",

                  border:
                    "1px solid rgba(160,133,255,0.12)",

                  borderRadius:
                    "9px",

                  fontSize:
                    "0.64rem",
                }}
              >
                Gemma examine les ambiguïtés une par une…
              </div>
            )
          : null
      }


      {
        reviewError
          ? (
              <p
                style={{
                  margin:
                    "11px 0 0",

                  padding:
                    "10px",

                  border:
                    "1px solid rgba(255,142,117,0.16)",

                  borderRadius:
                    "9px",

                  fontSize:
                    "0.63rem",

                  lineHeight:
                    1.5,
                }}
              >
                {
                  reviewError
                }
              </p>
            )
          : null
      }


      {
        review
          ? (
              <div
                style={{
                  display:
                    "grid",

                  gridTemplateColumns:
                    "repeat(auto-fit, minmax(150px, 1fr))",

                  gap:
                    "8px",

                  marginTop:
                    "13px",
                }}
              >
                <article
                  style={{
                    padding:
                      "10px",

                    border:
                      "1px solid rgba(255,255,255,0.05)",

                    borderRadius:
                      "9px",
                  }}
                >
                  <span
                    style={{
                      display:
                        "block",

                      fontSize:
                        "0.55rem",

                      opacity:
                        0.46,
                    }}
                  >
                    Candidats examinés
                  </span>

                  <strong
                    style={{
                      display:
                        "block",

                      marginTop:
                        "4px",

                      fontSize:
                        "0.76rem",
                    }}
                  >
                    {
                      review.candidate_count
                    }
                  </strong>
                </article>


                <article
                  style={{
                    padding:
                      "10px",

                    border:
                      "1px solid rgba(160,133,255,0.12)",

                    borderRadius:
                      "9px",
                  }}
                >
                  <span
                    style={{
                      display:
                        "block",

                      fontSize:
                        "0.55rem",

                      opacity:
                        0.46,
                    }}
                  >
                    Fusions possibles
                  </span>

                  <strong
                    style={{
                      display:
                        "block",

                      marginTop:
                        "4px",

                      fontSize:
                        "0.76rem",
                    }}
                  >
                    {
                      review.merge_proposal_count
                    }
                  </strong>
                </article>


                <article
                  style={{
                    padding:
                      "10px",

                    border:
                      "1px solid rgba(255,255,255,0.05)",

                    borderRadius:
                      "9px",
                  }}
                >
                  <span
                    style={{
                      display:
                        "block",

                      fontSize:
                        "0.55rem",

                      opacity:
                        0.46,
                    }}
                  >
                    Revue manuelle
                  </span>

                  <strong
                    style={{
                      display:
                        "block",

                      marginTop:
                        "4px",

                      fontSize:
                        "0.76rem",
                    }}
                  >
                    {
                      review.abstention_count +
                      flaggedCount
                    }
                  </strong>
                </article>
              </div>
            )
          : null
      }


      {
        planLoading
          ? (
              <p
                style={{
                  margin:
                    "12px 0 0",

                  fontSize:
                    "0.63rem",

                  opacity:
                    0.54,
                }}
              >
                Python reconstruit les actions sémantiques autorisées…
              </p>
            )
          : null
      }


      {
        planError
          ? (
              <p
                style={{
                  margin:
                    "11px 0 0",

                  padding:
                    "10px",

                  border:
                    "1px solid rgba(255,142,117,0.16)",

                  borderRadius:
                    "9px",

                  fontSize:
                    "0.63rem",
                }}
              >
                {
                  planError
                }
              </p>
            )
          : null
      }


      {
        plan &&
        plan.actions.length >
        0
          ? (
              <div
                style={{
                  display:
                    "grid",

                  gap:
                    "9px",

                  marginTop:
                    "13px",
                }}
              >
                {
                  plan.actions.map(
                    (
                      action
                    ) => {
                      const shouldMerge =
                        selectedActionIds.includes(
                          action.action_id
                        );


                      const canonicalValue =
                        canonicalValues[
                          action.action_id
                        ] ??
                        action
                          .suggested_canonical_value;


                      const actionResult =
                        execution
                          ?.action_results
                          .find(
                            (
                              result
                            ) =>
                              result.action_id ===
                              action.action_id
                          );


                      const applied =
                        actionResult?.status ===
                        "applied";


                      return (
                        <article
                          key={
                            action.action_id
                          }
                          style={{
                            padding:
                              "12px",

                            border:
                              applied
                                ? "1px solid rgba(122,203,160,0.18)"
                                : "1px solid rgba(255,255,255,0.055)",

                            borderRadius:
                              "10px",

                            background:
                              applied
                                ? "rgba(122,203,160,0.018)"
                                : "rgba(255,255,255,0.008)",
                          }}
                        >
                          <div
                            style={{
                              display:
                                "flex",

                              justifyContent:
                                "space-between",

                              gap:
                                "10px",

                              flexWrap:
                                "wrap",
                            }}
                          >
                            <div>
                              <strong
                                style={{
                                  display:
                                    "block",

                                  fontSize:
                                    "0.72rem",
                                }}
                              >
                                {
                                  action.column
                                }
                              </strong>

                              <span
                                style={{
                                  display:
                                    "block",

                                  marginTop:
                                    "4px",

                                  fontSize:
                                    "0.62rem",

                                  lineHeight:
                                    1.45,

                                  opacity:
                                    0.56,
                                }}
                              >
                                {
                                  action.source_values.join(
                                    " · "
                                  )
                                }
                              </span>
                            </div>

                            <span
                              style={{
                                fontSize:
                                  "0.56rem",

                                opacity:
                                  0.58,
                              }}
                            >
                              {
                                `IA ${Math.round(
                                  action.confidence *
                                  100
                                )} % · Python validé`
                              }
                            </span>
                          </div>


                          <p
                            style={{
                              margin:
                                "8px 0 0",

                              fontSize:
                                "0.61rem",

                              lineHeight:
                                1.5,

                              opacity:
                                0.52,
                            }}
                          >
                            {
                              action.rationale
                            }
                          </p>


                          <div
                            style={{
                              marginTop:
                                "10px",

                              paddingTop:
                                "10px",

                              borderTop:
                                "1px solid rgba(255,255,255,0.05)",
                            }}
                          >
                            <span
                              style={{
                                display:
                                  "block",

                                marginBottom:
                                  "7px",

                                fontSize:
                                  "0.57rem",

                                textTransform:
                                  "uppercase",

                                letterSpacing:
                                  "0.05em",

                                opacity:
                                  0.46,
                              }}
                            >
                              Valeur canonique si fusion
                            </span>

                            <div
                              style={{
                                display:
                                  "flex",

                                gap:
                                  "7px",

                                flexWrap:
                                  "wrap",
                              }}
                            >
                              {
                                action
                                  .allowed_canonical_values
                                  .map(
                                    (
                                      value
                                    ) => (
                                      <label
                                        key={
                                          value
                                        }
                                        style={{
                                          display:
                                            "inline-flex",

                                          alignItems:
                                            "center",

                                          gap:
                                            "6px",

                                          padding:
                                            "6px 8px",

                                          border:
                                            canonicalValue ===
                                            value
                                              ? "1px solid rgba(160,133,255,0.28)"
                                              : "1px solid rgba(255,255,255,0.06)",

                                          borderRadius:
                                            "8px",

                                          fontSize:
                                            "0.61rem",

                                          cursor:
                                            execution
                                              ? "default"
                                              : "pointer",
                                        }}
                                      >
                                        <input
                                          type="radio"
                                          name={
                                            `semantic-${action.action_id}`
                                          }
                                          value={
                                            value
                                          }
                                          checked={
                                            canonicalValue ===
                                            value
                                          }
                                          disabled={
                                            Boolean(
                                              execution
                                            )
                                          }
                                          onChange={
                                            () =>
                                              onCanonicalChange(
                                                action.action_id,
                                                value
                                              )
                                          }
                                        />

                                        {
                                          value
                                        }
                                      </label>
                                    )
                                  )
                              }
                            </div>
                          </div>


                          <div
                            style={{
                              display:
                                "flex",

                              justifyContent:
                                "flex-end",

                              gap:
                                "7px",

                              flexWrap:
                                "wrap",

                              marginTop:
                                "10px",
                            }}
                          >
                            <button
                              type="button"
                              className={
                                styles.secondaryButton
                              }
                              disabled={
                                Boolean(
                                  execution
                                )
                              }
                              onClick={
                                () =>
                                  onSetDecision(
                                    action.action_id,
                                    false
                                  )
                              }
                              style={{
                                minWidth:
                                  "145px",
                              }}
                            >
                              {
                                !shouldMerge
                                  ? "Conserver séparées"
                                  : "Ne pas fusionner"
                              }
                            </button>

                            <button
                              type="button"
                              className={
                                styles.submitButton
                              }
                              disabled={
                                Boolean(
                                  execution
                                )
                              }
                              onClick={
                                () =>
                                  onSetDecision(
                                    action.action_id,
                                    true
                                  )
                              }
                              style={{
                                minWidth:
                                  "145px",
                              }}
                            >
                              {
                                applied
                                  ? "Fusion appliquée"
                                  : shouldMerge
                                    ? "Fusion sélectionnée"
                                    : "Fusionner"
                              }
                            </button>
                          </div>
                        </article>
                      );
                    }
                  )
                }
              </div>
            )
          : null
      }


      {
        review &&
        plan &&
        plan.actions.length ===
        0
          ? (
              <div
                style={{
                  marginTop:
                    "12px",

                  padding:
                    "11px",

                  border:
                    "1px solid rgba(255,255,255,0.055)",

                  borderRadius:
                    "9px",

                  fontSize:
                    "0.63rem",

                  lineHeight:
                    1.5,

                  opacity:
                    0.58,
                }}
              >
                Aucune fusion sémantique suffisamment sûre
                n’a été validée par Python. Les ambiguïtés restent
                disponibles pour revue manuelle.
              </div>
            )
          : null
      }


      {
        applyError
          ? (
              <p
                style={{
                  margin:
                    "11px 0 0",

                  padding:
                    "10px",

                  border:
                    "1px solid rgba(255,142,117,0.16)",

                  borderRadius:
                    "9px",

                  fontSize:
                    "0.63rem",
                }}
              >
                {
                  applyError
                }
              </p>
            )
          : null
      }


      {
        plan &&
        plan.actions.length >
        0
          ? (
              <div
                style={{
                  display:
                    "flex",

                  justifyContent:
                    "space-between",

                  alignItems:
                    "center",

                  gap:
                    "12px",

                  flexWrap:
                    "wrap",

                  marginTop:
                    "13px",

                  paddingTop:
                    "12px",

                  borderTop:
                    "1px solid rgba(255,255,255,0.05)",
                }}
              >
                <p
                  style={{
                    margin:
                      0,

                    fontSize:
                      "0.62rem",

                    lineHeight:
                      1.45,

                    opacity:
                      0.52,
                  }}
                >
                  {
                    execution
                      ? `${execution.changed_cell_count} cellule(s) modifiée(s) · les analyses utiliseront ce dataset dérivé.`
                      : `${selectedCount} fusion(s) explicitement sélectionnée(s).`
                  }
                </p>

                <button
                  type="button"
                  className={
                    styles.submitButton
                  }
                  disabled={
                    Boolean(
                      execution ||
                      applyLoading ||
                      selectedCount ===
                        0
                    )
                  }
                  onClick={
                    onApply
                  }
                  style={{
                    minWidth:
                      "220px",
                  }}
                >
                  {
                    execution
                      ? "Nettoyage sémantique appliqué"
                      : applyLoading
                        ? "Application…"
                        : "Appliquer les fusions confirmées"
                  }
                </button>
              </div>
            )
          : null
      }
    </section>
  );
}


function QualityReportSection({
  report,
  cleaningPlan,
  cleaningExecution,
}: {
  report:
    DataQualityReportView |
    null;

  cleaningPlan:
    CleaningPlanView |
    null;

  cleaningExecution:
    CleaningExecutionView |
    null;
}) {
  if (
    report ===
    null
  ) {
    return null;
  }


  const duplicateRows =
    report.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        dataset.duplicate_row_count,
      0
    );


  const missingCells =
    report.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        dataset.missing_cell_count,
      0
    );


  const deterministicProposalCount =
    cleaningPlan?.action_count ??
    report.issues.filter(
      (
        issue
      ) =>
        issue.proposal
          .automatic_safe
    ).length;


  const appliedTransformationCount =
    cleaningExecution
      ?.applied_action_count ??
    0;


  const preparedRowsBefore =
    cleaningExecution?.provenance.reduce(
      (
        total,
        item
      ) =>
        total +
        item.rows_before,
      0
    ) ??
    report.total_rows;


  const preparedRowsAfter =
    cleaningExecution?.provenance.reduce(
      (
        total,
        item
      ) =>
        total +
        item.rows_after,
      0
    ) ??
    report.total_rows;


  const importantIssues =
    report.issues.filter(
      (
        issue
      ) =>
        issue.severity ===
        "important"
    );


  const hiddenIssueCount =
    Math.max(
      report.issue_count -
      importantIssues.length,
      0
    );


  return (
    <section
      style={{
        marginBottom:
          "16px",

        padding:
          "16px",

        border:
          "1px solid rgba(126,177,255,0.11)",

        borderRadius:
          "14px",

        background:
          "linear-gradient(180deg, rgba(126,177,255,0.022), rgba(255,255,255,0.008))",
      }}
    >
      <div
        style={{
          display:
            "flex",

          justifyContent:
            "space-between",

          alignItems:
            "flex-start",

          gap:
            "14px",

          flexWrap:
            "wrap",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Préparation des données
          </span>

          <h2
            style={{
              marginBottom:
                "4px",
            }}
          >
            Qualité des données
          </h2>

          <p
            className={
              styles.resultSubtitle
            }
            style={{
              maxWidth:
                "700px",
            }}
          >
            Résumé des contrôles effectués avant
            l’analyse. Le détail complet reste
            disponible sans alourdir le rapport.
          </p>
        </div>


        <div
          style={{
            display:
              "flex",

            gap:
              "6px",

            flexWrap:
              "wrap",
          }}
        >
          <span
            style={{
              padding:
                "5px 8px",

              border:
                "1px solid rgba(122,203,160,0.14)",

              borderRadius:
                "999px",

              fontSize:
                "0.56rem",

              fontWeight:
                700,
            }}
          >
            SOURCE CONSERVÉE
          </span>

          <span
            style={{
              padding:
                "5px 8px",

              border:
                "1px solid rgba(255,255,255,0.07)",

              borderRadius:
                "999px",

              fontSize:
                "0.56rem",

              fontWeight:
                700,

              opacity:
                0.68,
            }}
          >
            0 MODIFICATION SILENCIEUSE
          </span>
        </div>
      </div>


      <div
        style={{
          display:
            "grid",

          gridTemplateColumns:
            "repeat(auto-fit, minmax(170px, 1fr))",

          gap:
            "8px",

          marginTop:
            "12px",
        }}
      >
        <article
          style={{
            padding:
              "11px 12px",

            border:
              "1px solid rgba(255,255,255,0.055)",

            borderRadius:
              "10px",
          }}
        >
          <span
            style={{
              display:
                "block",

              fontSize:
                "0.57rem",

              textTransform:
                "uppercase",

              letterSpacing:
                "0.06em",

              opacity:
                0.44,
            }}
          >
            Détecté
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "4px",

              fontSize:
                "0.82rem",
            }}
          >
            {
              `${formatNumber(
                report.issue_count
              )} problème(s)`
            }
          </strong>

          <small
            style={{
              display:
                "block",

              marginTop:
                "3px",

              opacity:
                0.48,
            }}
          >
            {
              `${report.important_count} importants · ` +
              `${report.moderate_count} modérés · ` +
              `${report.minor_count} mineurs`
            }
          </small>
        </article>


        <article
          style={{
            padding:
              "11px 12px",

            border:
              "1px solid rgba(126,177,255,0.11)",

            borderRadius:
              "10px",

            background:
              "rgba(126,177,255,0.014)",
          }}
        >
          <span
            style={{
              display:
                "block",

              fontSize:
                "0.57rem",

              textTransform:
                "uppercase",

              letterSpacing:
                "0.06em",

              opacity:
                0.44,
            }}
          >
            Corrigé
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "4px",

              fontSize:
                "0.82rem",
            }}
          >
            {
              `${formatNumber(
                appliedTransformationCount
              )} transformation(s)`
            }
          </strong>

          <small
            style={{
              display:
                "block",

              marginTop:
                "3px",

              opacity:
                0.48,
            }}
          >
            {
              `${formatNumber(
                deterministicProposalCount
              )} correction(s) déterministe(s) proposée(s)`
            }
          </small>
        </article>


        <article
          style={{
            padding:
              "11px 12px",

            border:
              "1px solid rgba(122,203,160,0.13)",

            borderRadius:
              "10px",

            background:
              "rgba(122,203,160,0.014)",
          }}
        >
          <span
            style={{
              display:
                "block",

              fontSize:
                "0.57rem",

              textTransform:
                "uppercase",

              letterSpacing:
                "0.06em",

              opacity:
                0.44,
            }}
          >
            Analysé
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "4px",

              fontSize:
                "0.82rem",
            }}
          >
            {
              `${formatNumber(
                preparedRowsAfter
              )} ligne(s)`
            }
          </strong>

          <small
            style={{
              display:
                "block",

              marginTop:
                "3px",

              opacity:
                0.48,
            }}
          >
            {
              cleaningExecution
                ? (
                    `${formatNumber(
                      preparedRowsBefore
                    )} avant préparation`
                  )
                : "Dataset source"
            }
          </small>
        </article>
      </div>


      <details
        style={{
          marginTop:
            "11px",

          border:
            "1px solid rgba(255,255,255,0.05)",

          borderRadius:
            "10px",

          overflow:
            "hidden",
        }}
      >
        <summary
          style={{
            padding:
              "10px 11px",

            cursor:
              "pointer",

            fontSize:
              "0.63rem",

            fontWeight:
              700,
          }}
        >
          Voir le détail de la préparation
        </summary>


        <div
          style={{
            padding:
              "0 11px 11px",
          }}
        >
          <div
            style={{
              display:
                "grid",

              gridTemplateColumns:
                "repeat(auto-fit, minmax(145px, 1fr))",

              gap:
                "7px",
            }}
          >
            <article
              style={{
                padding:
                  "9px",

                border:
                  "1px solid rgba(255,255,255,0.045)",

                borderRadius:
                  "8px",
              }}
            >
              <span
                style={{
                  display:
                    "block",

                  fontSize:
                    "0.55rem",

                  opacity:
                    0.44,
                }}
              >
                Cellules manquantes
              </span>

              <strong
                style={{
                  display:
                    "block",

                  marginTop:
                    "3px",

                  fontSize:
                    "0.72rem",
                }}
              >
                {
                  formatNumber(
                    missingCells
                  )
                }
              </strong>
            </article>


            <article
              style={{
                padding:
                  "9px",

                border:
                  "1px solid rgba(255,255,255,0.045)",

                borderRadius:
                  "8px",
              }}
            >
              <span
                style={{
                  display:
                    "block",

                  fontSize:
                    "0.55rem",

                  opacity:
                    0.44,
                }}
              >
                Doublons stricts
              </span>

              <strong
                style={{
                  display:
                    "block",

                  marginTop:
                    "3px",

                  fontSize:
                    "0.72rem",
                }}
              >
                {
                  formatNumber(
                    duplicateRows
                  )
                }
              </strong>
            </article>


            <article
              style={{
                padding:
                  "9px",

                border:
                  "1px solid rgba(255,255,255,0.045)",

                borderRadius:
                  "8px",
              }}
            >
              <span
                style={{
                  display:
                    "block",

                  fontSize:
                    "0.55rem",

                  opacity:
                    0.44,
                }}
              >
                Revue sémantique
              </span>

              <strong
                style={{
                  display:
                    "block",

                  marginTop:
                    "3px",

                  fontSize:
                    "0.72rem",
                }}
              >
                {
                  formatNumber(
                    report.semantic_review_count
                  )
                }
              </strong>
            </article>


            <article
              style={{
                padding:
                  "9px",

                border:
                  "1px solid rgba(255,255,255,0.045)",

                borderRadius:
                  "8px",
              }}
            >
              <span
                style={{
                  display:
                    "block",

                  fontSize:
                    "0.55rem",

                  opacity:
                    0.44,
                }}
              >
                Moteur qualité
              </span>

              <strong
                style={{
                  display:
                    "block",

                  marginTop:
                    "3px",

                  fontSize:
                    "0.62rem",

                  overflowWrap:
                    "anywhere",
                }}
              >
                {
                  report.rule_version
                }
              </strong>
            </article>
          </div>


          {
            importantIssues.length >
            0
              ? (
                  <div
                    style={{
                      marginTop:
                        "10px",
                    }}
                  >
                    <div
                      style={{
                        display:
                          "flex",

                        justifyContent:
                          "space-between",

                        alignItems:
                          "center",

                        gap:
                          "10px",

                        flexWrap:
                          "wrap",

                        marginBottom:
                          "6px",
                      }}
                    >
                      <strong
                        style={{
                          fontSize:
                            "0.61rem",

                          textTransform:
                            "uppercase",

                          letterSpacing:
                            "0.055em",

                          opacity:
                            0.58,
                        }}
                      >
                        Problèmes importants
                      </strong>

                      <span
                        style={{
                          fontSize:
                            "0.54rem",

                          opacity:
                            0.42,
                        }}
                      >
                        {
                          `${importantIssues.length} à surveiller`
                        }
                      </span>
                    </div>


                    <div
                      style={{
                        display:
                          "grid",

                        gap:
                          "6px",
                      }}
                    >
                      {
                        importantIssues.map(
                          (
                            issue
                          ) => (
                            <article
                              key={
                                `report-detail:${issue.issue_id}`
                              }
                              style={{
                                display:
                                  "grid",

                                gridTemplateColumns:
                                  "minmax(0, 1fr) auto",

                                gap:
                                  "9px",

                                alignItems:
                                  "center",

                                padding:
                                  "8px 9px",

                                border:
                                  "1px solid rgba(255,255,255,0.042)",

                                borderRadius:
                                  "8px",
                              }}
                            >
                              <div
                                style={{
                                  minWidth:
                                    0,
                                }}
                              >
                                <strong
                                  style={{
                                    display:
                                      "block",

                                    fontSize:
                                      "0.62rem",
                                  }}
                                >
                                  {
                                    issue.title
                                  }
                                </strong>

                                <span
                                  style={{
                                    display:
                                      "block",

                                    marginTop:
                                      "2px",

                                    fontSize:
                                      "0.56rem",

                                    lineHeight:
                                      1.4,

                                    opacity:
                                      0.46,

                                    overflowWrap:
                                      "anywhere",
                                  }}
                                >
                                  {
                                    issue.dataset_filename
                                  }
                                  {
                                    issue.column
                                      ? ` · ${issue.column}`
                                      : ""
                                  }
                                  {" · "}
                                  {
                                    formatNumber(
                                      issue.evidence
                                        .observed_count
                                    )
                                  }
                                  {" observation(s)"}
                                </span>
                              </div>

                              <span
                                style={{
                                  padding:
                                    "3px 6px",

                                  border:
                                    `1px solid ${preparationSeverityBorder(
                                      issue.severity
                                    )}`,

                                  borderRadius:
                                    "999px",

                                  fontSize:
                                    "0.52rem",

                                  fontWeight:
                                    700,

                                  whiteSpace:
                                    "nowrap",
                                }}
                              >
                                Important
                              </span>
                            </article>
                          )
                        )
                      }
                    </div>
                  </div>
                )
              : null
          }


          {
            hiddenIssueCount >
            0
              ? (
                  <p
                    style={{
                      margin:
                        "9px 0 0",

                      padding:
                        "8px 9px",

                      border:
                        "1px solid rgba(126,177,255,0.07)",

                      borderRadius:
                        "8px",

                      fontSize:
                        "0.56rem",

                      lineHeight:
                        1.45,

                      opacity:
                        0.5,
                    }}
                  >
                    {
                      `${hiddenIssueCount} autre(s) problème(s) modéré(s) ou mineur(s) ` +
                      "restent consultables dans l’étape Préparation."
                    }
                  </p>
                )
              : null
          }


          <p
            style={{
              margin:
                "9px 0 0",

              paddingTop:
                "9px",

              borderTop:
                "1px solid rgba(255,255,255,0.045)",

              fontSize:
                "0.56rem",

              lineHeight:
                1.45,

              opacity:
                0.46,
            }}
          >
            Le rapport conserve uniquement les
            anomalies importantes et la traçabilité.
            Le diagnostic complet reste disponible
            dans Préparation. Les corrections
            ambiguës restent protégées et aucune
            donnée source n’est remplacée
            silencieusement.
          </p>
        </div>
      </details>
    </section>
  );
}


type WorkspaceStep =
  | "data"
  | "documents"
  | "preparation"
  | "analyses"
  | "report";


type WorkspaceStepDefinition = {
  id:
    WorkspaceStep;

  label:
    string;

  shortLabel:
    string;
};


const WORKSPACE_STEPS:
  WorkspaceStepDefinition[] = [
    {
      id:
        "data",

      label:
        "Données",

      shortLabel:
        "1",
    },

    {
      id:
        "documents",

      label:
        "Documents",

      shortLabel:
        "2",
    },

    {
      id:
        "preparation",

      label:
        "Préparation",

      shortLabel:
        "3",
    },

    {
      id:
        "analyses",

      label:
        "Analyses",

      shortLabel:
        "4",
    },

    {
      id:
        "report",

      label:
        "Rapport",

      shortLabel:
        "5",
    },
  ];


function WorkspaceNavigation({
  activeStep,
  onStepChange,
  dataReady,
  reportReady,
  interventionCount,
}: {
  activeStep:
    WorkspaceStep;

  onStepChange:
    (
      step:
        WorkspaceStep
    ) => void;

  dataReady:
    boolean;

  reportReady:
    boolean;

  interventionCount:
    number;
}) {
  function isEnabled(
    step:
      WorkspaceStep
  ): boolean {
    if (
      step ===
      "data"
    ) {
      return true;
    }


    if (
      step ===
      "documents"
    ) {
      return dataReady;
    }


    if (
      step ===
      "preparation"
    ) {
      return dataReady;
    }


    return reportReady;
  }


  return (
    <nav
      aria-label="Navigation du workspace"
      className={
        styles.workspaceNav
      }
    >
      <div
        role="group"
        aria-label="Étapes d’analyse"
        className={
          styles.workspaceSteps
        }
      >
        {
          WORKSPACE_STEPS.map(
            (
              step
            ) => {
              const active =
                step.id ===
                activeStep;

              const enabled =
                isEnabled(
                  step.id
                );

              const showIntervention =
                step.id ===
                  "analyses" &&
                interventionCount >
                  0;


              return (
                <button
                  key={
                    step.id
                  }
                  type="button"
                  aria-disabled={
                    !enabled
                  }
                  aria-current={
                    active
                      ? "step"
                      : undefined
                  }
                  className={
                    `${styles.workspaceStep} ${
                      active
                        ? styles.workspaceStepActive
                        : ""
                    } ${
                      !enabled
                        ? styles.workspaceStepDisabled
                        : ""
                    }`
                  }
                  onClick={
                    () => {
                      if (
                        !enabled
                      ) {
                        return;
                      }


                      onStepChange(
                        step.id
                      );
                    }
                  }
                >
                  <span
                    aria-hidden="true"
                    className={
                      styles.workspaceStepNumber
                    }
                  >
                    {
                      step.shortLabel
                    }
                  </span>

                  <span
                    className={
                      styles.workspaceStepLabel
                    }
                  >
                    {
                      step.label
                    }
                  </span>

                  {
                    showIntervention
                      ? (
                          <span
                            className={
                              styles.workspaceInterventionDot
                            }
                            title={
                              `${interventionCount} intervention(s) requise(s)`
                            }
                          />
                        )
                      : null
                  }
                </button>
              );
            }
          )
        }
      </div>


      <Link
        href="/observability"
        target="_blank"
        rel="noreferrer"
        title="Ouvrir l’observabilité IA locale dans un nouvel onglet"
        className={
          styles.observabilityLink
        }
      >
        <span
          aria-hidden="true"
          className={
            styles.observabilityIcon
          }
        >
          AI
        </span>

        <span>
          Observabilité
          {" ↗"}
        </span>
      </Link>
    </nav>
  );
}


export default function WorkspaceClient() {
  const [
    objective,
    setObjective,
  ] =
    useState(
      ""
    );


  const [
    documents,
    setDocuments,
  ] =
    useState<
      File[]
    >(
      []
    );


  const [
    datasetFiles,
    setDatasetFiles,
  ] =
    useState<
      File[]
    >(
      []
    );


  const [
    ingestion,
    setIngestion,
  ] =
    useState<
      MultiDatasetIngestion |
      null
    >(
      null
    );


  const [
    preparationSession,
    setPreparationSession,
  ] =
    useState<
      PreparationSessionView |
      null
    >(
      null
    );


  const [
    preparationSessionLoading,
    setPreparationSessionLoading,
  ] =
    useState(
      false
    );


  const [
    preparationSessionError,
    setPreparationSessionError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    qualityReport,
    setQualityReport,
  ] =
    useState<
      DataQualityReportView |
      null
    >(
      null
    );


  const [
    qualityLoading,
    setQualityLoading,
  ] =
    useState(
      false
    );


  const [
    qualityError,
    setQualityError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    cleaningPlan,
    setCleaningPlan,
  ] =
    useState<
      CleaningPlanView |
      null
    >(
      null
    );


  const [
    cleaningPlanLoading,
    setCleaningPlanLoading,
  ] =
    useState(
      false
    );


  const [
    cleaningPlanError,
    setCleaningPlanError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    selectedCleaningActionIds,
    setSelectedCleaningActionIds,
  ] =
    useState<
      string[]
    >(
      []
    );


  const [
    cleaningApplyLoading,
    setCleaningApplyLoading,
  ] =
    useState(
      false
    );


  const [
    cleaningApplyError,
    setCleaningApplyError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    preparedExportLoading,
    setPreparedExportLoading,
  ] =
    useState(
      false
    );


  const [
    preparedExportError,
    setPreparedExportError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    cleaningExecution,
    setCleaningExecution,
  ] =
    useState<
      CleaningExecutionView |
      null
    >(
      null
    );


  const [
    appliedCleaningActionIds,
    setAppliedCleaningActionIds,
  ] =
    useState<
      string[]
    >(
      []
    );


  const [
    semanticReview,
    setSemanticReview,
  ] =
    useState<
      SemanticReviewReportView |
      null
    >(
      null
    );


  const [
    semanticReviewLoading,
    setSemanticReviewLoading,
  ] =
    useState(
      false
    );


  const [
    semanticReviewError,
    setSemanticReviewError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    semanticCleaningPlan,
    setSemanticCleaningPlan,
  ] =
    useState<
      SemanticCleaningPlanView |
      null
    >(
      null
    );


  const [
    semanticPlanLoading,
    setSemanticPlanLoading,
  ] =
    useState(
      false
    );


  const [
    semanticPlanError,
    setSemanticPlanError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    selectedSemanticActionIds,
    setSelectedSemanticActionIds,
  ] =
    useState<
      string[]
    >(
      []
    );


  const [
    semanticCanonicalValues,
    setSemanticCanonicalValues,
  ] =
    useState<
      Record<
        string,
        string
      >
    >(
      {}
    );


  const [
    semanticApplyLoading,
    setSemanticApplyLoading,
  ] =
    useState(
      false
    );


  const [
    semanticApplyError,
    setSemanticApplyError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    semanticCleaningExecution,
    setSemanticCleaningExecution,
  ] =
    useState<
      SemanticCleaningExecutionView |
      null
    >(
      null
    );


  const [
    appliedSemanticChoices,
    setAppliedSemanticChoices,
  ] =
    useState<
      SemanticCleaningChoiceView[]
    >(
      []
    );


  const [
    confirmedSemanticIssueIds,
    setConfirmedSemanticIssueIds,
  ] =
    useState<
      string[]
    >(
      []
    );


  const [
    semanticManualResolutionNotes,
    setSemanticManualResolutionNotes,
  ] =
    useState<
      Record<
        string,
        string
      >
    >(
      {}
    );


  const [
    semanticConfirmation,
    setSemanticConfirmation,
  ] =
    useState<
      SemanticConfirmationReportView |
      null
    >(
      null
    );


  const [
    semanticConfirmationLoading,
    setSemanticConfirmationLoading,
  ] =
    useState(
      false
    );


  const [
    semanticConfirmationError,
    setSemanticConfirmationError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    finalValidationLoading,
    setFinalValidationLoading,
  ] =
    useState(
      false
    );


  const [
    finalValidationError,
    setFinalValidationError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    activeDatasetIndex,
    setActiveDatasetIndex,
  ] =
    useState(
      0
    );


  const [
    ingestionLoading,
    setIngestionLoading,
  ] =
    useState(
      false
    );


  const [
    analysisLoading,
    setAnalysisLoading,
  ] =
    useState(
      false
    );


  const [
    aiPlanLoading,
    setAiPlanLoading,
  ] =
    useState(
      false
    );


  const [
    aiPlanReport,
    setAiPlanReport,
  ] =
    useState<
      AIPlannerReportView |
      null
    >(
      null
    );


  const [
    aiPlanError,
    setAiPlanError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    aiNativeLoading,
    setAiNativeLoading,
  ] =
    useState(
      false
    );


  const [
    aiNativeReport,
    setAiNativeReport,
  ] =
    useState<
      AINativePipelineReportView |
      null
    >(
      null
    );


  const [
    aiNativeError,
    setAiNativeError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    pdfExportLoading,
    setPdfExportLoading,
  ] =
    useState(
      false
    );


  const [
    report,
    setReport,
  ] =
    useState<
      RoutedUnifiedAnalysisReportView |
      null
    >(
      null
    );


  const [
    ragReport,
    setRagReport,
  ] =
    useState<
      RagContextReport |
      null
    >(
      null
    );


  const [
    documentSummary,
    setDocumentSummary,
  ] =
    useState<
      DocumentSummaryView |
      null
    >(
      null
    );


  const [
    requestedPlan,
    setRequestedPlan,
  ] =
    useState<
      RequestedPlanView |
      null
    >(
      null
    );


  const [
    error,
    setError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    activeStep,
    setActiveStep,
  ] =
    useState<
      WorkspaceStep
    >(
      "data"
    );


  const [
    activePreparationStep,
    setActivePreparationStep,
  ] =
    useState<
      PreparationSubstep
    >(
      "diagnostic"
    );


  useEffect(
    () => {
      setActivePreparationStep(
        "diagnostic"
      );
    },
    [
      preparationSession
        ?.workflow_id,
    ]
  );


  const activeManifest:
    DatasetManifest |
    null =
      ingestion
        ?.datasets[
          activeDatasetIndex
        ] ??
      null;


  const signalKpis =
    useMemo(
      () =>
        report
          ? buildSignalKpis(
              report
            )
          : [],
      [
        report,
      ]
    );


  const ragContextByAnalysisId =
    useMemo(
      () => {
        const lookup =
          new Map<
            string,
            FindingRagContext
          >();


        for (
          const context
          of ragReport?.contexts ??
          []
        ) {
          lookup.set(
            context.analysis_id,
            context
          );
        }


        return lookup;
      },
      [
        ragReport,
      ]
    );


  const submitDisabled =
    Boolean(
      analysisLoading ||
      ingestionLoading ||
      preparationSessionLoading ||
      cleaningApplyLoading ||
      datasetFiles.length ===
        0 ||
      !preparationSession
        ?.snapshot
        .ready_for_analysis
    );


  const aiPlanPreviewDisabled =
    Boolean(
      aiPlanLoading ||
      aiNativeLoading ||
      ingestionLoading ||
      datasetFiles.length ===
        0 ||
      !objective.trim()
    );


  const aiNativeRunDisabled =
    Boolean(
      aiNativeLoading ||
      aiPlanLoading ||
      ingestionLoading ||
      cleaningApplyLoading ||
      datasetFiles.length ===
        0 ||
      !objective.trim()
    );


  const plannerModelForUi =
    aiNativeReport
      ?.planner_model ??
    aiPlanReport
      ?.model ??
    null;


  const activePlannerUi =
    plannerUiCopy(
      plannerModelForUi
    );


  const dataReady =
    Boolean(
      ingestion &&
      datasetFiles.length >
        0
    );


  const reportReady =
    report !==
    null;


  const interventionCount =
    requestedPlan?.requests.filter(
      (
        request
      ) =>
        request.status !==
        "ready"
    ).length ??
    0;


  async function loadCleaningPlan(
    files:
      File[],

    workflowId:
      string
  ) {
    setCleaningPlanLoading(
      true
    );

    setCleaningPlanError(
      null
    );

    setCleaningPlan(
      null
    );

    setSelectedCleaningActionIds(
      []
    );

    setCleaningExecution(
      null
    );

    setAppliedCleaningActionIds(
      []
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of files
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "workflow_id",
        workflowId
      );


      const response =
        await fetch(
          `${API_URL}/preparation/cleaning-plan`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedPlan =
        payload as
          CleaningPlanView;


      setCleaningPlan(
        typedPlan
      );


      setSelectedCleaningActionIds(
        typedPlan.actions
          .filter(
            (
              action
            ) =>
              action.safe_candidate
          )
          .map(
            (
              action
            ) =>
              action.action_id
          )
      );


      const synchronizedSession =
        await getPreparationSession(
          workflowId
        );


      setPreparationSession(
        synchronizedSession
      );
    } catch (
      caughtError
    ) {
      setCleaningPlan(
        null
      );

      setCleaningPlanError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de construire le plan de nettoyage."
      );
    } finally {
      setCleaningPlanLoading(
        false
      );
    }
  }


  function handleToggleCleaningAction(
    actionId:
      string
  ) {
    if (
      cleaningExecution
    ) {
      return;
    }


    setSelectedCleaningActionIds(
      (
        current
      ) =>
        current.includes(
          actionId
        )
          ? current.filter(
              (
                value
              ) =>
                value !==
                actionId
            )
          : [
              ...current,
              actionId,
            ]
    );

    setCleaningApplyError(
      null
    );

    resetSemanticPreparation();
  }


  async function handleApplyCleaning() {
    if (
      datasetFiles.length ===
      0
    ) {
      setCleaningApplyError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    if (
      cleaningPlan ===
      null
    ) {
      setCleaningApplyError(
        "Aucun plan de nettoyage n’est disponible."
      );

      return;
    }


    if (
      !preparationSession
    ) {
      setCleaningApplyError(
        "La session de préparation est indisponible."
      );

      return;
    }


    if (
      selectedCleaningActionIds.length ===
      0
    ) {
      setCleaningApplyError(
        "Aucune correction sûre n’est sélectionnée."
      );

      return;
    }


    setCleaningApplyLoading(
      true
    );

    setCleaningApplyError(
      null
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "approved_action_ids_json",
        JSON.stringify(
          selectedCleaningActionIds
        )
      );


      formData.append(
        "workflow_id",
        preparationSession.workflow_id
      );


      const response =
        await fetch(
          `${API_URL}/preparation/cleaning-apply`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedPayload =
        payload as
          CleaningApplyResponseView;


      setQualityReport(
        typedPayload
          .quality_report
      );

      setCleaningPlan(
        typedPayload
          .cleaning_plan
      );

      setCleaningExecution(
        typedPayload
          .execution
      );


      const appliedIds =
        typedPayload
          .execution
          .action_results
          .filter(
            (
              result
            ) =>
              result.status ===
              "applied"
          )
          .map(
            (
              result
            ) =>
              result.action_id
          );


      setAppliedCleaningActionIds(
        appliedIds
      );

      setSelectedCleaningActionIds(
        appliedIds
      );


      resetSemanticPreparation();


      const synchronizedSession =
        await getPreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        synchronizedSession
      );


      if (
        synchronizedSession
          .snapshot
          .next_stage ===
          "clean"
      ) {
        setActivePreparationStep(
          "semantic"
        );
      } else if (
        synchronizedSession
          .snapshot
          .next_stage ===
          "validate"
      ) {
        setActivePreparationStep(
          "validation"
        );
      }


      setReport(
        null
      );

      setRagReport(
        null
      );

      setDocumentSummary(
        null
      );

      setRequestedPlan(
        null
      );

      setAiNativeReport(
        null
      );

      setAiNativeError(
        null
      );
    } catch (
      caughtError
    ) {
      setCleaningExecution(
        null
      );

      setAppliedCleaningActionIds(
        []
      );

      setCleaningApplyError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Le nettoyage contrôlé a échoué."
      );
    } finally {
      setCleaningApplyLoading(
        false
      );
    }
  }


  function resetSemanticPreparation() {
    setSemanticReview(
      null
    );

    setSemanticReviewError(
      null
    );

    setSemanticCleaningPlan(
      null
    );

    setSemanticPlanError(
      null
    );

    setSelectedSemanticActionIds(
      []
    );

    setSemanticCanonicalValues(
      {}
    );

    setSemanticCleaningExecution(
      null
    );

    setAppliedSemanticChoices(
      []
    );

    setSemanticApplyError(
      null
    );

    setConfirmedSemanticIssueIds(
      []
    );

    setSemanticManualResolutionNotes(
      {}
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationLoading(
      false
    );

    setSemanticConfirmationError(
      null
    );

    setFinalValidationLoading(
      false
    );

    setFinalValidationError(
      null
    );
  }


  async function buildSemanticCleaningPlan(
    review:
      SemanticReviewReportView
  ) {
    if (
      preparationSession ===
      null
    ) {
      setSemanticPlanError(
        "Aucune session de préparation active."
      );

      return;
    }


    setSemanticPlanLoading(
      true
    );

    setSemanticPlanError(
      null
    );

    setSemanticCleaningPlan(
      null
    );

    setSelectedSemanticActionIds(
      []
    );

    setSemanticCleaningExecution(
      null
    );

    setAppliedSemanticChoices(
      []
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "workflow_id",
        preparationSession.workflow_id
      );


      if (
        appliedCleaningActionIds.length >
        0
      ) {
        formData.append(
          "approved_action_ids_json",
          JSON.stringify(
            appliedCleaningActionIds
          )
        );
      }


      formData.append(
        "semantic_decisions_json",
        JSON.stringify(
          review.decisions
        )
      );


      const response =
        await fetch(
          `${API_URL}/preparation/semantic-cleaning-plan`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedPlan =
        payload as
          SemanticCleaningPlanView;


      setSemanticCleaningPlan(
        typedPlan
      );


      const canonicalValues: Record<
        string,
        string
      > = {};


      for (
        const action
        of typedPlan.actions
      ) {
        canonicalValues[
          action.action_id
        ] =
          action
            .suggested_canonical_value;
      }


      setSemanticCanonicalValues(
        canonicalValues
      );
    } catch (
      caughtError
    ) {
      setSemanticCleaningPlan(
        null
      );

      setSemanticPlanError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de reconstruire le plan sémantique."
      );
    } finally {
      setSemanticPlanLoading(
        false
      );
    }
  }


  async function handleRunSemanticReview() {
    if (
      datasetFiles.length ===
      0
    ) {
      setSemanticReviewError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    if (
      preparationSession ===
      null
    ) {
      setSemanticReviewError(
        "Aucune session de préparation active."
      );

      return;
    }


    const deterministicCleaningReady =
      cleaningPlan !==
        null &&
      (
        cleaningPlan.action_count ===
          0 ||
        cleaningExecution !==
          null
      );


    if (
      !deterministicCleaningReady
    ) {
      setSemanticReviewError(
        "Terminez d’abord l’étape de nettoyage déterministe."
      );

      return;
    }


    setSemanticReviewLoading(
      true
    );

    setSemanticReviewError(
      null
    );

    setSemanticReview(
      null
    );

    setSemanticCleaningPlan(
      null
    );

    setSemanticPlanError(
      null
    );

    setSelectedSemanticActionIds(
      []
    );

    setSemanticCanonicalValues(
      {}
    );

    setSemanticCleaningExecution(
      null
    );

    setAppliedSemanticChoices(
      []
    );

    setConfirmedSemanticIssueIds(
      []
    );

    setSemanticManualResolutionNotes(
      {}
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    setFinalValidationError(
      null
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "workflow_id",
        preparationSession.workflow_id
      );


      if (
        appliedCleaningActionIds.length >
        0
      ) {
        formData.append(
          "approved_action_ids_json",
          JSON.stringify(
            appliedCleaningActionIds
          )
        );
      }


      formData.append(
        "model",
        "gemma3:4b"
      );


      const response =
        await fetch(
          `${API_URL}/preparation/semantic-review`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedReview =
        payload as
          SemanticReviewReportView;


      setSemanticReview(
        typedReview
      );


      const synchronizedSession =
        await getPreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        synchronizedSession
      );


      await buildSemanticCleaningPlan(
        typedReview
      );
    } catch (
      caughtError
    ) {
      setSemanticReview(
        null
      );

      setSemanticReviewError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La revue sémantique locale a échoué."
      );
    } finally {
      setSemanticReviewLoading(
        false
      );
    }
  }


  function handleSetSemanticDecision(
    actionId:
      string,

    shouldMerge:
      boolean
  ) {
    if (
      semanticCleaningExecution
    ) {
      return;
    }


    setSelectedSemanticActionIds(
      (
        current
      ) => {
        if (
          shouldMerge
        ) {
          return current.includes(
            actionId
          )
            ? current
            : [
                ...current,
                actionId,
              ];
        }


        return current.filter(
          (
            value
          ) =>
            value !==
            actionId
        );
      }
    );

    setSemanticApplyError(
      null
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    setFinalValidationError(
      null
    );
  }


  function handleSemanticCanonicalChange(
    actionId:
      string,

    canonicalValue:
      string
  ) {
    if (
      semanticCleaningExecution
    ) {
      return;
    }


    setSemanticCanonicalValues(
      (
        current
      ) => ({
        ...current,

        [actionId]:
          canonicalValue,
      })
    );

    setSemanticApplyError(
      null
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    setFinalValidationError(
      null
    );
  }


  async function handleApplySemanticCleaning() {
    if (
      semanticReview ===
      null
      ||
      semanticCleaningPlan ===
      null
    ) {
      setSemanticApplyError(
        "Aucun plan sémantique n’est disponible."
      );

      return;
    }


    if (
      preparationSession ===
      null
    ) {
      setSemanticApplyError(
        "Aucune session de préparation active."
      );

      return;
    }


    if (
      selectedSemanticActionIds.length ===
      0
    ) {
      setSemanticApplyError(
        "Sélectionnez au moins une fusion à appliquer."
      );

      return;
    }


    let choices:
      SemanticCleaningChoiceView[];


    try {
      choices =
        selectedSemanticActionIds.map(
          (
            actionId
          ) => {
            const canonicalValue =
              semanticCanonicalValues[
                actionId
              ];


            if (
              !canonicalValue
            ) {
              throw new Error(
                "Une valeur canonique manque pour une fusion sélectionnée."
              );
            }


            return {
              action_id:
                actionId,

              canonical_value:
                canonicalValue,
            };
          }
        );
    } catch (
      caughtError
    ) {
      setSemanticApplyError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Une valeur canonique est invalide."
      );

      return;
    }


    setSemanticApplyLoading(
      true
    );

    setSemanticApplyError(
      null
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "workflow_id",
        preparationSession.workflow_id
      );


      if (
        appliedCleaningActionIds.length >
        0
      ) {
        formData.append(
          "approved_action_ids_json",
          JSON.stringify(
            appliedCleaningActionIds
          )
        );
      }


      formData.append(
        "semantic_decisions_json",
        JSON.stringify(
          semanticReview.decisions
        )
      );


      formData.append(
        "approved_semantic_choices_json",
        JSON.stringify(
          choices
        )
      );


      const response =
        await fetch(
          `${API_URL}/preparation/semantic-cleaning-apply`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedPayload =
        payload as
          SemanticCleaningApplyResponseView;


      setSemanticCleaningPlan(
        typedPayload.plan
      );

      setSemanticCleaningExecution(
        typedPayload.execution
      );

      setAppliedSemanticChoices(
        choices
      );

      setSemanticConfirmation(
        null
      );

      setSemanticConfirmationError(
        null
      );

      setFinalValidationError(
        null
      );


      setReport(
        null
      );

      setRagReport(
        null
      );

      setDocumentSummary(
        null
      );

      setRequestedPlan(
        null
      );

      setAiNativeReport(
        null
      );

      setAiNativeError(
        null
      );
    } catch (
      caughtError
    ) {
      setSemanticCleaningExecution(
        null
      );

      setAppliedSemanticChoices(
        []
      );

      setSemanticApplyError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Le nettoyage sémantique a échoué."
      );
    } finally {
      setSemanticApplyLoading(
        false
      );
    }
  }


  function handleToggleSemanticIssueConfirmation(
    issueId:
      string,

    checked:
      boolean
  ) {
    setConfirmedSemanticIssueIds(
      (
        current
      ) => {
        if (
          checked
        ) {
          return current.includes(
            issueId
          )
            ? current
            : [
                ...current,
                issueId,
              ];
        }


        return current.filter(
          (
            value
          ) =>
            value !==
            issueId
        );
      }
    );


    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    setFinalValidationError(
      null
    );
  }


  function handleSemanticManualResolutionChange(
    issueId:
      string,

    note:
      string
  ) {
    setSemanticManualResolutionNotes(
      (
        current
      ) => ({
        ...current,

        [issueId]:
          note,
      })
    );


    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    setFinalValidationError(
      null
    );
  }


  async function handleConfirmSemanticReview() {
    if (
      semanticReview ===
      null
    ) {
      setSemanticConfirmationError(
        "Aucune revue sémantique n’est disponible."
      );

      return;
    }


    if (
      preparationSession ===
      null
    ) {
      setSemanticConfirmationError(
        "Aucune session de préparation active."
      );

      return;
    }


    if (
      semanticReview.decisions.length ===
      0
    ) {
      setSemanticConfirmationError(
        "Aucune décision sémantique confirmable n’est disponible. Les problèmes protégés doivent être examinés manuellement."
      );

      return;
    }


    const manualResolutions =
      semanticReview.decisions
        .filter(
          (
            decision
          ) =>
            (
              decision.verdict ===
                "abstain" ||
              decision.verdict ===
                "flag_for_review"
            ) &&
            confirmedSemanticIssueIds.includes(
              decision.issue_id
            )
        )
        .map(
          (
            decision
          ) => ({
            issue_id:
              decision.issue_id,

            note:
              semanticManualResolutionNotes[
                decision.issue_id
              ]?.trim() ??
              "",
          })
        )
        .filter(
          (
            resolution
          ) =>
            resolution.note.length >=
            3
        );


    setSemanticConfirmationLoading(
      true
    );

    setSemanticConfirmationError(
      null
    );

    setFinalValidationError(
      null
    );


    try {
      const response =
        await confirmSemanticReview(
          {
            datasetFiles,

            workflowId:
              preparationSession.workflow_id,

            semanticDecisions:
              semanticReview.decisions,

            confirmedIssueIds:
              confirmedSemanticIssueIds,

            approvedSemanticChoices:
              appliedSemanticChoices,

            manualResolutions,

            approvedCleaningActionIds:
              appliedCleaningActionIds,
          }
        );


      setSemanticConfirmation(
        response.confirmation
      );


      const synchronizedSession =
        await getPreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        synchronizedSession
      );


      if (
        synchronizedSession
          .snapshot
          .next_stage ===
          "validate"
      ) {
        setActivePreparationStep(
          "validation"
        );
      }
    } catch (
      caughtError
    ) {
      if (
        caughtError instanceof
        SemanticConfirmationApiError
      ) {
        const payload =
          caughtError.detail;


        if (
          payload &&
          typeof payload ===
            "object"
        ) {
          const outer =
            payload as Record<
              string,
              unknown
            >;

          const detail =
            outer.detail;


          if (
            detail &&
            typeof detail ===
              "object" &&
            !Array.isArray(
              detail
            )
          ) {
            const detailRecord =
              detail as Record<
                string,
                unknown
              >;

            const confirmation =
              detailRecord.confirmation;


            if (
              confirmation &&
              typeof confirmation ===
                "object"
            ) {
              setSemanticConfirmation(
                confirmation as
                  SemanticConfirmationReportView
              );
            }
          }
        }
      }


      setSemanticConfirmationError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La confirmation de la revue sémantique a échoué."
      );


      try {
        const synchronizedSession =
          await getPreparationSession(
            preparationSession.workflow_id
          );

        setPreparationSession(
          synchronizedSession
        );
      } catch {
        // Preserve the original confirmation error.
      }
    } finally {
      setSemanticConfirmationLoading(
        false
      );
    }
  }


  async function handleValidatePreparation() {
    if (
      preparationSession ===
      null
    ) {
      setFinalValidationError(
        "Aucune session de préparation active."
      );

      return;
    }


    setFinalValidationLoading(
      true
    );

    setFinalValidationError(
      null
    );


    try {
      const validatedSession =
        await validatePreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        validatedSession
      );
    } catch (
      caughtError
    ) {
      setFinalValidationError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La validation finale de la préparation a échoué."
      );


      try {
        const synchronizedSession =
          await getPreparationSession(
            preparationSession.workflow_id
          );

        setPreparationSession(
          synchronizedSession
        );
      } catch {
        // Preserve the original validation error.
      }
    } finally {
      setFinalValidationLoading(
        false
      );
    }
  }


  async function handleRefreshPreparationSession() {
    if (
      !preparationSession
    ) {
      return;
    }


    setPreparationSessionLoading(
      true
    );

    setPreparationSessionError(
      null
    );


    try {
      const refreshedSession =
        await getPreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        refreshedSession
      );
    } catch (
      caughtError
    ) {
      setPreparationSessionError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible d’actualiser la session de préparation."
      );
    } finally {
      setPreparationSessionLoading(
        false
      );
    }
  }


  async function handleDatasetsChange(
    event:
      ChangeEvent<HTMLInputElement>
  ) {
    const files =
      Array.from(
        event.target.files ??
        []
      );


    setDatasetFiles(
      files
    );

    setIngestion(
      null
    );

    setPreparationSession(
      null
    );

    setPreparationSessionError(
      null
    );

    setPreparationSessionLoading(
      false
    );

    setQualityReport(
      null
    );

    setQualityError(
      null
    );

    setCleaningPlan(
      null
    );

    setCleaningPlanError(
      null
    );

    setSelectedCleaningActionIds(
      []
    );

    setCleaningExecution(
      null
    );

    setAppliedCleaningActionIds(
      []
    );

    setCleaningApplyError(
      null
    );

    setPreparedExportError(
      null
    );

    resetSemanticPreparation();

    setReport(
      null
    );

    setRagReport(
      null
    );

    setDocumentSummary(
      null
    );

    setRequestedPlan(
      null
    );

    setAiPlanReport(
      null
    );

    setAiPlanError(
      null
    );

    setAiNativeReport(
      null
    );

    setAiNativeError(
      null
    );

    setError(
      null
    );

    setActiveDatasetIndex(
      0
    );


    setActiveStep(
      "data"
    );


    if (
      files.length ===
      0
    ) {
      return;
    }


    setIngestionLoading(
      true
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of files
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      const response =
        await fetch(
          `${API_URL}/ingestion/datasets`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedIngestion =
        payload as
          MultiDatasetIngestion;


      setIngestion(
        typedIngestion
      );


      let createdPreparationSession:
        PreparationSessionView |
        null =
          null;


      setPreparationSessionLoading(
        true
      );

      setPreparationSessionError(
        null
      );


      try {
        const createdSession =
          await createPreparationSession(
            typedIngestion.datasets.map(
              (
                dataset
              ) =>
                dataset.dataset_id
            )
          );


        createdPreparationSession =
          createdSession;


        setPreparationSession(
          createdSession
        );
      } catch (
        sessionCaughtError
      ) {
        setPreparationSession(
          null
        );

        setPreparationSessionError(
          sessionCaughtError
            instanceof Error
            ? sessionCaughtError.message
            : "Impossible de créer la session de préparation."
        );
      } finally {
        setPreparationSessionLoading(
          false
        );
      }


      setQualityLoading(
        true
      );

      setQualityError(
        null
      );


      try {
        const qualityFormData =
          new FormData();


        for (
          const file
          of files
        ) {
          qualityFormData.append(
            "dataset_files",
            file
          );
        }


        if (
          !createdPreparationSession
        ) {
          throw new Error(
            "La session de préparation n’a pas pu être créée avant le diagnostic qualité."
          );
        }


        qualityFormData.append(
          "workflow_id",
          createdPreparationSession.workflow_id
        );


        const qualityResponse =
          await fetch(
            `${API_URL}/preparation/quality`,
            {
              method:
                "POST",

              body:
                qualityFormData,
            }
          );


        const qualityPayload =
          await qualityResponse.json();


        if (
          !qualityResponse.ok
        ) {
          const detail =
            typeof qualityPayload.detail ===
            "string"
              ? qualityPayload.detail
              : JSON.stringify(
                  qualityPayload.detail ??
                  qualityPayload
                );


          throw new Error(
            detail
          );
        }


        setQualityReport(
          qualityPayload as
            DataQualityReportView
        );


        const synchronizedSession =
          await getPreparationSession(
            createdPreparationSession.workflow_id
          );


        setPreparationSession(
          synchronizedSession
        );
      } catch (
        qualityCaughtError
      ) {
        setQualityReport(
          null
        );

        setQualityError(
          qualityCaughtError
            instanceof Error
            ? qualityCaughtError.message
            : "Impossible d’exécuter le diagnostic qualité."
        );
      } finally {
        setQualityLoading(
          false
        );
      }


      if (
        createdPreparationSession
      ) {
        await loadCleaningPlan(
          files,
          createdPreparationSession.workflow_id
        );
      }
    } catch (
      caughtError
    ) {
      setDatasetFiles(
        []
      );

      setIngestion(
        null
      );

      setPreparationSession(
        null
      );

      setPreparationSessionError(
        null
      );

      setPreparationSessionLoading(
        false
      );

      setQualityReport(
        null
      );

      setQualityError(
        null
      );

      setQualityLoading(
        false
      );

      setCleaningPlan(
        null
      );

      setCleaningPlanError(
        null
      );

      setSelectedCleaningActionIds(
        []
      );

      setCleaningExecution(
        null
      );

      setAppliedCleaningActionIds(
        []
      );

      setCleaningApplyError(
        null
      );

      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de lire les fichiers."
      );
    } finally {
      setIngestionLoading(
        false
      );
    }
  }


  function handleDocumentsChange(
    event:
      ChangeEvent<HTMLInputElement>
  ) {
    const files =
      Array.from(
        event.target.files ??
        []
      );


    setDocuments(
      files
    );

    setReport(
      null
    );

    setRagReport(
      null
    );

    setDocumentSummary(
      null
    );

    setRequestedPlan(
      null
    );

    setError(
      null
    );
  }


  async function handleAiPlanningPreview() {
    if (
      datasetFiles.length ===
      0
    ) {
      setAiPlanError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    const normalizedObjective =
      objective.trim();


    if (
      !normalizedObjective
    ) {
      setAiPlanError(
        "Décrivez d’abord ce que vous souhaitez comprendre."
      );

      return;
    }


    setAiPlanLoading(
      true
    );

    setAiPlanError(
      null
    );

    setAiPlanReport(
      null
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "objective",
        normalizedObjective
      );

      formData.append(
        "planner_model",
        "gemma3:4b"
      );


      const response =
        await fetch(
          `${API_URL}/planning/ai-preview`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      setAiPlanReport(
        payload as
          AIPlannerReportView
      );
    } catch (
      caughtError
    ) {
      setAiPlanReport(
        null
      );

      setAiPlanError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La préparation du plan analytique a échoué."
      );
    } finally {
      setAiPlanLoading(
        false
      );
    }
  }


  async function handleExportPreparedData() {
    if (
      datasetFiles.length ===
      0
    ) {
      setPreparedExportError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    if (
      appliedCleaningActionIds.length ===
      0
    ) {
      setPreparedExportError(
        "Appliquez d’abord les corrections sélectionnées."
      );

      return;
    }


    setPreparedExportLoading(
      true
    );

    setPreparedExportError(
      null
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "approved_action_ids_json",
        JSON.stringify(
          appliedCleaningActionIds
        )
      );


      const response =
        await fetch(
          `${API_URL}/preparation/cleaning-export`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      if (
        !response.ok
      ) {
        let detail =
          "L’export des données préparées a échoué.";


        try {
          const payload =
            await response.json();


          detail =
            typeof payload.detail ===
            "string"
              ? payload.detail
              : JSON.stringify(
                  payload.detail ??
                  payload
                );
        } catch {
          // Keep the generic message when the server
          // did not return JSON.
        }


        throw new Error(
          detail
        );
      }


      const blob =
        await response.blob();


      const disposition =
        response.headers.get(
          "content-disposition"
        ) ??
        "";


      const filenameMatch =
        disposition.match(
          /filename="?([^"]+)"?/i
        );


      const fallbackFilename =
        datasetFiles.length ===
        1
          ? `${datasetFiles[0].name.replace(
              /\.csv$/i,
              ""
            )}_prepared.csv`
          : "datalens_prepared_datasets.zip";


      const filename =
        filenameMatch?.[1] ??
        fallbackFilename;


      const objectUrl =
        URL.createObjectURL(
          blob
        );


      const link =
        document.createElement(
          "a"
        );


      link.href =
        objectUrl;

      link.download =
        filename;

      document.body.appendChild(
        link
      );

      link.click();

      link.remove();


      URL.revokeObjectURL(
        objectUrl
      );
    } catch (
      caughtError
    ) {
      setPreparedExportError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "L’export des données préparées a échoué."
      );
    } finally {
      setPreparedExportLoading(
        false
      );
    }
  }


  async function handleAiNativeRun() {
    if (
      datasetFiles.length ===
      0
    ) {
      setAiNativeError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    const normalizedObjective =
      objective.trim();


    if (
      !normalizedObjective
    ) {
      setAiNativeError(
        "Décrivez d’abord ce que vous souhaitez comprendre."
      );

      return;
    }


    setAiNativeLoading(
      true
    );

    setAiNativeError(
      null
    );

    setAiNativeReport(
      null
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "objective",
        normalizedObjective
      );

      formData.append(
        "planner_model",
        "gemma3:4b"
      );

      formData.append(
        "tool_model",
        "qwen2.5:1.5b-instruct"
      );


      if (
        appliedCleaningActionIds.length >
        0
      ) {
        formData.append(
          "approved_action_ids_json",
          JSON.stringify(
            appliedCleaningActionIds
          )
        );
      }



      if (
        semanticCleaningExecution &&
        semanticReview &&
        appliedSemanticChoices.length >
        0
      ) {
        formData.append(
          "semantic_decisions_json",
          JSON.stringify(
            semanticReview.decisions
          )
        );

        formData.append(
          "approved_semantic_choices_json",
          JSON.stringify(
            appliedSemanticChoices
          )
        );
      }


      const response =
        await fetch(
          `${API_URL}/planning/ai-native-run`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedPayload =
        payload as
          AINativePipelineReportView;


      setAiNativeReport(
        typedPayload
      );


      setAiPlanReport(
        typedPayload.planner
      );
    } catch (
      caughtError
    ) {
      setAiNativeReport(
        null
      );

      setAiNativeError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "L’analyse orchestrée a échoué."
      );
    } finally {
      setAiNativeLoading(
        false
      );
    }
  }


  async function handleSubmit(
    event:
      FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();


    if (
      datasetFiles.length ===
      0
    ) {
      setError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    if (
      !preparationSession
    ) {
      setError(
        "La session de préparation n’est pas disponible. Rechargez les données pour en créer une nouvelle."
      );

      setActiveStep(
        "preparation"
      );

      return;
    }


    if (
      !preparationSession
        .snapshot
        .ready_for_analysis
    ) {
      setError(
        "La préparation doit être validée par le backend avant de lancer l’analyse."
      );

      setActiveStep(
        "preparation"
      );

      return;
    }


    setAnalysisLoading(
      true
    );

    setError(
      null
    );

    setReport(
      null
    );

    setRagReport(
      null
    );

    setDocumentSummary(
      null
    );

    setRequestedPlan(
      null
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "workflow_id",
        preparationSession.workflow_id
      );


      if (
        objective.trim()
      ) {
        formData.append(
          "objective",
          objective.trim()
        );
      }


      if (
        appliedCleaningActionIds.length >
        0
      ) {
        formData.append(
          "approved_action_ids_json",
          JSON.stringify(
            appliedCleaningActionIds
          )
        );
      }



      if (
        semanticCleaningExecution &&
        semanticReview &&
        appliedSemanticChoices.length >
        0
      ) {
        formData.append(
          "semantic_decisions_json",
          JSON.stringify(
            semanticReview.decisions
          )
        );

        formData.append(
          "approved_semantic_choices_json",
          JSON.stringify(
            appliedSemanticChoices
          )
        );
      }


      const contextualized =
        documents.length >
        0;


      if (
        contextualized
      ) {
        for (
          const document
          of documents
        ) {
          formData.append(
            "document_files",
            document
          );
        }


        formData.append(
          "rag_top_k",
          "3"
        );

        formData.append(
          "embedding_model",
          "embeddinggemma"
        );
      }


      const endpoint =
        contextualized
          ? "/analysis/run-contextualized"
          : "/analysis/run";


      const response =
        await fetch(
          `${API_URL}${endpoint}`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      if (
        contextualized
      ) {
        const contextualizedPayload =
          payload as
            RoutedContextualizedAnalysisResponseView;


        setReport(
          contextualizedPayload
            .analysis
        );

        setRagReport(
          contextualizedPayload
            .rag
        );


        const contextualizedDetails =
          contextualizedPayload as
            RoutedContextualizedAnalysisResponseView &
            {
              document_summary?:
                unknown;

              requested_analysis_plan?:
                unknown;
            };


        setDocumentSummary(
          (
            contextualizedDetails
              .document_summary ??
            null
          ) as
            DocumentSummaryView |
            null
        );

        setRequestedPlan(
          (
            contextualizedDetails
              .requested_analysis_plan ??
            null
          ) as
            RequestedPlanView |
            null
        );
      } else {
        setReport(
          payload as
            RoutedUnifiedAnalysisReportView
        );

        setRagReport(
          null
        );

        setDocumentSummary(
          null
        );

        setRequestedPlan(
          null
        );
      }


      setActiveStep(
        "analyses"
      );
    } catch (
      caughtError
    ) {
      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "L’analyse a échoué."
      );
    } finally {
      setAnalysisLoading(
        false
      );
    }
  }


  async function handlePdfExport() {
    if (
      report ===
      null
    ) {
      setError(
        "Aucun rapport n’est disponible pour l’export PDF."
      );

      return;
    }


    setPdfExportLoading(
      true
    );

    setError(
      null
    );


    try {
      const response =
        await fetch(
          `${API_URL}/analysis/export-pdf`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                {
                  analysis:
                    report,

                  objective:
                    objective.trim() ||
                    null,

                  document_summary:
                    documentSummary,

                  requested_analysis_plan:
                    requestedPlan,

                  quality_report:
                    qualityReport,
                }
              ),
          }
        );


      if (
        !response.ok
      ) {
        const contentType =
          response.headers.get(
            "content-type"
          ) ??
          "";


        if (
          contentType.includes(
            "application/json"
          )
        ) {
          const payload =
            await response.json();

          const detail =
            typeof payload.detail ===
              "string"
              ? payload.detail
              : JSON.stringify(
                  payload.detail ??
                  payload
                );


          throw new Error(
            detail
          );
        }


        throw new Error(
          await response.text() ||
          "La génération du PDF a échoué."
        );
      }


      const blob =
        await response.blob();


      const objectUrl =
        URL.createObjectURL(
          blob
        );


      const date =
        new Date()
          .toISOString()
          .slice(
            0,
            10
          );


      const anchor =
        document.createElement(
          "a"
        );


      anchor.href =
        objectUrl;

      anchor.download =
        `datalens-rapport-${date}.pdf`;

      document.body.appendChild(
        anchor
      );

      anchor.click();

      anchor.remove();


      URL.revokeObjectURL(
        objectUrl
      );
    } catch (
      caughtError
    ) {
      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de générer le PDF."
      );
    } finally {
      setPdfExportLoading(
        false
      );
    }
  }


  return (
    <main
      className={
        styles.page
      }
    >
      <div
        className={
          styles.ambient
        }
        aria-hidden="true"
      />


      <header
        className={
          styles.header
        }
      >
        <div
          className={
            styles.brand
          }
        >
          <span
            className={
              styles.brandMark
            }
            aria-hidden="true"
          />

          <strong>
            DataLens
          </strong>
        </div>


        <div
          className={
            styles.privacyStatus
          }
        >
          <span
            className={
              styles.statusDot
            }
            aria-hidden="true"
          />

          Traitement local
          {" · "}
          données privées
        </div>
      </header>


      <div
        className={
          styles.shell
        }
      >
        <section
          className={
            styles.hero
          }
        >
          <h1>
            Analysez vos données

            <span>
              avec plus de clarté.
            </span>
          </h1>

          <p>
            Décrivez ce que vous souhaitez comprendre,
            ajoutez vos données et, si nécessaire, votre
            contexte métier. DataLens confie les calculs à
            Python et utilise l’IA locale pour comprendre la
            demande, préparer le plan et contextualiser les résultats.
          </p>
        </section>


        <WorkspaceNavigation
          activeStep={
            activeStep
          }
          onStepChange={
            setActiveStep
          }
          dataReady={
            dataReady
          }
          reportReady={
            reportReady
          }
          interventionCount={
            interventionCount
          }
        />


        <form
          className={
            styles.workspace
          }
          style={{
            display:
              (
                activeStep ===
                  "analyses" ||
                activeStep ===
                  "report"
              )
                ? "none"
                : undefined,
          }}
          onSubmit={
            handleSubmit
          }
          autoComplete="off"
        >
          <section
            className={
              `${styles.panel} ${styles.objectivePanel} ${styles.analysisRequestPanel}`
            }
            style={{
              display:
                activeStep ===
                  "documents"
                  ? undefined
                  : "none",
            }}
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
                  02 · Documents
                </span>

                <h2>
                  Votre demande d’analyse
                </h2>

                <p
                  className={
                    styles.sectionDescription
                  }
                >
                  Indiquez ce que vous voulez comprendre ou rechercher.
                  Cette demande sera traitée en priorité dans les résultats.
                </p>
              </div>

              <span
                className={
                  styles.priorityBadge
                }
              >
                Prioritaire
              </span>
            </div>


            <label
              className={
                styles.fieldLabel
              }
              htmlFor="objective"
            >
              Que voulez-vous comprendre ou rechercher ?
            </label>


            <textarea
              id="objective"
              className={
                styles.objectiveInput
              }
              value={
                objective
              }
              onChange={
                (
                  event
                ) => {
                  setObjective(
                    event.target.value
                  );

                  setAiPlanReport(
                    null
                  );

                  setAiPlanError(
                    null
                  );

                  setAiNativeReport(
                    null
                  );

                  setAiNativeError(
                    null
                  );
                }
              }
              placeholder="Ex. Valeurs atypiques · Comparer des groupes · Relation entre variables"
            />


            <p
              className={
                styles.helper
              }
            >
              Facultatif. Si vous laissez ce champ vide, DataLens explore les
              analyses compatibles automatiquement. Si vous formulez une demande,
              elle devient prioritaire ; Python conserve toujours la validation
              des datasets, des colonnes et des calculs.
            </p>
          </section>


          <div
            className={
              `${styles.sourceGrid} ${styles.sourceGridSingle}`
            }
            style={{
              display:
                (
                  activeStep ===
                    "data" ||
                  activeStep ===
                    "documents"
                )
                  ? undefined
                  : "none",
            }}
          >
            <section
              className={
                `${styles.panel} ${styles.contextPanel}`
              }
              style={{
                display:
                  activeStep ===
                    "documents"
                    ? undefined
                    : "none",
              }}
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
                    Contexte facultatif
                  </span>

                  <h2>
                    Documents métier
                  </h2>

                  <p
                    className={
                      styles.sectionDescription
                    }
                  >
                    Ajoutez des rapports, définitions, procédures ou briefs si
                    leur contenu peut aider à interpréter les résultats.
                  </p>
                </div>

                <span
                  className={
                    styles.optionalBadge
                  }
                >
                  Optionnel
                </span>
              </div>


              <label
                className={
                  `${styles.dropZone} ${
                    documents.length >
                      0
                      ? styles.dropZoneCompact
                      : ""
                  }`
                }
              >
                <input
                  className={
                    styles.fileInput
                  }
                  type="file"
                  multiple
                  accept=".pdf,.doc,.docx,.txt,.md,text/plain,application/pdf"
                  onChange={
                    handleDocumentsChange
                  }
                />

                <strong
                  className={
                    styles.dropLabel
                  }
                >
                  {
                    documents.length >
                      0
                      ? "Ajouter d’autres documents"
                      : "Ajouter des documents métier"
                  }
                </strong>

                <span
                  className={
                    styles.dropFormats
                  }
                >
                  PDF · DOCX · TXT · MD
                </span>

                <span
                  className={
                    styles.dropNote
                  }
                >
                  Sélectionnez uniquement les documents utiles au contexte.
                  Ils ne remplacent jamais les calculs Python.
                </span>
              </label>


              {
                documents.length >
                0
                  ? (
                      <>
                        <div
                          className={
                            styles.fileList
                          }
                        >
                          {
                            documents.map(
                              (
                                file
                              ) => (
                                <div
                                  className={
                                    styles.fileRow
                                  }
                                  key={
                                    `${file.name}-${file.size}`
                                  }
                                >
                                  <span
                                    className={
                                      styles.fileBadge
                                    }
                                  >
                                    DOC
                                  </span>

                                  <div
                                    className={
                                      styles.fileMeta
                                    }
                                  >
                                    <strong>
                                      {
                                        file.name
                                      }
                                    </strong>

                                    <small>
                                      {
                                        formatBytes(
                                          file.size
                                        )
                                      }
                                    </small>
                                  </div>
                                </div>
                              )
                            )
                          }
                        </div>

                        <p
                          className={
                            styles.helper
                          }
                        >
                          Ces documents seront
                          utilisés uniquement pour
                          contextualiser les résultats
                          calculés par le moteur
                          analytique.
                        </p>
                      </>
                    )
                  : (
                      <p
                        className={
                          styles.helper
                        }
                      >
                        Optionnel. Sans document,
                        DataLens exécute uniquement
                        l’analyse déterministe.
                      </p>
                    )
              }
            </section>


            <section
              className={
                `${styles.panel} ${styles.dataUploadPanel}`
              }
              style={{
                display:
                  activeStep ===
                    "data"
                    ? undefined
                    : "none",
              }}
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
                    01 · Données
                  </span>

                  <h2>
                    Importer vos données
                  </h2>

                  <p
                    className={
                      styles.sectionDescription
                    }
                  >
                    Chargez un ou plusieurs fichiers CSV. DataLens inspecte ensuite
                    leur structure avant toute préparation ou analyse.
                  </p>
                </div>

                {
                  ingestion
                    ? (
                        <span
                          className={
                            styles.sectionStatus
                          }
                        >
                          {
                            ingestion.dataset_count
                          }
                          {" fichier"}
                          {
                            ingestion.dataset_count >
                              1
                              ? "s"
                              : ""
                          }
                        </span>
                      )
                    : null
                }
              </div>


              <label
                className={
                  `${styles.dropZone} ${
                    ingestion
                      ? styles.dropZoneCompact
                      : ""
                  }`
                }
              >
                <input
                  className={
                    styles.fileInput
                  }
                  type="file"
                  multiple
                  accept=".csv,text/csv"
                  onChange={
                    handleDatasetsChange
                  }
                />

                <strong
                  className={
                    styles.dropLabel
                  }
                >
                  {
                    ingestion
                      ? "Ajouter d’autres fichiers CSV"
                      : "Sélectionner un ou plusieurs fichiers CSV"
                  }
                </strong>

                <span
                  className={
                    styles.dropFormats
                  }
                >
                  CSV · plusieurs fichiers acceptés
                </span>

                <span
                  className={
                    styles.dropNote
                  }
                >
                  Les fichiers restent locaux. Plusieurs datasets peuvent
                  être analysés dans le même workspace.
                </span>
              </label>


              {
                ingestionLoading
                  ? (
                      <div
                        className={
                          styles.ingestionState
                        }
                      >
                        Lecture des fichiers…
                      </div>
                    )
                  : null
              }


              {
                ingestion
                  ? (
                      <div
                        className={
                          styles.ingestionSummary
                        }
                      >
                        <div>
                          <strong>
                            {
                              ingestion.dataset_count
                            }
                          </strong>

                          <span>
                            fichier
                            {
                              ingestion.dataset_count >
                              1
                                ? "s"
                                : ""
                            }
                          </span>
                        </div>

                        <div>
                          <strong>
                            {
                              formatNumber(
                                ingestion.total_rows
                              )
                            }
                          </strong>

                          <span>
                            lignes au total
                          </span>
                        </div>
                      </div>
                    )
                  : null
              }
            </section>
          </div>


          {
            ingestion &&
            activeStep ===
              "data"
              ? (
                  <section
                    className={
                      styles.datasetWorkspace
                    }
                  >
                    <div
                      className={
                        styles.datasetWorkspaceHeader
                      }
                    >
                      <div>
                        <span
                          className={
                            styles.eyebrow
                          }
                        >
                          Datasets chargés
                        </span>

                        <h2>
                          Fichiers détectés
                        </h2>

                        <p>
                          Vérifiez les colonnes et leur typage détecté avant de
                          poursuivre. La préparation détaillée aura lieu à l’étape 3.
                        </p>
                      </div>

                      <span
                        className={
                          styles.sectionStatus
                        }
                      >
                        {
                          ingestion.dataset_count
                        }
                        {" dataset"}
                        {
                          ingestion.dataset_count >
                            1
                            ? "s"
                            : ""
                        }
                      </span>
                    </div>


                    <div
                      className={
                        styles.datasetGrid
                      }
                    >
                      {
                        ingestion.datasets.map(
                          (
                            manifest,
                            index
                          ) => {
                            const active =
                              index ===
                              activeDatasetIndex;


                            return (
                              <button
                                className={
                                  `${styles.datasetTile} ${
                                    active
                                      ? styles.datasetTileActive
                                      : ""
                                  }`
                                }
                                key={
                                  manifest.dataset_id
                                }
                                type="button"
                                onClick={
                                  () =>
                                    setActiveDatasetIndex(
                                      index
                                    )
                                }
                              >
                                <div
                                  className={
                                    styles.datasetTileTop
                                  }
                                >
                                  <span
                                    className={
                                      styles.datasetIcon
                                    }
                                  >
                                    CSV
                                  </span>

                                  {
                                    active
                                      ? (
                                          <span
                                            className={
                                              styles.selectedPill
                                            }
                                          >
                                            Sélectionné
                                          </span>
                                        )
                                      : null
                                  }
                                </div>


                                <strong
                                  className={
                                    styles.datasetName
                                  }
                                >
                                  {
                                    manifest.filename
                                  }
                                </strong>


                                <div
                                  className={
                                    styles.datasetStats
                                  }
                                >
                                  <span>
                                    {
                                      formatNumber(
                                        manifest.row_count
                                      )
                                    } lignes
                                  </span>

                                  <span>
                                    {
                                      manifest.column_count
                                    } colonnes
                                  </span>
                                </div>
                              </button>
                            );
                          }
                        )
                      }
                    </div>


                    {
                      activeManifest
                        ? (
                            <div
                              className={
                                styles.activeDataset
                              }
                            >
                              <div
                                className={
                                  styles.activeDatasetHeader
                                }
                              >
                                <div>
                                  <h3>
                                    {
                                      activeManifest.filename
                                    }
                                  </h3>

                                  <p>
                                    {
                                      formatNumber(
                                        activeManifest.row_count
                                      )
                                    } lignes
                                    {" · "}
                                    {
                                      activeManifest.column_count
                                    } colonnes
                                  </p>
                                </div>
                              </div>


                              <div
                                className={
                                  styles.columnTable
                                }
                              >
                                <div
                                  className={
                                    styles.columnTableHeader
                                  }
                                >
                                  <span>
                                    Variable
                                  </span>

                                  <span>
                                    Type
                                  </span>

                                  <span>
                                    Manquantes
                                  </span>

                                  <span>
                                    Distinctes
                                  </span>
                                </div>


                                {
                                  activeManifest.columns.map(
                                    (
                                      column
                                    ) => (
                                      <div
                                        className={
                                          styles.columnRow
                                        }
                                        key={
                                          column.name
                                        }
                                      >
                                        <strong>
                                          {
                                            friendlyVariableLabel(
                                              column.name
                                            )
                                          }
                                        </strong>

                                        <span
                                          className={
                                            styles.kindBadge
                                          }
                                        >
                                          {
                                            analysisKindLabel(
                                              column.analysis_kind
                                            )
                                          }
                                        </span>

                                        <span>
                                          {
                                            column.missing_count ===
                                            0
                                              ? "Aucune"
                                              : formatNumber(
                                                  column.missing_count
                                                )
                                          }
                                        </span>

                                        <span>
                                          {
                                            formatNumber(
                                              column.unique_count
                                            )
                                          }
                                        </span>
                                      </div>
                                    )
                                  )
                                }
                              </div>


                              <div
                                className={
                                  `${styles.availabilityCard} ${styles.availabilityReady}`
                                }
                              >
                                <div>
                                  <strong>
                                    Inclus dans
                                    l’analyse globale
                                  </strong>

                                  <p>
                                    Le moteur Python utilisera
                                    ce fichier pour découvrir
                                    et exécuter les analyses
                                    compatibles.
                                  </p>
                                </div>
                              </div>
                            </div>
                          )
                        : null
                    }
                  </section>
                )
              : null
          }


          {
            activeStep ===
              "data" &&
            dataReady
              ? (
                  <div
                    className={
                      styles.submitArea
                    }
                  >
                    <div
                      className={
                        styles.submitInfo
                      }
                    >
                      <strong>
                        Étape Données terminée
                      </strong>

                      <span>
                        Vos fichiers sont chargés. Ajoutez maintenant une demande
                        d’analyse ou du contexte métier si nécessaire.
                      </span>
                    </div>

                    <button
                      className={
                        styles.submitButton
                      }
                      type="button"
                      onClick={
                        () =>
                          setActiveStep(
                            "documents"
                          )
                      }
                    >
                      Continuer vers Documents
                    </button>
                  </div>
                )
              : null
          }


          {
            activeStep ===
              "documents"
              ? (
                  <div
                    className={
                      styles.submitArea
                    }
                  >
                    <div
                      className={
                        styles.submitInfo
                      }
                    >
                      <strong>
                        {
                          documents.length >
                            0
                            ? `${documents.length} document${
                                documents.length >
                                1
                                  ? "s"
                                  : ""
                              } ajouté${
                                documents.length >
                                1
                                  ? "s"
                                  : ""
                              }`
                            : "Aucun document ajouté"
                        }
                      </strong>

                      <span>
                        Votre demande et le contexte éventuel sont prêts.
                        Vous pouvez poursuivre même sans document métier.
                      </span>
                    </div>

                    <button
                      className={
                        styles.submitButton
                      }
                      type="button"
                      disabled={
                        !dataReady
                      }
                      onClick={
                        () =>
                          setActiveStep(
                            "preparation"
                          )
                      }
                    >
                      Continuer vers Préparation
                    </button>
                  </div>
                )
              : null
          }


          {
            activeStep ===
              "preparation"
              ? (
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
                          Préflight
                        </span>

                        <h2>
                          Préparation avant analyse
                        </h2>

                        <p
                          className={
                            styles.resultSubtitle
                          }
                        >
                          DataLens inspecte les fichiers,
                          mesure les problèmes de qualité et
                          prépare uniquement les corrections
                          qui pourront être justifiées et tracées.
                        </p>
                      </div>
                    </div>


                    <PreparationSubstepNavigation
                      session={
                        preparationSession
                      }
                      activeStep={
                        activePreparationStep
                      }
                      onStepChange={
                        setActivePreparationStep
                      }
                      qualityReady={
                        qualityReport !==
                        null
                      }
                      cleaningPlanReady={
                        cleaningPlan !==
                        null
                      }
                      cleaningActionCount={
                        cleaningPlan
                          ?.action_count ??
                        0
                      }
                      cleaningApplied={
                        cleaningExecution !==
                        null
                      }
                      semanticReviewReady={
                        semanticReview !==
                        null
                      }
                      semanticDecisionCount={
                        semanticReview
                          ?.decisions
                          .length ??
                        0
                      }
                      semanticConfirmed={
                        semanticConfirmation
                          ?.confirmed ===
                        true
                      }
                    />


                    <details
                      style={{
                        marginTop:
                          "10px",

                        padding:
                          "10px 12px",

                        border:
                          "1px solid rgba(255,255,255,0.055)",

                        borderRadius:
                          "11px",

                        background:
                          "rgba(255,255,255,0.008)",
                      }}
                    >
                      <summary
                        style={{
                          cursor:
                            "pointer",

                          fontSize:
                            "0.6rem",

                          fontWeight:
                            700,

                          opacity:
                            0.58,
                        }}
                      >
                        Voir le workflow technique
                        {" · "}
                        7 étapes backend
                      </summary>

                      <div
                        style={{
                          marginTop:
                            "10px",
                        }}
                      >
                        <PreparationWorkflowPanel
                          session={
                            preparationSession
                          }
                          loading={
                            preparationSessionLoading
                          }
                          error={
                            preparationSessionError
                          }
                          onRefresh={
                            preparationSession
                              ? handleRefreshPreparationSession
                              : undefined
                          }
                        />
                      </div>
                    </details>


                    {
                      activePreparationStep ===
                        "diagnostic"
                        ? (
                            <DataPreparationStudio
                              ingestion={
                                ingestion
                              }
                              qualityReport={
                                qualityReport
                              }
                              qualityLoading={
                                qualityLoading
                              }
                              qualityError={
                                qualityError
                              }
                            />
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "cleaning"
                        ? (
                            <CleaningPlanPanel
                              plan={
                                cleaningPlan
                              }
                              loading={
                                cleaningPlanLoading
                              }
                              error={
                                cleaningPlanError
                              }
                              selectedActionIds={
                                selectedCleaningActionIds
                              }
                              execution={
                                cleaningExecution
                              }
                              applyLoading={
                                cleaningApplyLoading
                              }
                              applyError={
                                cleaningApplyError
                              }
                              exportLoading={
                                preparedExportLoading
                              }
                              exportError={
                                preparedExportError
                              }
                              onToggleAction={
                                handleToggleCleaningAction
                              }
                              onApply={
                                handleApplyCleaning
                              }
                              onExportPrepared={
                                handleExportPreparedData
                              }
                              onContinueSemantic={
                                () => {
                                  if (
                                    (
                                      cleaningPlan
                                        ?.protected_issue_count ??
                                      0
                                    ) >
                                    0
                                  ) {
                                    setActivePreparationStep(
                                      "semantic"
                                    );

                                    return;
                                  }


                                  const nextStage =
                                    preparationSession
                                      ?.snapshot
                                      .next_stage;


                                  if (
                                    nextStage ===
                                    "validate"
                                  ) {
                                    setActivePreparationStep(
                                      "validation"
                                    );

                                    return;
                                  }


                                  if (
                                    nextStage ===
                                      "transform" ||
                                    nextStage ===
                                      "combine"
                                  ) {
                                    setActivePreparationStep(
                                      "transform"
                                    );

                                    return;
                                  }


                                  setActivePreparationStep(
                                    "transform"
                                  );
                                }
                              }
                            />
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "semantic"
                        ? (
                            <>
                              <SemanticReviewPanel
                                deterministicCleaningReady={
                                  cleaningPlan !==
                                    null &&
                                  (
                                    cleaningPlan.action_count ===
                                      0 ||
                                    cleaningExecution !==
                                      null
                                  )
                                }
                                review={
                                  semanticReview
                                }
                                reviewLoading={
                                  semanticReviewLoading
                                }
                                reviewError={
                                  semanticReviewError
                                }
                                plan={
                                  semanticCleaningPlan
                                }
                                planLoading={
                                  semanticPlanLoading
                                }
                                planError={
                                  semanticPlanError
                                }
                                selectedActionIds={
                                  selectedSemanticActionIds
                                }
                                canonicalValues={
                                  semanticCanonicalValues
                                }
                                execution={
                                  semanticCleaningExecution
                                }
                                applyLoading={
                                  semanticApplyLoading
                                }
                                applyError={
                                  semanticApplyError
                                }
                                onRunReview={
                                  handleRunSemanticReview
                                }
                                onSetDecision={
                                  handleSetSemanticDecision
                                }
                                onCanonicalChange={
                                  handleSemanticCanonicalChange
                                }
                                onApply={
                                  handleApplySemanticCleaning
                                }
                              />


                              <SemanticConfirmationPanel
                                review={
                                  semanticReview
                                }
                                plan={
                                  semanticCleaningPlan
                                }
                                execution={
                                  semanticCleaningExecution
                                }
                                confirmedIssueIds={
                                  confirmedSemanticIssueIds
                                }
                                manualResolutionNotes={
                                  semanticManualResolutionNotes
                                }
                                confirmation={
                                  semanticConfirmation
                                }
                                loading={
                                  semanticConfirmationLoading
                                }
                                error={
                                  semanticConfirmationError
                                }
                                onToggleIssue={
                                  handleToggleSemanticIssueConfirmation
                                }
                                onManualResolutionChange={
                                  handleSemanticManualResolutionChange
                                }
                                onConfirm={
                                  handleConfirmSemanticReview
                                }
                              />
                            </>
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "transform"
                        ? (
                            <PreparationTransformPanel
                              session={
                                preparationSession
                              }
                            />
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "validation"
                        ? (
                            <>
                              <PreparationFinalizationPanel
                                session={
                                  preparationSession
                                }
                                loading={
                                  finalValidationLoading
                                }
                                error={
                                  finalValidationError
                                }
                                onValidate={
                                  handleValidatePreparation
                                }
                              />


                              <div
                                className={
                                  styles.metricGrid
                                }
                                style={{
                                  marginTop:
                                    "18px",
                                }}
                              >
                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Datasets
                                  </span>

                                  <strong>
                                    {
                                      ingestion?.dataset_count ??
                                      0
                                    }
                                  </strong>
                                </article>

                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Lignes
                                  </span>

                                  <strong>
                                    {
                                      formatNumber(
                                        semanticCleaningExecution
                                          ? semanticCleaningExecution.provenance.reduce(
                                              (
                                                total,
                                                item
                                              ) =>
                                                total +
                                                item.rows_after,
                                              0
                                            )
                                          : cleaningExecution
                                            ? cleaningExecution.provenance.reduce(
                                                (
                                                  total,
                                                  item
                                                ) =>
                                                  total +
                                                  item.rows_after,
                                                0
                                              )
                                            : (
                                                ingestion?.total_rows ??
                                                0
                                              )
                                      )
                                    }
                                  </strong>
                                </article>

                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Documents
                                  </span>

                                  <strong>
                                    {
                                      documents.length
                                    }
                                  </strong>
                                </article>

                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Mode
                                  </span>

                                  <strong>
                                    {
                                      documents.length >
                                        0
                                        ? "Analyse + contexte"
                                        : "Analyse"
                                    }
                                  </strong>
                                </article>
                              </div>


                              <div
                                className={
                                  styles.summaryPanel
                                }
                              >
                                <div
                                  className={
                                    styles.summaryItem
                                  }
                                >
                                  <span>
                                    Règles de préparation
                                  </span>

                                  <p>
                                    Les jointures automatiques ne sont
                                    acceptées que si elles préservent
                                    le grain de la table de faits et
                                    satisfont les garde-fous du moteur.
                                  </p>

                                  <p>
                                    Les agrégations et variables dérivées
                                    restent déterministes et traçables.
                                    Le LLM n’est pas utilisé pour calculer
                                    les résultats statistiques.
                                  </p>
                                </div>
                              </div>
                            </>
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "transform"
                        ? (
                    <section
                      aria-labelledby="ai-planner-title"
                      style={{
                        marginTop:
                          "22px",

                        padding:
                          "20px",

                        border:
                          "1px solid rgba(126, 177, 255, 0.18)",

                        borderRadius:
                          "16px",

                        background:
                          "linear-gradient(180deg, rgba(56, 110, 196, 0.09), rgba(255,255,255,0.018))",
                      }}
                    >
                      <div
                        style={{
                          display:
                            "flex",

                          alignItems:
                            "flex-start",

                          justifyContent:
                            "space-between",

                          gap:
                            "18px",

                          flexWrap:
                            "wrap",
                        }}
                      >
                        <div
                          style={{
                            maxWidth:
                              "720px",
                          }}
                        >
                          <span
                            className={
                              styles.eyebrow
                            }
                          >
                            {
                              activePlannerUi.eyebrow
                            }
                          </span>

                          <h3
                            id="ai-planner-title"
                            style={{
                              margin:
                                "8px 0 8px",

                              fontSize:
                                "1.12rem",
                            }}
                          >
                            {
                              activePlannerUi.title
                            }
                          </h3>

                          <p
                            className={
                              styles.resultSubtitle
                            }
                            style={{
                              margin:
                                0,
                            }}
                          >
                            {
                              activePlannerUi.description
                            }
                          </p>
                        </div>


                        <div
                          style={{
                            display:
                              "flex",

                            gap:
                              "10px",

                            flexWrap:
                              "wrap",

                            justifyContent:
                              "flex-end",
                          }}
                        >
                          <button
                            className={
                              styles.submitButton
                            }
                            type="button"
                            aria-disabled={
                              aiPlanPreviewDisabled
                            }
                            onClick={
                              () => {
                                if (
                                  aiPlanPreviewDisabled
                                ) {
                                  return;
                                }


                                void handleAiPlanningPreview();
                              }
                            }
                            style={{
                              opacity:
                                aiPlanPreviewDisabled
                                  ? 0.55
                                  : 1,

                              cursor:
                                aiPlanPreviewDisabled
                                  ? "not-allowed"
                                  : "pointer",
                            }}
                          >
                            {
                              aiPlanLoading
                                ? "DataLens prépare le plan…"
                                : aiPlanReport
                                  ? "Regénérer le plan"
                                  : "Préparer le plan"
                            }
                          </button>


                          <button
                            className={
                              styles.submitButton
                            }
                            type="button"
                            aria-disabled={
                              aiNativeRunDisabled
                            }
                            onClick={
                              () => {
                                if (
                                  aiNativeRunDisabled
                                ) {
                                  return;
                                }


                                void handleAiNativeRun();
                              }
                            }
                            style={{
                              opacity:
                                aiNativeRunDisabled
                                  ? 0.55
                                  : 1,

                              cursor:
                                aiNativeRunDisabled
                                  ? "not-allowed"
                                  : "pointer",

                              border:
                                "1px solid rgba(126, 177, 255, 0.28)",
                            }}
                          >
                            {
                              aiNativeLoading
                                ? "Analyse en cours…"
                                : aiNativeReport
                                  ? "Relancer l’analyse"
                                  : "Exécuter l’analyse"
                            }
                          </button>
                        </div>
                      </div>


                      {
                        !objective.trim()
                          ? (
                              <p
                                style={{
                                  margin:
                                    "16px 0 0",

                                  fontSize:
                                    "0.82rem",

                                  opacity:
                                    0.66,
                                }}
                              >
                                Ajoutez une demande d’analyse dans
                                l’étape Documents pour préparer
                                le plan analytique.
                              </p>
                            )
                          : null
                      }


                      <div
                        style={{
                          marginTop:
                            "10px",

                          display:
                            "grid",

                          gap:
                            "8px",
                        }}
                      >
                        <p
                          style={{
                            margin:
                              0,

                            fontSize:
                              "0.76rem",

                            opacity:
                              0.62,
                          }}
                        >
                          {
                            activePlannerUi.details
                          }
                        </p>


                        <div
                          aria-label="Familles analytiques natives supportées"
                          style={{
                            display:
                              "flex",

                            gap:
                              "6px",

                            flexWrap:
                              "wrap",
                          }}
                        >
                          {
                            (
                              aiNativeReport
                                ?.supported_native_families
                              ??
                              [
                                "quantitative_association",
                                "categorical_association",
                                "group_comparison",
                                "distribution",
                                "time_series",
                              ]
                            ).map(
                              (
                                family
                              ) => (
                                <span
                                  key={
                                    family
                                  }
                                  style={{
                                    padding:
                                      "5px 7px",

                                    border:
                                      "1px solid rgba(126, 177, 255, 0.15)",

                                    borderRadius:
                                      "999px",

                                    background:
                                      "rgba(126, 177, 255, 0.035)",

                                    fontSize:
                                      "0.67rem",

                                    opacity:
                                      0.74,
                                  }}
                                >
                                  {
                                    family
                                  }
                                </span>
                              )
                            )
                          }
                        </div>
                      </div>


                      {
                        aiPlanError
                          ? (
                              <div
                                role="alert"
                                style={{
                                  marginTop:
                                    "16px",

                                  padding:
                                    "13px 14px",

                                  border:
                                    "1px solid rgba(255, 132, 132, 0.22)",

                                  borderRadius:
                                    "12px",

                                  background:
                                    "rgba(154, 50, 50, 0.10)",
                                }}
                              >
                                <strong>
                                  Plan analytique indisponible
                                </strong>

                                <p
                                  style={{
                                    margin:
                                      "6px 0 0",
                                  }}
                                >
                                  {
                                    aiPlanError
                                  }
                                </p>
                              </div>
                            )
                          : null
                      }


                      {
                        aiNativeError
                          ? (
                              <div
                                role="alert"
                                style={{
                                  marginTop:
                                    "16px",

                                  padding:
                                    "13px 14px",

                                  border:
                                    "1px solid rgba(255, 132, 132, 0.22)",

                                  borderRadius:
                                    "12px",

                                  background:
                                    "rgba(154, 50, 50, 0.10)",
                                }}
                              >
                                <strong>
                                  Analyse non exécutée
                                </strong>

                                <p
                                  style={{
                                    margin:
                                      "6px 0 0",
                                  }}
                                >
                                  {
                                    aiNativeError
                                  }
                                </p>
                              </div>
                            )
                          : null
                      }


                      {
                        aiPlanReport
                          ? (
                              <div
                                style={{
                                  marginTop:
                                    "18px",

                                  display:
                                    "grid",

                                  gap:
                                    "14px",
                                }}
                              >
                                <div
                                  style={{
                                    display:
                                      "grid",

                                    gridTemplateColumns:
                                      "repeat(auto-fit, minmax(120px, 1fr))",

                                    gap:
                                      "10px",
                                  }}
                                >
                                  <article
                                    className={
                                      styles.metricCard
                                    }
                                  >
                                    <span>
                                      Moteur de planification
                                    </span>

                                    <strong
                                      title={
                                        aiPlanReport.model
                                      }
                                    >
                                      {
                                        plannerEngineLabel(
                                          aiPlanReport.model
                                        )
                                      }
                                    </strong>
                                  </article>

                                  <article
                                    className={
                                      styles.metricCard
                                    }
                                  >
                                    <span>
                                      Propositions
                                    </span>

                                    <strong>
                                      {
                                        aiPlanReport.proposal_count
                                      }
                                    </strong>
                                  </article>

                                  <article
                                    className={
                                      styles.metricCard
                                    }
                                  >
                                    <span>
                                      Validées
                                    </span>

                                    <strong>
                                      {
                                        aiPlanReport.validated_count
                                      }
                                    </strong>
                                  </article>

                                  <article
                                    className={
                                      styles.metricCard
                                    }
                                  >
                                    <span>
                                      Rejetées / bloquées
                                    </span>

                                    <strong>
                                      {
                                        aiPlanReport.rejected_count +
                                        aiPlanReport.blocked_count +
                                        aiPlanReport.ambiguous_count
                                      }
                                    </strong>
                                  </article>
                                </div>


                                {
                                  aiPlanReport.items.map(
                                    (
                                      item
                                    ) => {
                                      const contract =
                                        item.contract;

                                      const confidence =
                                        Math.round(
                                          (
                                            contract
                                              ?.planner_confidence ??
                                            item.proposal.confidence
                                          ) *
                                          100
                                        );

                                      const statusLabel =
                                        item.validation_status ===
                                          "validated"
                                          ? "Validé par Python"
                                          : item.validation_status ===
                                              "rejected"
                                            ? "Rejeté par Python"
                                            : item.validation_status ===
                                                "blocked"
                                              ? "Bloqué"
                                              : "Ambigu";

                                      const bindings =
                                        contract
                                          ?.bindings ??
                                        [];


                                      return (
                                        <article
                                          key={
                                            `${item.proposal_index}-${item.proposal.title}`
                                          }
                                          style={{
                                            padding:
                                              "16px",

                                            border:
                                              item.validation_status ===
                                                "validated"
                                                ? "1px solid rgba(122, 203, 160, 0.22)"
                                                : "1px solid rgba(255, 167, 105, 0.20)",

                                            borderRadius:
                                              "14px",

                                            background:
                                              "rgba(4, 10, 20, 0.28)",
                                          }}
                                        >
                                          <div
                                            style={{
                                              display:
                                                "flex",

                                              justifyContent:
                                                "space-between",

                                              gap:
                                                "12px",

                                              alignItems:
                                                "flex-start",

                                              flexWrap:
                                                "wrap",
                                            }}
                                          >
                                            <div>
                                              <span
                                                style={{
                                                  display:
                                                    "block",

                                                  marginBottom:
                                                    "6px",

                                                  fontSize:
                                                    "0.72rem",

                                                  letterSpacing:
                                                    "0.08em",

                                                  textTransform:
                                                    "uppercase",

                                                  opacity:
                                                    0.62,
                                                }}
                                              >
                                                {
                                                  item.proposal.family
                                                }
                                              </span>

                                              <strong>
                                                {
                                                  item.proposal.title
                                                }
                                              </strong>
                                            </div>

                                            <div
                                              style={{
                                                display:
                                                  "flex",

                                                gap:
                                                  "8px",

                                                alignItems:
                                                  "center",

                                                flexWrap:
                                                  "wrap",
                                              }}
                                            >
                                              <span
                                                style={{
                                                  padding:
                                                    "5px 8px",

                                                  borderRadius:
                                                    "999px",

                                                  border:
                                                    "1px solid rgba(255,255,255,0.10)",

                                                  fontSize:
                                                    "0.72rem",

                                                  fontWeight:
                                                    700,
                                                }}
                                              >
                                                {
                                                  confidence
                                                } % confiance
                                              </span>

                                              <span
                                                style={{
                                                  padding:
                                                    "5px 8px",

                                                  borderRadius:
                                                    "999px",

                                                  border:
                                                    item.validation_status ===
                                                      "validated"
                                                      ? "1px solid rgba(122, 203, 160, 0.30)"
                                                      : "1px solid rgba(255, 167, 105, 0.28)",

                                                  fontSize:
                                                    "0.72rem",

                                                  fontWeight:
                                                    700,
                                                }}
                                              >
                                                {
                                                  statusLabel
                                                }
                                              </span>
                                            </div>
                                          </div>


                                          {
                                            bindings.length >
                                            0
                                              ? (
                                                  <div
                                                    style={{
                                                      marginTop:
                                                        "14px",

                                                      display:
                                                        "flex",

                                                      gap:
                                                        "8px",

                                                      flexWrap:
                                                        "wrap",
                                                    }}
                                                  >
                                                    {
                                                      bindings.map(
                                                        (
                                                          binding
                                                        ) => (
                                                          <span
                                                            key={
                                                              `${binding.role}-${binding.column}`
                                                            }
                                                            style={{
                                                              padding:
                                                                "7px 9px",

                                                              border:
                                                                "1px solid rgba(255,255,255,0.08)",

                                                              borderRadius:
                                                                "9px",

                                                              background:
                                                                "rgba(255,255,255,0.025)",

                                                              fontSize:
                                                                "0.78rem",
                                                            }}
                                                          >
                                                            <strong>
                                                              {
                                                                binding.role
                                                              }
                                                            </strong>

                                                            {" · "}

                                                            {
                                                              binding.column
                                                            }

                                                            {
                                                              binding.analysis_kind
                                                                ? (
                                                                    <>
                                                                      {" · "}

                                                                      {
                                                                        binding.analysis_kind
                                                                      }
                                                                    </>
                                                                  )
                                                                : null
                                                            }
                                                          </span>
                                                        )
                                                      )
                                                    }
                                                  </div>
                                                )
                                              : null
                                          }


                                          {
                                            item.errors.length >
                                            0
                                              ? (
                                                  <div
                                                    style={{
                                                      marginTop:
                                                        "14px",
                                                    }}
                                                  >
                                                    <strong>
                                                      Pourquoi Python a refusé
                                                    </strong>

                                                    {
                                                      item.errors.map(
                                                        (
                                                          message,
                                                          index
                                                        ) => (
                                                          <p
                                                            key={
                                                              `${index}-${message}`
                                                            }
                                                            style={{
                                                              margin:
                                                                "6px 0 0",

                                                              fontSize:
                                                                "0.8rem",

                                                              opacity:
                                                                0.78,
                                                            }}
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
                                            contract &&
                                            item.validation_status ===
                                              "validated"
                                              ? (
                                                  <div
                                                    style={{
                                                      marginTop:
                                                        "14px",

                                                      paddingTop:
                                                        "12px",

                                                      borderTop:
                                                        "1px solid rgba(255,255,255,0.07)",

                                                      display:
                                                        "grid",

                                                      gap:
                                                        "5px",

                                                      fontSize:
                                                        "0.78rem",

                                                      opacity:
                                                        0.74,
                                                    }}
                                                  >
                                                    <span>
                                                      <strong>
                                                        LLM
                                                      </strong>
                                                      {" · "}
                                                      sélection de la famille
                                                      et des rôles
                                                    </span>

                                                    <span>
                                                      <strong>
                                                        Python
                                                      </strong>
                                                      {" · "}
                                                      dataset, colonnes,
                                                      types et contrat
                                                      vérifiés
                                                    </span>

                                                    <span>
                                                      <strong>
                                                        Exécution
                                                      </strong>
                                                      {" · "}
                                                      non lancée depuis
                                                      ce preview
                                                    </span>
                                                  </div>
                                                )
                                              : null
                                          }
                                        </article>
                                      );
                                    }
                                  )
                                }
                              </div>
                            )
                          : null
                      }


                      {
                        aiNativeReport
                          ? (
                              <section
                                aria-labelledby="ai-native-pipeline-title"
                                style={{
                                  marginTop:
                                    "18px",

                                  paddingTop:
                                    "18px",

                                  borderTop:
                                    "1px solid rgba(255,255,255,0.08)",
                                }}
                              >
                                <div
                                  style={{
                                    display:
                                      "flex",

                                    justifyContent:
                                      "space-between",

                                    gap:
                                      "12px",

                                    alignItems:
                                      "flex-start",

                                    flexWrap:
                                      "wrap",
                                  }}
                                >
                                  <div>
                                    <span
                                      className={
                                        styles.eyebrow
                                      }
                                    >
                                      AI Execution Trace
                                    </span>

                                    <h3
                                      id="ai-native-pipeline-title"
                                      style={{
                                        margin:
                                          "7px 0 4px",

                                        fontSize:
                                          "1.02rem",
                                      }}
                                    >
                                      Pipeline analytique vérifié
                                    </h3>

                                    <p
                                      style={{
                                        margin:
                                          0,

                                        fontSize:
                                          "0.8rem",

                                        opacity:
                                          0.68,
                                      }}
                                    >
                                      Une trace observable de la
                                      planification jusqu’au calcul
                                      statistique déterministe.
                                    </p>
                                  </div>


                                  <div
                                    style={{
                                      display:
                                        "flex",

                                      alignItems:
                                        "center",

                                      gap:
                                        "8px",

                                      flexWrap:
                                        "wrap",

                                      justifyContent:
                                        "flex-end",
                                    }}
                                  >
                                    <span
                                      style={{
                                        padding:
                                          "6px 9px",

                                        borderRadius:
                                          "999px",

                                        border:
                                          aiNativeReport.executed_count >
                                          0
                                            ? "1px solid rgba(122, 203, 160, 0.30)"
                                            : "1px solid rgba(255, 167, 105, 0.28)",

                                        fontSize:
                                          "0.72rem",

                                        fontWeight:
                                          700,
                                      }}
                                    >
                                      {
                                        aiNativeReport.executed_count >
                                        0
                                          ? "Exécution vérifiée"
                                          : "Aucune exécution"
                                      }
                                    </span>


                                    {
                                      aiNativeReport.trace_id
                                        ? (
                                            <Link
                                              href={
                                                `/observability?trace=${encodeURIComponent(
                                                  aiNativeReport.trace_id
                                                )}`
                                              }
                                              target="_blank"
                                              rel="noreferrer"
                                              title="Ouvrir la trace exacte dans l’observabilité"
                                              style={{
                                                display:
                                                  "inline-flex",

                                                alignItems:
                                                  "center",

                                                justifyContent:
                                                  "center",

                                                minHeight:
                                                  "30px",

                                                padding:
                                                  "0 10px",

                                                border:
                                                  "1px solid rgba(126, 177, 255, 0.22)",

                                                borderRadius:
                                                  "9px",

                                                color:
                                                  "inherit",

                                                background:
                                                  "rgba(126, 177, 255, 0.055)",

                                                textDecoration:
                                                  "none",

                                                fontSize:
                                                  "0.7rem",

                                                fontWeight:
                                                  700,

                                                whiteSpace:
                                                  "nowrap",
                                              }}
                                            >
                                              Voir cette trace ↗
                                            </Link>
                                          )
                                        : null
                                    }
                                  </div>
                                </div>


                                <div
                                  style={{
                                    marginTop:
                                      "14px",

                                    display:
                                      "grid",

                                    gridTemplateColumns:
                                      "repeat(auto-fit, minmax(150px, 1fr))",

                                    gap:
                                      "10px",
                                  }}
                                >
                                  {
                                    [
                                      {
                                        stage:
                                          "1 · Planner",

                                        actor:
                                          plannerEngineLabel(
                                            aiNativeReport.planner_model
                                          ),

                                        detail:
                                          `${
                                            aiNativeReport
                                              .planner
                                              .attempt_count
                                            ??
                                            1
                                          } tentative(s) · ${
                                            aiNativeReport
                                              .planner
                                              .retry_count
                                            ??
                                            0
                                          } retry`,
                                      },

                                      {
                                        stage:
                                          "2 · Validation",

                                        actor:
                                          "Python",

                                        detail:
                                          `${aiNativeReport.validated_contract_count} contrat(s) validé(s) · ${
                                            aiNativeReport
                                              .planner
                                              .normalization_count
                                            ??
                                            0
                                          } normalisation(s)`,
                                      },

                                      {
                                        stage:
                                          "3 · Tool calling",

                                        actor:
                                          toolEngineLabel(
                                            aiNativeReport.tool_model
                                          ),

                                        detail:
                                          `${
                                            aiNativeReport
                                              .items[
                                                0
                                              ]
                                              ?.native_tool
                                              ?.available_tools
                                              ?.length
                                            ??
                                            aiNativeReport
                                              .supported_native_families
                                              ?.length
                                            ??
                                            0
                                          } outil(s) natif(s) disponible(s)`,
                                      },

                                      {
                                        stage:
                                          "4 · Guardrail",

                                        actor:
                                          "Python",

                                        detail:
                                          "Nom + arguments vérifiés",
                                      },

                                      {
                                        stage:
                                          "5 · Exécution",

                                        actor:
                                          "Python",

                                        detail:
                                          `${aiNativeReport.executed_count} outil(s) exécuté(s)`,
                                      },
                                    ].map(
                                      (
                                        stage
                                      ) => (
                                        <article
                                          key={
                                            stage.stage
                                          }
                                          style={{
                                            padding:
                                              "12px",

                                            border:
                                              "1px solid rgba(255,255,255,0.075)",

                                            borderRadius:
                                              "12px",

                                            background:
                                              "rgba(255,255,255,0.022)",
                                          }}
                                        >
                                          <span
                                            style={{
                                              display:
                                                "block",

                                              marginBottom:
                                                "5px",

                                              fontSize:
                                                "0.68rem",

                                              textTransform:
                                                "uppercase",

                                              letterSpacing:
                                                "0.06em",

                                              opacity:
                                                0.56,
                                            }}
                                          >
                                            {
                                              stage.stage
                                            }
                                          </span>

                                          <strong
                                            style={{
                                              display:
                                                "block",

                                              fontSize:
                                                "0.83rem",
                                            }}
                                          >
                                            {
                                              stage.actor
                                            }
                                          </strong>

                                          <span
                                            style={{
                                              display:
                                                "block",

                                              marginTop:
                                                "4px",

                                              fontSize:
                                                "0.72rem",

                                              opacity:
                                                0.66,
                                            }}
                                          >
                                            {
                                              stage.detail
                                            }
                                          </span>
                                        </article>
                                      )
                                    )
                                  }
                                </div>


                                {
                                  (
                                    aiNativeReport
                                      .planner
                                      .normalization_count
                                    ??
                                    0
                                  ) >
                                  0
                                    ? (
                                        <div
                                          style={{
                                            marginTop:
                                              "12px",

                                            padding:
                                              "10px 12px",

                                            border:
                                              "1px solid rgba(126, 177, 255, 0.12)",

                                            borderRadius:
                                              "10px",

                                            background:
                                              "rgba(126, 177, 255, 0.025)",
                                          }}
                                        >
                                          <span
                                            style={{
                                              display:
                                                "block",

                                              marginBottom:
                                                "5px",

                                              fontSize:
                                                "0.67rem",

                                              textTransform:
                                                "uppercase",

                                              letterSpacing:
                                                "0.06em",

                                              opacity:
                                                0.56,
                                            }}
                                          >
                                            Canonicalisation Python
                                          </span>

                                          <p
                                            style={{
                                              margin:
                                                0,

                                              fontSize:
                                                "0.75rem",

                                              opacity:
                                                0.72,
                                            }}
                                          >
                                            {
                                              aiNativeReport
                                                .planner
                                                .normalization_count
                                            }

                                            {" normalisation(s) de protocole appliquée(s) avant validation du contrat. La sortie brute du planner reste conservée dans la trace backend."}
                                          </p>
                                        </div>
                                      )
                                    : null
                                }


                                {
                                  aiNativeReport.items.map(
                                    (
                                      item
                                    ) => {
                                      const nativeTool =
                                        item.native_tool;

                                      const execution =
                                        nativeTool
                                          ?.execution;

                                      const result =
                                        execution
                                          ?.result;

                                      const metrics =
                                        result
                                          ?.metrics ??
                                        {};

                                      const coefficient =
                                        typeof metrics.coefficient ===
                                        "number"
                                          ? metrics.coefficient
                                          : null;

                                      const pValue =
                                        typeof metrics.p_value ===
                                        "number"
                                          ? metrics.p_value
                                          : null;

                                      const test =
                                        typeof metrics.test ===
                                        "string"
                                          ? metrics.test
                                          : null;


                                      return (
                                        <article
                                          key={
                                            item.contract_id
                                          }
                                          style={{
                                            marginTop:
                                              "14px",

                                            padding:
                                              "15px",

                                            border:
                                              item.pipeline_status ===
                                                "executed"
                                                ? "1px solid rgba(122, 203, 160, 0.20)"
                                                : "1px solid rgba(255, 167, 105, 0.20)",

                                            borderRadius:
                                              "13px",

                                            background:
                                              "rgba(4, 10, 20, 0.24)",
                                          }}
                                        >
                                          <div
                                            style={{
                                              display:
                                                "flex",

                                              justifyContent:
                                                "space-between",

                                              gap:
                                                "10px",

                                              flexWrap:
                                                "wrap",
                                            }}
                                          >
                                            <div>
                                              <span
                                                style={{
                                                  display:
                                                    "block",

                                                  fontSize:
                                                    "0.68rem",

                                                  textTransform:
                                                    "uppercase",

                                                  letterSpacing:
                                                    "0.06em",

                                                  opacity:
                                                    0.55,
                                                }}
                                              >
                                                {
                                                  item.family
                                                }
                                              </span>

                                              <strong
                                                style={{
                                                  display:
                                                    "block",

                                                  marginTop:
                                                    "4px",
                                                }}
                                              >
                                                {
                                                  nativeTool
                                                    ?.requested_tool ??
                                                  "Aucun outil demandé"
                                                }
                                              </strong>
                                            </div>

                                            <span
                                              style={{
                                                fontSize:
                                                  "0.74rem",

                                                fontWeight:
                                                  700,

                                                opacity:
                                                  0.8,
                                              }}
                                            >
                                              {
                                                item.pipeline_status ===
                                                  "executed"
                                                  ? "EXECUTED"
                                                  : item.pipeline_status.toUpperCase()
                                              }
                                            </span>
                                          </div>


                                          {
                                            nativeTool
                                              ? (
                                                  <div
                                                    style={{
                                                      marginTop:
                                                        "12px",

                                                      display:
                                                        "grid",

                                                      gridTemplateColumns:
                                                        "repeat(auto-fit, minmax(150px, 1fr))",

                                                      gap:
                                                        "8px",
                                                    }}
                                                  >
                                                    <div
                                                      style={{
                                                        padding:
                                                          "9px 10px",

                                                        borderRadius:
                                                          "9px",

                                                        background:
                                                          "rgba(255,255,255,0.024)",
                                                      }}
                                                    >
                                                      <span
                                                        style={{
                                                          display:
                                                            "block",

                                                          fontSize:
                                                            "0.66rem",

                                                          opacity:
                                                            0.54,
                                                        }}
                                                      >
                                                        Tool call
                                                      </span>

                                                      <strong
                                                        style={{
                                                          fontSize:
                                                            "0.78rem",
                                                        }}
                                                      >
                                                        {
                                                          nativeTool.tool_call_received
                                                            ? "Reçu"
                                                            : "Absent"
                                                        }
                                                      </strong>
                                                    </div>

                                                    <div
                                                      style={{
                                                        padding:
                                                          "9px 10px",

                                                        borderRadius:
                                                          "9px",

                                                        background:
                                                          "rgba(255,255,255,0.024)",
                                                      }}
                                                    >
                                                      <span
                                                        style={{
                                                          display:
                                                            "block",

                                                          fontSize:
                                                            "0.66rem",

                                                          opacity:
                                                            0.54,
                                                        }}
                                                      >
                                                        Validation Python
                                                      </span>

                                                      <strong
                                                        style={{
                                                          fontSize:
                                                            "0.78rem",
                                                        }}
                                                      >
                                                        {
                                                          nativeTool.validation_status
                                                        }
                                                      </strong>
                                                    </div>

                                                    <div
                                                      style={{
                                                        padding:
                                                          "9px 10px",

                                                        borderRadius:
                                                          "9px",

                                                        background:
                                                          "rgba(255,255,255,0.024)",
                                                      }}
                                                    >
                                                      <span
                                                        style={{
                                                          display:
                                                            "block",

                                                          fontSize:
                                                            "0.66rem",

                                                          opacity:
                                                            0.54,
                                                        }}
                                                      >
                                                        Tentatives
                                                      </span>

                                                      <strong
                                                        style={{
                                                          fontSize:
                                                            "0.78rem",
                                                        }}
                                                      >
                                                        {
                                                          nativeTool.attempt_count
                                                        }

                                                        {
                                                          nativeTool.retry_count >
                                                          0
                                                            ? ` · ${nativeTool.retry_count} retry`
                                                            : ""
                                                        }
                                                      </strong>
                                                    </div>


                                                    <div
                                                      style={{
                                                        padding:
                                                          "9px 10px",

                                                        borderRadius:
                                                          "9px",

                                                        background:
                                                          "rgba(255,255,255,0.024)",
                                                      }}
                                                    >
                                                      <span
                                                        style={{
                                                          display:
                                                            "block",

                                                          fontSize:
                                                            "0.66rem",

                                                          opacity:
                                                            0.54,
                                                        }}
                                                      >
                                                        Catalogue natif
                                                      </span>

                                                      <strong
                                                        style={{
                                                          fontSize:
                                                            "0.78rem",
                                                        }}
                                                      >
                                                        {
                                                          nativeTool
                                                            .available_tools
                                                            ?.length
                                                          ??
                                                          0
                                                        }

                                                        {" outil(s)"}
                                                      </strong>
                                                    </div>
                                                  </div>
                                                )
                                              : null
                                          }


                                          {
                                            execution
                                              ?.arguments
                                              .variables
                                              ? (
                                                  <div
                                                    style={{
                                                      marginTop:
                                                        "10px",

                                                      display:
                                                        "flex",

                                                      gap:
                                                        "7px",

                                                      flexWrap:
                                                        "wrap",
                                                    }}
                                                  >
                                                    {
                                                      Object.entries(
                                                        execution
                                                          .arguments
                                                          .variables
                                                      ).map(
                                                        (
                                                          [
                                                            role,
                                                            column,
                                                          ]
                                                        ) => (
                                                          <span
                                                            key={
                                                              role
                                                            }
                                                            style={{
                                                              padding:
                                                                "6px 8px",

                                                              border:
                                                                "1px solid rgba(255,255,255,0.07)",

                                                              borderRadius:
                                                                "8px",

                                                              fontSize:
                                                                "0.74rem",
                                                            }}
                                                          >
                                                            <strong>
                                                              {
                                                                role
                                                              }
                                                            </strong>

                                                            {" · "}

                                                            {
                                                              column
                                                            }
                                                          </span>
                                                        )
                                                      )
                                                    }
                                                  </div>
                                                )
                                              : null
                                          }


                                          {
                                            result
                                              ? (
                                                  <div
                                                    style={{
                                                      marginTop:
                                                        "12px",

                                                      paddingTop:
                                                        "12px",

                                                      borderTop:
                                                        "1px solid rgba(255,255,255,0.07)",
                                                    }}
                                                  >
                                                    <span
                                                      style={{
                                                        display:
                                                          "block",

                                                        marginBottom:
                                                          "7px",

                                                        fontSize:
                                                          "0.68rem",

                                                        textTransform:
                                                          "uppercase",

                                                        letterSpacing:
                                                          "0.06em",

                                                        opacity:
                                                          0.55,
                                                      }}
                                                    >
                                                      Résultat déterministe
                                                    </span>

                                                    <div
                                                      style={{
                                                        display:
                                                          "flex",

                                                        gap:
                                                          "14px",

                                                        flexWrap:
                                                          "wrap",

                                                        alignItems:
                                                          "baseline",
                                                      }}
                                                    >
                                                      {
                                                        test
                                                          ? (
                                                              <strong>
                                                                {
                                                                  test
                                                                }
                                                              </strong>
                                                            )
                                                          : null
                                                      }

                                                      {
                                                        coefficient !==
                                                        null
                                                          ? (
                                                              <span>
                                                                coefficient&nbsp;
                                                                <strong>
                                                                  {
                                                                    formatDecimal(
                                                                      coefficient
                                                                    )
                                                                  }
                                                                </strong>
                                                              </span>
                                                            )
                                                          : null
                                                      }

                                                      {
                                                        pValue !==
                                                        null
                                                          ? (
                                                              <span>
                                                                p-value&nbsp;
                                                                <strong>
                                                                  {
                                                                    formatDecimal(
                                                                      pValue
                                                                    )
                                                                  }
                                                                </strong>
                                                              </span>
                                                            )
                                                          : null
                                                      }
                                                    </div>

                                                    {
                                                      result.summary[
                                                        0
                                                      ]
                                                        ? (
                                                            <p
                                                              style={{
                                                                margin:
                                                                  "8px 0 0",

                                                                fontSize:
                                                                  "0.78rem",

                                                                opacity:
                                                                  0.72,
                                                              }}
                                                            >
                                                              {
                                                                result.summary[
                                                                  0
                                                                ]
                                                              }
                                                            </p>
                                                          )
                                                        : null
                                                    }
                                                  </div>
                                                )
                                              : null
                                          }


                                          {
                                            nativeTool
                                              ?.attempts
                                              .length
                                              ? (
                                                  <details
                                                    style={{
                                                      marginTop:
                                                        "12px",
                                                    }}
                                                  >
                                                    <summary
                                                      style={{
                                                        cursor:
                                                          "pointer",

                                                        fontSize:
                                                          "0.74rem",

                                                        opacity:
                                                          0.72,
                                                      }}
                                                    >
                                                      Voir la trace des tentatives
                                                    </summary>

                                                    <div
                                                      style={{
                                                        marginTop:
                                                          "8px",

                                                        display:
                                                          "grid",

                                                        gap:
                                                          "7px",
                                                      }}
                                                    >
                                                      {
                                                        nativeTool.attempts.map(
                                                          (
                                                            attempt
                                                          ) => (
                                                            <div
                                                              key={
                                                                attempt.attempt_index
                                                              }
                                                              style={{
                                                                padding:
                                                                  "8px 9px",

                                                                border:
                                                                  "1px solid rgba(255,255,255,0.06)",

                                                                borderRadius:
                                                                  "8px",

                                                                fontSize:
                                                                  "0.72rem",

                                                                opacity:
                                                                  0.72,
                                                              }}
                                                            >
                                                              Tentative&nbsp;
                                                              {
                                                                attempt.attempt_index
                                                              }

                                                              {" · "}

                                                              {
                                                                attempt.prompt_variant
                                                              }

                                                              {" · "}

                                                              {
                                                                attempt.tool_call_count
                                                              }

                                                              &nbsp;tool call(s)

                                                              {
                                                                attempt.selected_tool_name
                                                                  ? (
                                                                      <>
                                                                        {" · "}

                                                                        {
                                                                          attempt.selected_tool_name
                                                                        }
                                                                      </>
                                                                    )
                                                                  : null
                                                              }
                                                            </div>
                                                          )
                                                        )
                                                      }
                                                    </div>
                                                  </details>
                                                )
                                              : null
                                          }


                                          {
                                            item.errors.length >
                                            0
                                              ? (
                                                  <div
                                                    style={{
                                                      marginTop:
                                                        "10px",
                                                    }}
                                                  >
                                                    {
                                                      item.errors.map(
                                                        (
                                                          message,
                                                          index
                                                        ) => (
                                                          <p
                                                            key={
                                                              `${index}-${message}`
                                                            }
                                                            style={{
                                                              margin:
                                                                "4px 0 0",

                                                              fontSize:
                                                                "0.75rem",

                                                              opacity:
                                                                0.74,
                                                            }}
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
                                        </article>
                                      );
                                    }
                                  )
                                }
                              </section>
                            )
                          : null
                      }
                    </section>
                          )
                        : null
                    }
                  </section>
                )
              : null
          }


          <div
            className={
              styles.submitArea
            }
            style={{
              display:
                (
                  activeStep ===
                    "preparation" &&
                  activePreparationStep ===
                    "validation"
                )
                  ? undefined
                  : "none",
            }}
          >
            <div
              className={
                styles.submitInfo
              }
            >
              {
                ingestion
                  ? (
                      <>
                        <strong>
                          {
                            ingestion.dataset_count
                          } dataset
                          {
                            ingestion.dataset_count >
                            1
                              ? "s"
                              : ""
                          }
                          {
                            documents.length >
                            0
                              ? ` · ${documents.length} document${
                                  documents.length >
                                  1
                                    ? "s"
                                    : ""
                                }`
                              : ""
                          }
                        </strong>

                        <span>
                          {
                            documents.length >
                            0
                              ? "Analyse déterministe + contexte documentaire local"
                              : "Analyse déterministe"
                          }
                        </span>
                      </>
                    )
                  : (
                      <span>
                        Ajoutez des données
                        pour commencer.
                      </span>
                    )
              }
            </div>


            <button
              className={
                styles.submitButton
              }
              type="submit"
              disabled={
                submitDisabled
              }
            >
              {
                analysisLoading
                  ? (
                      documents.length >
                      0
                        ? "Analyse et contextualisation…"
                        : "Analyse en cours…"
                    )
                  : (
                      !preparationSession
                        ?.snapshot
                        .ready_for_analysis
                        ? "Préparation à terminer"
                        : (
                            documents.length >
                            0
                              ? "Analyser avec le contexte"
                              : "Analyser les données"
                          )
                    )
              }
            </button>
          </div>


          {
            error
              ? (
                  <div
                    className={
                      styles.error
                    }
                    role="alert"
                  >
                    <strong>
                      Impossible de lancer
                      l’analyse
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
        </form>


        {
          report &&
          (
            activeStep ===
              "analyses" ||
            activeStep ===
              "report"
          )
            ? (
                <section
                  className={
                    styles.results
                  }
                >
                  <header
                    className={
                      styles.resultHeader
                    }
                  >
                    <div>
                      <span
                        className={
                          styles.eyebrow
                        }
                      >
                        Analyse terminée
                      </span>

                      <h2
                        className={
                          styles.resultTitle
                        }
                      >
                        {
                          report.title
                        }
                      </h2>

                      <p
                        className={
                          styles.resultSubtitle
                        }
                      >
                        {
                          report.inventory.dataset_count
                        } fichiers
                        {" · "}
                        {
                          report.inventory
                            .discovered_analysis_count
                        } analyses découvertes
                        {" · "}
                        {
                          report.inventory
                            .executed_analysis_count
                        } exécutées
                      </p>
                    </div>


                    <div
                      className={
                        styles.resultMeta
                      }
                    >
                      <span>
                        Mode
                      </span>

                      <strong>
                        {
                          ragReport
                            ? "Analyse + RAG"
                            : "Analyse"
                        }
                      </strong>
                    </div>
                  </header>


                  {
                    !(
                      activeStep ===
                        "analyses" &&
                      report.entity_outlier_finding
                    )
                      ? (
                          <div
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
                        Fichiers
                      </span>

                      <strong>
                        {
                          report.inventory.dataset_count
                        }
                      </strong>
                    </article>


                    <article
                      className={
                        styles.metricCard
                      }
                    >
                      <span>
                        Analyses découvertes
                      </span>

                      <strong>
                        {
                          report.inventory
                            .discovered_analysis_count
                        }
                      </strong>
                    </article>


                    <article
                      className={
                        styles.metricCard
                      }
                    >
                      <span>
                        Analyses exécutées
                      </span>

                      <strong>
                        {
                          report.inventory
                            .executed_analysis_count
                        }
                      </strong>
                    </article>


                    <article
                      className={
                        styles.metricCard
                      }
                    >
                      <span>
                        Contrôles qualité
                      </span>

                      <strong>
                        {
                          report.inventory
                            .quality_check_count
                        }
                      </strong>
                    </article>
                  </div>


                 


                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report" &&
                    signalKpis.length >
                    0
                      ? (
                          <>
                            <div
                              className={
                                styles.sectionHead
                              }
                            >
                              <h2>
                                Signaux clés
                              </h2>
                            </div>


                            <div
                              className={
                                styles.metricGrid
                              }
                            >
                              {
                                signalKpis.map(
                                  (
                                    kpi
                                  ) => (
                                    <article
                                      className={
                                        styles.metricCard
                                      }
                                      key={
                                        kpi.key
                                      }
                                    >
                                      <span>
                                        {
                                          kpi.label
                                        }
                                      </span>

                                      <strong>
                                        {
                                          kpi.value
                                        }
                                      </strong>

                                      <small>
                                        {
                                          kpi.context
                                        }
                                      </small>
                                    </article>
                                  )
                                )
                              }
                            </div>
                          </>
                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report"
                      ? (
                          <QualityReportSection
                            report={
                              qualityReport
                            }
                            cleaningPlan={
                              cleaningPlan
                            }
                            cleaningExecution={
                              cleaningExecution
                            }
                          />
                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report" &&
                    report.executive_summary
                      .length >
                    0
                      ? (
                          <section
                            className={
                              styles.summaryPanel
                            }
                          >
                            <div
                              className={
                                styles.summaryItem
                              }
                            >
                              <span>
                                Synthèse
                              </span>

                              {
                                report.executive_summary.map(
                                  (
                                    item
                                  ) => (
                                    <p
                                      key={
                                        item
                                      }
                                    >
                                      {
                                        item
                                      }
                                    </p>
                                  )
                                )
                              }
                            </div>
                          </section>
                        )
                      : null
                  }

                  {
                    activeStep ===
                      "analyses"
                      ? (
                          <>
                            {
                              report.entity_outlier_finding
                                ? (
                                    <EntityOutlierRequestedAnswer
                                      finding={
                                        report.entity_outlier_finding
                                      }
                                      objective={
                                        objective
                                      }
                                    />
                                  )
                                : aiNativeReport
                                  ? (
                                      <NativeRequestedAnalysisCard
                                        report={
                                          aiNativeReport
                                        }
                                        objective={
                                          objective
                                        }
                                      />
                                    )
                                  : aiPlanReport
                                    ? (
                                        <PlannerBlockedAnalysisCard
                                          planner={
                                            aiPlanReport
                                          }
                                          objective={
                                            objective
                                          }
                                        />
                                      )
                                    : null
                            }


                            {
                              (
                                documentSummary !==
                                  null ||
                                requestedPlan !==
                                  null ||
                                report.requested_findings.length >
                                  0
                              )
                                ? (
                                    <details
                                      className={
                                        styles.analysisDisclosure
                                      }
                                    >
                                      <summary
                                        className={
                                          styles.analysisDisclosureSummary
                                        }
                                      >
                                        <div>
                                          <span
                                            className={
                                              styles.eyebrow
                                            }
                                          >
                                            Documentation métier
                                          </span>

                                          <strong>
                                            Demandes issues des documents
                                          </strong>

                                          <small>
                                            Cadrage, demandes vérifiées et résultats
                                            explicitement demandés dans vos documents.
                                          </small>
                                        </div>

                                        <span
                                          className={
                                            styles.analysisDisclosureCount
                                          }
                                        >
                                          {
                                            report.requested_findings.length
                                          }
                                        </span>
                                      </summary>

                                      <div
                                        className={
                                          styles.analysisDisclosureBody
                                        }
                                      >
                                        <DocumentRequestsSummary
                                          summary={
                                            documentSummary
                                          }
                                          plan={
                                            requestedPlan
                                          }
                                        />

                                        {
                                          report.requested_findings.length >
                                          0
                                            ? (
                                                <div
                                                  className={
                                                    styles.explanationGrid
                                                  }
                                                >
                                                  {
                                                    report
                                                      .requested_findings
                                                      .map(
                                                        (
                                                          finding,
                                                          index
                                                        ) => (
                                                          <RequestedFindingCard
                                                            finding={
                                                              finding
                                                            }
                                                            index={
                                                              index
                                                            }
                                                            ragContext={
                                                              ragContextByAnalysisId.get(
                                                                finding.analysis_id
                                                              ) ??
                                                              null
                                                            }
                                                            key={
                                                              finding.analysis_id
                                                            }
                                                          />
                                                        )
                                                      )
                                                  }
                                                </div>
                                              )
                                            : null
                                        }
                                      </div>
                                    </details>
                                  )
                                : null
                            }


                            {
                              report.main_findings.length >
                              0
                                ? (
                                    <details
                                      className={
                                        styles.analysisDisclosure
                                      }
                                    >
                                      <summary
                                        className={
                                          styles.analysisDisclosureSummary
                                        }
                                      >
                                        <div>
                                          <span
                                            className={
                                              styles.eyebrow
                                            }
                                          >
                                            Exploration automatique
                                          </span>

                                          <strong>
                                            Analyses complémentaires
                                          </strong>

                                          <small>
                                            Autres signaux utiles découverts par DataLens.
                                            Ils restent séparés de la réponse à votre demande.
                                          </small>
                                        </div>

                                        <span
                                          className={
                                            styles.analysisDisclosureCount
                                          }
                                        >
                                          {
                                            report.main_findings.length
                                          }
                                        </span>
                                      </summary>

                                      <div
                                        className={
                                          styles.analysisDisclosureBody
                                        }
                                      >
                                        <div
                                          className={
                                            styles.explanationGrid
                                          }
                                        >
                                          {
                                            report.main_findings
                                              .slice(
                                                0,
                                                3
                                              )
                                              .map(
                                                (
                                                  finding,
                                                  index
                                                ) => (
                                                  <FindingCard
                                                    finding={
                                                      finding
                                                    }
                                                    index={
                                                      index
                                                    }
                                                    ragContext={
                                                      finding.analysis_id
                                                        ? (
                                                            ragContextByAnalysisId.get(
                                                              finding.analysis_id
                                                            ) ??
                                                            null
                                                          )
                                                        : null
                                                    }
                                                    key={
                                                      `${
                                                        finding.analysis_id ??
                                                        `${finding.family}-${finding.title}`
                                                      }-${index}`
                                                    }
                                                  />
                                                )
                                              )
                                          }
                                        </div>

                                        {
                                          report.main_findings.length >
                                          3
                                            ? (
                                                <CompactFindingList
                                                  title="Autres analyses découvertes"
                                                  findings={
                                                    report.main_findings.slice(
                                                      3
                                                    )
                                                  }
                                                />
                                              )
                                            : null
                                        }
                                      </div>
                                    </details>
                                  )
                                : null
                            }


                            <details
                              className={
                                `${styles.analysisDisclosure} ${styles.analysisTechnicalDisclosure}`
                              }
                            >
                              <summary
                                className={
                                  styles.analysisDisclosureSummary
                                }
                              >
                                <div>
                                  <span
                                    className={
                                      styles.eyebrow
                                    }
                                  >
                                    Audit
                                  </span>

                                  <strong>
                                    Preuves & méthodologie
                                  </strong>

                                  <small>
                                    Diagnostics, qualité, RAG, analyses non exécutées
                                    et traçabilité du moteur.
                                  </small>
                                </div>

                                <span
                                  className={
                                    styles.analysisDisclosureMeta
                                  }
                                >
                                  Détails
                                </span>
                              </summary>

                              <div
                                className={
                                  styles.analysisDisclosureBody
                                }
                              >
                                <CompactFindingList
                                  title="Analyses complémentaires"
                                  findings={
                                    report.additional_findings
                                  }
                                />

                                <CompactFindingList
                                  title="Diagnostics"
                                  findings={
                                    report.diagnostics
                                  }
                                />

                                <QualityList
                                  items={
                                    report.quality
                                  }
                                />

                                <CompactFindingList
                                  title="Analyses contextuelles"
                                  findings={
                                    report.context_analyses
                                  }
                                />

                                <BlockedAnalysisList
                                  items={
                                    report.blocked_analyses
                                  }
                                />

                                {
                                  ragReport
                                    ? (
                                        <RagReportSummary
                                          rag={
                                            ragReport
                                          }
                                        />
                                      )
                                    : null
                                }

                                {
                                  report.methodology_notes.length >
                                  0
                                    ? (
                                        <details
                                          className={
                                            styles.technicalPanel
                                          }
                                        >
                                          <summary>
                                            Méthodologie et traçabilité
                                          </summary>

                                          <div
                                            className={
                                              styles.technicalReasons
                                            }
                                          >
                                            {
                                              report.methodology_notes.map(
                                                (
                                                  note
                                                ) => (
                                                  <p
                                                    key={
                                                      note
                                                    }
                                                  >
                                                    {
                                                      note
                                                    }
                                                  </p>
                                                )
                                              )
                                            }
                                          </div>
                                        </details>
                                      )
                                    : null
                                }
                              </div>
                            </details>
                          </>
                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report"
                      ? (
                          <div
                            className={
                              styles.submitArea
                            }
                          >
                            <div
                              className={
                                styles.submitInfo
                              }
                            >
                              <strong>
                                Rapport de synthèse
                              </strong>

                              <span>
                                Cette vue rassemble uniquement
                                les éléments de décision. Le PDF
                                est généré par l’API locale ; les
                                données brutes ne sont pas envoyées
                                vers un service externe.
                              </span>
                            </div>

                            <div
                              style={{
                                display:
                                  "flex",

                                flexWrap:
                                  "wrap",

                                gap:
                                  "10px",

                                justifyContent:
                                  "flex-end",
                              }}
                            >
                              <button
                                className={
                                  styles.submitButton
                                }
                                type="button"
                                disabled={
                                  pdfExportLoading
                                }
                                onClick={
                                  handlePdfExport
                                }
                              >
                                {
                                  pdfExportLoading
                                    ? "Génération du PDF…"
                                    : "Exporter en PDF"
                                }
                              </button>


                              <button
                                className={
                                  styles.submitButton
                                }
                                type="button"
                                onClick={
                                  () =>
                                    setActiveStep(
                                      "analyses"
                                    )
                                }
                              >
                                Revoir les analyses
                              </button>
                            </div>
                          </div>
                        )
                      : (
                          <div
                            className={
                              styles.submitArea
                            }
                          >
                            <div
                              className={
                                styles.submitInfo
                              }
                            >
                              <strong>
                                Analyse complète
                              </strong>

                              <span>
                                Passez au rapport pour une lecture
                                plus courte orientée décision.
                              </span>
                            </div>

                            <button
                              className={
                                styles.submitButton
                              }
                              type="button"
                              onClick={
                                () =>
                                  setActiveStep(
                                    "report"
                                  )
                              }
                            >
                              Voir le rapport
                            </button>
                          </div>
                        )
                  }
                </section>
              )
            : null
        }


        <footer
          className={
            styles.footer
          }
        >
          <strong>
            DataLens
          </strong>

          <span>
            Python déterministe
            {" · "}
            IA locale
            {" · "}
            preuves vérifiables
          </span>
        </footer>
      </div>
    </main>
  );
}