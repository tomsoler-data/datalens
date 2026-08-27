import Link from "next/link";

import styles from "../../app/page.module.css";

import type {
  AINativePipelineReportView,
  AIPlannerReportView,
} from "./analysisTypes";

import {
  nativePipelineHasExecutedResult,
  toolEngineLabel,
} from "./analysisExecutionPresentation";

import {
  plannerEngineLabel,
} from "./analysisPlanningPresentation";

import {
  formatDecimal,
} from "./analysisPresentation";


type AnalysisExecutionPanelProps = {
  activePlannerUi: {
    eyebrow: string;
    title: string;
    description: string;
    details: string;
  };

  aiNativeError:
    string | null;

  aiNativeLoading:
    boolean;

  aiNativeReport:
    AINativePipelineReportView | null;

  aiPlanError:
    string | null;

  aiPlanReport:
    AIPlannerReportView | null;

  objective:
    string;

  preparationReadyForAnalysis:
    boolean;
};


export default function AnalysisExecutionPanel({
  activePlannerUi,
  aiNativeError,
  aiNativeLoading,
  aiNativeReport,
  aiPlanError,
  aiPlanReport,
  objective,
  preparationReadyForAnalysis,
}: AnalysisExecutionPanelProps) {
  return (
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
              aria-live="polite"
              style={{
                flex:
                  "0 0 auto",

                display:
                  "inline-flex",

                alignItems:
                  "center",

                gap:
                  "8px",

                minHeight:
                  "34px",

                padding:
                  "7px 10px",

                border:
                  "1px solid rgba(126, 177, 255, 0.16)",

                borderRadius:
                  "999px",

                color:
                  aiNativeError
                    ? "#d8b779"
                    : "#a9cfff",

                background:
                  aiNativeError
                    ? "rgba(216, 181, 121, 0.04)"
                    : "rgba(126, 177, 255, 0.04)",

                fontSize:
                  "0.69rem",

                fontWeight:
                  680,

                whiteSpace:
                  "nowrap",
              }}
            >
              <span
                aria-hidden="true"
                style={{
                  width:
                    "6px",

                  height:
                    "6px",

                  borderRadius:
                    "999px",

                  background:
                    aiNativeError
                      ? "#d8b779"
                      : aiNativeLoading
                        ? "#8bb9ff"
                        : nativePipelineHasExecutedResult(
                            aiNativeReport
                          )
                          ? "#8cd7b7"
                          : aiNativeReport
                            ? "#d8b779"
                            : "#8190a5",

                  boxShadow:
                    aiNativeLoading
                      ? "0 0 12px rgba(139, 185, 255, 0.45)"
                      : "none",
                }}
              />

              {
                aiNativeLoading
                  ? "Planification et analyse en cours…"
                  : nativePipelineHasExecutedResult(
                      aiNativeReport
                    )
                    ? "Plan et analyse générés"
                    : aiNativeReport
                      ? "Analyse non exécutée"
                      : aiNativeError
                        ? "Analyse ciblée à vérifier"
                        : objective.trim()
                          ? "Intégré au lancement de l’analyse"
                          : "Aucune demande ciblée"
              }
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


          {
            objective.trim() &&
            !preparationReadyForAnalysis
              ? (
                  <p
                    style={{
                      margin:
                        "16px 0 0",

                      fontSize:
                        "0.78rem",

                      lineHeight:
                        1.55,

                      opacity:
                        0.66,
                    }}
                  >
                    Le plan analytique et son exécution sont
                    verrouillés tant que l’étape Finaliser
                    n’a pas validé la sortie autorisée pour
                    l’analyse.
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
  );
}
