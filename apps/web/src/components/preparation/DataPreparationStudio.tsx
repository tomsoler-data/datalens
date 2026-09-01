"use client";

import {
  useState,
} from "react";

import type {
  MultiDatasetIngestion,
} from "../../app/types";

import type {
  DataQualityReportView,
} from "./preparationTypes";

import {
  preparationQualityLabel,
  preparationSeverityBorder,
  preparationSeverityLabel,
} from "./preparationPresentation";

import {
  formatNumber,
  formatPercent,
} from "../analysis/analysisPresentation";

import styles from "../../app/page.module.css";


/*
 * DATALENS_QUALITY_STUDIO_VISUAL_V0_1
 *
 * Presentation-only hook.
 * Quality report contracts and deterministic execution
 * semantics remain unchanged.
 */


export default function DataPreparationStudio({
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
  /*
   * DATALENS_COMPACT_QUALITY_ISSUES_V0_1
   *
   * Presentation-only state.
   *
   * Quality diagnostics, evidence and server-owned preparation
   * state remain untouched.
   */
  const [
    issuesExpanded,
    setIssuesExpanded,
  ] =
    useState(
      false
    );

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


  const qualitySeverityRank = (
    severity:
      string
  ): number => {
    if (
      severity ===
      "important"
    ) {
      return 0;
    }


    if (
      severity ===
      "moderate"
    ) {
      return 1;
    }


    return 2;
  };


  const orderedQualityIssues =
    [
      ...(
        qualityReport
          ?.issues ??
        []
      ),
    ].sort(
      (
        left,
        right
      ) =>
        qualitySeverityRank(
          left.severity
        ) -
        qualitySeverityRank(
          right.severity
        )
    );


  const visibleQualityIssues =
    issuesExpanded
      ? orderedQualityIssues.slice(
          0,
          12
        )
      : orderedQualityIssues.slice(
          0,
          2
        );


  const collapsedQualityIssueCount =
    Math.max(
      0,
      orderedQualityIssues.length -
        2
    );

  return (
    <section
      className={
        styles.qualityStudio
      }
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
            Le moteur déterministe inspecte réellement les fichiers
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
                      Datasets inspectés
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
                      Lignes inspectées
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
                              visibleQualityIssues.map(
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
                                            Proposition déterministe :
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
                            orderedQualityIssues.length >
                            2
                              ? (
                                  <div
                                    className={
                                      styles.qualityIssueToggleRow
                                    }
                                  >
                                    <span
                                      className={
                                        styles.qualityIssueCount
                                      }
                                    >
                                      {
                                        issuesExpanded
                                          ? (
                                              `${Math.min(
                                                orderedQualityIssues.length,
                                                12
                                              )} problème${
                                                orderedQualityIssues.length > 1
                                                  ? "s"
                                                  : ""
                                              } affiché${
                                                orderedQualityIssues.length > 1
                                                  ? "s"
                                                  : ""
                                              }`
                                            )
                                          : (
                                              `2 prioritaires · ${collapsedQualityIssueCount} autre${
                                                collapsedQualityIssueCount > 1
                                                  ? "s"
                                                  : ""
                                              }`
                                            )
                                      }
                                    </span>


                                    <button
                                      className={
                                        styles.qualityIssueToggle
                                      }
                                      type="button"
                                      aria-expanded={
                                        issuesExpanded
                                      }
                                      onClick={
                                        () =>
                                          setIssuesExpanded(
                                            (
                                              current
                                            ) =>
                                              !current
                                          )
                                      }
                                    >
                                      {
                                        issuesExpanded
                                          ? "Réduire"
                                          : (
                                              `Voir les ${collapsedQualityIssueCount} autre${
                                                collapsedQualityIssueCount > 1
                                                  ? "s"
                                                  : ""
                                              } problème${
                                                collapsedQualityIssueCount > 1
                                                  ? "s"
                                                  : ""
                                              }`
                                            )
                                      }
                                    </button>
                                  </div>
                                )
                              : null
                          }

{
                            issuesExpanded &&
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
                          Moteur déterministe
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
                          validation puis l’exécution par le moteur déterministe.
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
                    Diagnostic généré par le moteur déterministe · aucune donnée brute modifiée
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
