import type {
  CleaningExecutionView,
  CleaningPlanView,
  DataQualityReportView,
} from "./preparationTypes";

import {
  preparationSeverityBorder,
} from "./preparationPresentation";

import {
  formatNumber,
} from "../analysis/analysisPresentation";






/*
 * DATALENS_REPORT_QUALITY_COMPACT_V0_1
 * DATALENS_REPORT_QUALITY_GRAMMAR_R1_V0_1
 *
 * Report quality keeps decision metrics and audit access
 * while removing redundant explanatory copy.
 */

export default function QualityReportSection({
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
          <h2
            style={{
              marginBottom:
                "4px",
            }}
          >
            Qualité des données
          </h2>

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
            "10px",
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
              )} ${
                report.issue_count ===
                1
                  ? "problème"
                  : "problèmes"
              }`
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
              )} ${
                appliedTransformationCount ===
                1
                  ? "transformation"
                  : "transformations"
              }`
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
              )} ${
                deterministicProposalCount ===
                1
                  ? "correction déterministe proposée"
                  : "corrections déterministes proposées"
              }`
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
            Inspecté
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
              )} ${
                preparedRowsAfter ===
                1
                  ? "ligne"
                  : "lignes"
              }`
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
                : "Jeu de données source"
            }
          </small>
        </article>
      </div>


      <details
        style={{
          marginTop:
            "9px",

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
          Détails de préparation
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
