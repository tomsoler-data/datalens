"use client";

import type {
  CleaningExecutionView,
  CleaningPlanView,
} from "./preparationTypes";

import {
  formatNumber,
} from "../analysis/analysisPresentation";

import styles from "../../app/page.module.css";


/*
 * DATALENS_CLEANING_EXECUTION_VISUAL_V0_1
 *
 * Presentation-only visual hierarchy for deterministic
 * cleaning. Cleaning contracts and execution semantics
 * remain unchanged.
 */


export default function CleaningPlanPanel({
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
      className={
        styles.cleaningExecution
      }
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
                        applyLoading
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
