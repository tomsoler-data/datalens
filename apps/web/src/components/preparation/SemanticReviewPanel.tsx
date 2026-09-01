"use client";

import {
  useState,
} from "react";

import type {
  SemanticCleaningExecutionView,
  SemanticCleaningPlanView,
  SemanticReviewReportView,
} from "./preparationTypes";

import styles from "../../app/page.module.css";


/*
 * DATALENS_LOCAL_MODEL_PRODUCT_LANGUAGE_V0_1
 *
 * Business-facing copy refers to the local model generically.
 * Exact model identity remains available through technical
 * metadata such as review.model and review.rule_version.
 */


/*
 * DATALENS_DETERMINISTIC_PRODUCT_LANGUAGE_V0_1
 *
 * User-facing terminology describes product guarantees rather
 * than the implementation language.
 *
 * Internal API fields such as python_validated are preserved.
 */


export default function SemanticReviewPanel({
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
  // DATALENS_SEMANTIC_PARTIAL_REAPPLY_V0_1
  /*
   * DATALENS_COMPACT_SEMANTIC_REVIEW_V0_1
   *
   * Presentation-only expansion state.
   *
   * Semantic decisions, canonical values, deterministic
   * controls and execution state remain unchanged.
   */
  const [
    semanticExpanded,
    setSemanticExpanded,
  ] =
    useState(
      false
    );

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

  const semanticActions =
    plan?.actions ??
    [];


  const visibleSemanticActions =
    semanticExpanded
      ? semanticActions
      : semanticActions.slice(
          0,
          2
        );


  const hiddenSemanticActionCount =
    Math.max(
      0,
      semanticActions.length -
        2
    );


  const appliedSemanticActionCount =
    execution
      ?.action_results
      .filter(
        (
          result
        ) =>
          result.status ===
          "applied"
      )
      .length ??
    0;


  return (
    <section
      className={
        `${styles.semanticReviewPhase} ${
          execution
            ? styles.semanticReviewExecuted
            : ""
        }`
      }

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
                  ? `${review.merge_proposal_count} fusion(s) proposée(s) par le modèle local`
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
            Le modèle local propose. Le moteur déterministe revalide les valeurs exactes.
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
              : "MODÈLE LOCAL"
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
                Le modèle local examine les ambiguïtés une par une…
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
                Le moteur déterministe reconstruit les actions sémantiques autorisées…
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
                  visibleSemanticActions.map(
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
                                )} % · Contrôle déterministe`
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
                                            applied ||
                                            applyLoading
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
                                              applied ||
                                              applyLoading
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
                                              applied ||
                                              applyLoading
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
                                              applied ||
                                              applyLoading
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
        semanticActions.length >
        2
          ? (
              <button
                className={
                  styles.semanticReviewToggle
                }
                type="button"
                aria-expanded={
                  semanticExpanded
                }
                onClick={
                  () =>
                    setSemanticExpanded(
                      (
                        current
                      ) =>
                        !current
                    )
                }
              >
                <span>
                  {
                    semanticExpanded
                      ? (
                          `${semanticActions.length} décisions affichées`
                        )
                      : (
                          `+ ${hiddenSemanticActionCount} autre${
                            hiddenSemanticActionCount > 1
                              ? "s"
                              : ""
                          } décision${
                            hiddenSemanticActionCount > 1
                              ? "s"
                              : ""
                          }`
                        )
                  }
                </span>


                <strong>
                  {
                    semanticExpanded
                      ? "Réduire"
                      : "Tout afficher"
                  }
                </strong>
              </button>
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
                n’a été validée par le moteur déterministe. Les ambiguïtés restent
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
                      applyLoading ||
                      selectedCount ===
                        0 ||
                      (
                        execution &&
                        selectedCount <=
                          appliedSemanticActionCount
                      )
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
                    applyLoading
                      ? "Application…"
                      : execution &&
                        selectedCount >
                          appliedSemanticActionCount
                        ? "Réappliquer les fusions confirmées"
                        : execution
                          ? "Nettoyage sémantique appliqué"
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
