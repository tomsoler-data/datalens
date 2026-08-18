"use client";

import type {
  SemanticConfirmationReportView,
} from "./semanticConfirmationApi";


type SemanticDecisionLike = {
  issue_id: string;
  dataset_filename: string;
  column: string | null;
  verdict:
    | "merge_values"
    | "keep_separate"
    | "flag_for_review"
    | "contextualize"
    | "no_change"
    | "abstain";
  confidence: number;
  rationale: string;
  source_values: string[];
  canonical_value: string | null;
  user_message: string;
  python_validated: boolean;
  requires_user_confirmation: boolean;
};


type SemanticReviewLike = {
  decisions: SemanticDecisionLike[];
};


type SemanticPlanLike = {
  actions: Array<{
    action_id: string;
    issue_id: string;
  }>;
};


type SemanticExecutionLike = {
  action_results: Array<{
    action_id: string;
    status:
      | "applied"
      | "skipped";
  }>;
};


type SemanticConfirmationPanelProps = {
  review: SemanticReviewLike | null;
  plan: SemanticPlanLike | null;
  execution: SemanticExecutionLike | null;

  confirmedIssueIds: string[];

  manualResolutionNotes: Record<
    string,
    string
  >;

  confirmation:
    SemanticConfirmationReportView
    | null;

  loading: boolean;

  error:
    string
    | null;

  onToggleIssue: (
    issueId: string,
    checked: boolean
  ) => void;

  onManualResolutionChange: (
    issueId: string,
    note: string
  ) => void;

  onConfirm: () => void;
};


const VERDICT_LABELS: Record<
  SemanticDecisionLike["verdict"],
  string
> = {
  merge_values:
    "Fusionner",

  keep_separate:
    "Conserver séparé",

  flag_for_review:
    "Revue manuelle",

  contextualize:
    "Contextualiser",

  no_change:
    "Aucun changement",

  abstain:
    "Abstention",
};


function isManualReviewVerdict(
  verdict:
    SemanticDecisionLike["verdict"]
): boolean {
  return (
    verdict ===
      "abstain"
    ||
    verdict ===
      "flag_for_review"
  );
}


export default function SemanticConfirmationPanel({
  review,
  plan,
  execution,
  confirmedIssueIds,
  manualResolutionNotes,
  confirmation,
  loading,
  error,
  onToggleIssue,
  onManualResolutionChange,
  onConfirm,
}: SemanticConfirmationPanelProps) {
  if (
    review ===
    null
  ) {
    return null;
  }


  const confirmed =
    confirmation?.confirmed ===
    true;


  const appliedActionIds =
    new Set(
      execution
        ?.action_results
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
        )
        ??
        []
    );


  const actionByIssue =
    new Map(
      plan
        ?.actions
        .map(
          (
            action
          ) => [
            action.issue_id,
            action.action_id,
          ] as const
        )
        ??
        []
    );


  const decisionCount =
    review.decisions.length;


  const explicitlyConfirmedCount =
    review.decisions.filter(
      (
        decision
      ) =>
        confirmedIssueIds.includes(
          decision.issue_id
        )
    ).length;


  const unresolvedCount =
    confirmation
      ?.unresolved_issue_ids
      .length
    ??
    Math.max(
      0,
      decisionCount -
      explicitlyConfirmedCount
    );


  const allDecisionsReady =
    decisionCount >
      0
    &&
    review.decisions.every(
      (
        decision
      ) => {
        const issueConfirmed =
          confirmedIssueIds.includes(
            decision.issue_id
          );


        if (
          !issueConfirmed
        ) {
          return false;
        }


        if (
          decision.verdict !==
          "merge_values"
        ) {
          return true;
        }


        const actionId =
          actionByIssue.get(
            decision.issue_id
          );


        return Boolean(
          actionId
          &&
          appliedActionIds.has(
            actionId
          )
        );
      }
    );


  return (
    <section
      style={{
        marginTop:
          "12px",

        padding:
          "16px",

        border:
          confirmed
            ? "1px solid rgba(122,203,160,0.2)"
            : "1px solid rgba(160,133,255,0.16)",

        borderRadius:
          "14px",

        background:
          confirmed
            ? "rgba(122,203,160,0.018)"
            : "rgba(160,133,255,0.014)",
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
            Confirmation analyste
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
              confirmed
                ? "Revue sémantique confirmée"
                : "Confirmer les décisions validées par Python"
            }
          </strong>


          <p
            style={{
              margin:
                "6px 0 0",

              maxWidth:
                "700px",

              fontSize:
                "0.64rem",

              lineHeight:
                1.5,

              opacity:
                0.56,
            }}
          >
            Le modèle ne clôture jamais la préparation seul.
            Chaque décision doit être examinée explicitement
            par l’analyste. Pour les abstentions et les signaux
            à revoir, une note peut être ajoutée mais elle
            n’est plus obligatoire.
          </p>
        </div>


        <span
          style={{
            padding:
              "5px 8px",

            border:
              confirmed
                ? "1px solid rgba(122,203,160,0.2)"
                : "1px solid rgba(160,133,255,0.16)",

            borderRadius:
              "999px",

            fontSize:
              "0.55rem",

            fontWeight:
              700,
          }}
        >
          {
            confirmed
              ? "✓ CONFIRMÉ"
              : (
                  `${explicitlyConfirmedCount}/${decisionCount} EXAMINÉE(S)`
                )
          }
        </span>
      </div>


      {
        decisionCount ===
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
                Aucune décision sémantique confirmable n’a été
                produite. Les problèmes protégés sans décision
                restent à examiner avant de clôturer le nettoyage.
              </div>
            )
          : (
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
                  review.decisions.map(
                    (
                      decision
                    ) => {
                      const issueId =
                        decision.issue_id;


                      const checked =
                        confirmedIssueIds.includes(
                          issueId
                        );


                      const manualReview =
                        isManualReviewVerdict(
                          decision.verdict
                        );


                      const mergeActionId =
                        actionByIssue.get(
                          issueId
                        );


                      const mergeApplied =
                        decision.verdict !==
                          "merge_values"
                        ||
                        Boolean(
                          mergeActionId
                          &&
                          appliedActionIds.has(
                            mergeActionId
                          )
                        );


                      const checkboxDisabled =
                        confirmed
                        ||
                        loading
                        ||
                        !mergeApplied;


                      const checkboxText =
                        manualReview
                          ? (
                              "J’ai examiné ce signal et "
                              +
                              "j’accepte de poursuivre sans "
                              +
                              "correction automatique"
                            )
                          : "J’ai examiné cette décision";


                      return (
                        <article
                          key={
                            issueId
                          }
                          style={{
                            padding:
                              "12px",

                            border:
                              checked
                                ? "1px solid rgba(122,203,160,0.16)"
                                : "1px solid rgba(255,255,255,0.055)",

                            borderRadius:
                              "10px",

                            background:
                              checked
                                ? "rgba(122,203,160,0.012)"
                                : "rgba(255,255,255,0.006)",
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
                                    "0.71rem",
                                }}
                              >
                                {
                                  decision.column
                                  ??
                                  decision.dataset_filename
                                }
                              </strong>


                              <span
                                style={{
                                  display:
                                    "block",

                                  marginTop:
                                    "4px",

                                  fontSize:
                                    "0.59rem",

                                  opacity:
                                    0.5,
                                }}
                              >
                                {
                                  VERDICT_LABELS[
                                    decision.verdict
                                  ]
                                }

                                {" · "}

                                Python validé

                                {" · "}

                                {
                                  Math.round(
                                    decision.confidence *
                                    100
                                  )
                                }

                                {" %"}
                              </span>
                            </div>


                            <label
                              style={{
                                display:
                                  "flex",

                                alignItems:
                                  "center",

                                gap:
                                  "7px",

                                maxWidth:
                                  manualReview
                                    ? "360px"
                                    : undefined,

                                fontSize:
                                  "0.61rem",

                                lineHeight:
                                  1.4,

                                cursor:
                                  checkboxDisabled
                                    ? "default"
                                    : "pointer",

                                opacity:
                                  checkboxDisabled &&
                                  !checked
                                    ? 0.52
                                    : 1,
                              }}
                            >
                              <input
                                type="checkbox"
                                checked={
                                  checked
                                }
                                disabled={
                                  checkboxDisabled
                                }
                                onChange={
                                  (
                                    event
                                  ) =>
                                    onToggleIssue(
                                      issueId,
                                      event.target.checked
                                    )
                                }
                              />

                              {
                                checkboxText
                              }
                            </label>
                          </div>


                          <p
                            style={{
                              margin:
                                "8px 0 0",

                              fontSize:
                                "0.62rem",

                              lineHeight:
                                1.5,

                              opacity:
                                0.58,
                            }}
                          >
                            {
                              decision.user_message
                              ||
                              decision.rationale
                            }
                          </p>


                          {
                            decision.source_values.length >
                            0
                              ? (
                                  <p
                                    style={{
                                      margin:
                                        "6px 0 0",

                                      fontSize:
                                        "0.59rem",

                                      lineHeight:
                                        1.45,

                                      opacity:
                                        0.46,
                                    }}
                                  >
                                    Valeurs :
                                    {" "}

                                    {
                                      decision.source_values.join(
                                        " · "
                                      )
                                    }
                                  </p>
                                )
                              : null
                          }


                          {
                            decision.verdict ===
                            "merge_values"
                              ? (
                                  <p
                                    style={{
                                      margin:
                                        "8px 0 0",

                                      fontSize:
                                        "0.6rem",

                                      fontWeight:
                                        700,

                                      opacity:
                                        mergeApplied
                                          ? 0.72
                                          : 0.9,
                                    }}
                                  >
                                    {
                                      mergeApplied
                                        ? (
                                            "Fusion appliquée : "
                                            +
                                            "la décision peut être confirmée."
                                          )
                                        : (
                                            "Appliquez d’abord cette fusion "
                                            +
                                            "dans le bloc précédent."
                                          )
                                    }
                                  </p>
                                )
                              : null
                          }


                          {
                            manualReview
                              ? (
                                  <details
                                    style={{
                                      marginTop:
                                        "10px",

                                      padding:
                                        "9px 10px",

                                      border:
                                        "1px solid rgba(255,255,255,0.05)",

                                      borderRadius:
                                        "8px",
                                    }}
                                  >
                                    <summary
                                      style={{
                                        cursor:
                                          confirmed ||
                                          loading
                                            ? "default"
                                            : "pointer",

                                        fontSize:
                                          "0.59rem",

                                        fontWeight:
                                          700,

                                        opacity:
                                          0.58,
                                      }}
                                    >
                                      Ajouter une note analyste
                                      {" "}
                                      <span
                                        style={{
                                          fontWeight:
                                            500,

                                          opacity:
                                            0.7,
                                        }}
                                      >
                                        (optionnel)
                                      </span>
                                    </summary>


                                    <div
                                      style={{
                                        marginTop:
                                          "9px",
                                      }}
                                    >
                                      <label
                                        htmlFor={
                                          `manual-${issueId}`
                                        }
                                        style={{
                                          display:
                                            "block",

                                          marginBottom:
                                            "6px",

                                          fontSize:
                                            "0.58rem",

                                          textTransform:
                                            "uppercase",

                                          letterSpacing:
                                            "0.05em",

                                          opacity:
                                            0.44,
                                        }}
                                      >
                                        Note analyste
                                      </label>


                                      <textarea
                                        id={
                                          `manual-${issueId}`
                                        }
                                        value={
                                          manualResolutionNotes[
                                            issueId
                                          ]
                                          ??
                                          ""
                                        }
                                        disabled={
                                          confirmed
                                          ||
                                          loading
                                        }
                                        onChange={
                                          (
                                            event
                                          ) =>
                                            onManualResolutionChange(
                                              issueId,
                                              event.target.value
                                            )
                                        }
                                        placeholder={
                                          "Ajoutez du contexte uniquement "
                                          +
                                          "si cela apporte une information utile."
                                        }
                                        rows={2}
                                        style={{
                                          width:
                                            "100%",

                                          resize:
                                            "vertical",

                                          padding:
                                            "9px 10px",

                                          border:
                                            "1px solid rgba(255,255,255,0.08)",

                                          borderRadius:
                                            "8px",

                                          background:
                                            "rgba(255,255,255,0.018)",

                                          color:
                                            "inherit",

                                          font:
                                            "inherit",

                                          fontSize:
                                            "0.62rem",

                                          lineHeight:
                                            1.5,
                                        }}
                                      />
                                    </div>
                                  </details>
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
      }


      {
        confirmation
        &&
        !confirmation.confirmed
        &&
        confirmation.unresolved_reasons.length >
          0
          ? (
              <div
                style={{
                  marginTop:
                    "12px",

                  padding:
                    "10px",

                  border:
                    "1px solid rgba(255,176,102,0.16)",

                  borderRadius:
                    "9px",
                }}
              >
                {
                  confirmation.unresolved_reasons.map(
                    (
                      reason
                    ) => (
                      <p
                        key={
                          reason
                        }
                        style={{
                          margin:
                            "4px 0",

                          fontSize:
                            "0.61rem",

                          lineHeight:
                            1.5,

                          opacity:
                            0.68,
                        }}
                      >
                        {
                          reason
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
        error
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
                  error
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
            "13px",

          paddingTop:
            "12px",

          borderTop:
            "1px solid rgba(255,255,255,0.05)",
        }}
      >
        <div>
          <p
            style={{
              margin:
                0,

              fontSize:
                "0.62rem",

              lineHeight:
                1.45,

              opacity:
                0.58,
            }}
          >
            {
              confirmed
                ? (
                    "Toutes les décisions sont "
                    +
                    "résolues côté serveur."
                  )
                : (
                    `${explicitlyConfirmedCount}/${decisionCount} `
                    +
                    "décision(s) examinée(s)."
                  )
            }
          </p>


          {
            !confirmed
            &&
            unresolvedCount >
              0
              ? (
                  <p
                    style={{
                      margin:
                        "3px 0 0",

                      fontSize:
                        "0.58rem",

                      lineHeight:
                        1.4,

                      opacity:
                        0.42,
                    }}
                  >
                    {
                      unresolvedCount
                    }
                    {" "}
                    décision(s) restent à examiner.
                  </p>
                )
              : null
          }
        </div>


        <button
          type="button"
          onClick={
            onConfirm
          }
          disabled={
            confirmed
            ||
            loading
            ||
            !allDecisionsReady
          }
          style={{
            minWidth:
              "220px",

            padding:
              "9px 12px",

            border:
              "1px solid rgba(160,133,255,0.22)",

            borderRadius:
              "9px",

            background:
              confirmed
                ? "rgba(122,203,160,0.09)"
                : "rgba(160,133,255,0.08)",

            color:
              "inherit",

            cursor:
              confirmed
              ||
              loading
              ||
              !allDecisionsReady
                ? "default"
                : "pointer",

            font:
              "inherit",

            fontSize:
              "0.64rem",

            fontWeight:
              750,

            opacity:
              confirmed
                ? 0.78
                : (
                    allDecisionsReady
                      ? 1
                      : 0.48
                  ),
          }}
        >
          {
            confirmed
              ? "Revue confirmée"
              : loading
                ? "Confirmation…"
                : allDecisionsReady
                  ? "Confirmer la revue sémantique"
                  : "Examiner toutes les décisions"
          }
        </button>
      </div>
    </section>
  );
}