"use client";

import type { AIPlannerReportView }
  from "./analysisTypes";

import { familyLabel }
  from "./analysisPlanningPresentation";

import styles
  from "../../app/page.module.css";


export default function PlannerBlockedAnalysisCard({
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
