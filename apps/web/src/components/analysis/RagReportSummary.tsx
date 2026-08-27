"use client";

import type { RagContextReport }
  from "../../app/types";

import styles
  from "../../app/page.module.css";


export default function RagReportSummary({
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
