"use client";

import type {
  PreparationSessionView,
  PreparationStageRecord,
} from "./preparationTypes";


type PreparationFinalizationPanelProps = {
  session: PreparationSessionView | null;
  loading: boolean;
  error: string | null;
  onValidate: () => void;
};


function findStage(
  session: PreparationSessionView,
  name: PreparationStageRecord["stage"]
): PreparationStageRecord | null {
  return (
    session.snapshot.stages.find(
      (
        stage
      ) =>
        stage.stage ===
        name
    ) ??
    null
  );
}


export default function PreparationFinalizationPanel({
  session,
  loading,
  error,
  onValidate,
}: PreparationFinalizationPanelProps) {
  if (
    session ===
    null
  ) {
    return null;
  }


  const snapshot =
    session.snapshot;


  const clean =
    findStage(
      session,
      "clean"
    );


  const validate =
    findStage(
      session,
      "validate"
    );


  const ready =
    snapshot.ready_for_analysis;


  const canValidate =
    !ready &&
    snapshot.next_stage ===
      "validate";


  return (
    <section
      style={{
        marginTop:
          "12px",

        padding:
          "16px",

        border:
          ready
            ? "1px solid rgba(122,203,160,0.2)"
            : "1px solid rgba(126,177,255,0.14)",

        borderRadius:
          "14px",

        background:
          ready
            ? "rgba(122,203,160,0.018)"
            : "rgba(72,121,200,0.018)",
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
            Validation finale
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
              ready
                ? "Préparation validée pour l’analyse"
                : canValidate
                  ? "Tous les prérequis peuvent être contrôlés"
                  : "Terminez les étapes précédentes"
            }
          </strong>

          <p
            style={{
              margin:
                "6px 0 0",

              maxWidth:
                "720px",

              fontSize:
                "0.64rem",

              lineHeight:
                1.5,

              opacity:
                0.56,
            }}
          >
            Le navigateur n’envoie jamais un statut PASS.
            FastAPI relit la session, vérifie les preuves et
            décide seul si le dataset peut entrer dans le moteur
            analytique.
          </p>
        </div>


        <span
          style={{
            padding:
              "5px 8px",

            border:
              ready
                ? "1px solid rgba(122,203,160,0.2)"
                : "1px solid rgba(126,177,255,0.14)",

            borderRadius:
              "999px",

            fontSize:
              "0.55rem",

            fontWeight:
              700,
          }}
        >
          {
            ready
              ? "✓ READY FOR ANALYSIS"
              : validate?.status ===
                  "blocked"
                ? "× VALIDATION BLOQUÉE"
                : "VALIDATION EN ATTENTE"
          }
        </span>
      </div>


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
            Nettoyage
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "4px",

              fontSize:
                "0.72rem",
            }}
          >
            {
              clean?.status ??
              "not_started"
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
            Validation
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "4px",

              fontSize:
                "0.72rem",
            }}
          >
            {
              validate?.status ??
              "not_started"
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
            Datasets validés
          </span>

          <strong
            style={{
              display:
                "block",

              marginTop:
                "4px",

              fontSize:
                "0.72rem",
            }}
          >
            {
              snapshot
                .validated_analysis_dataset_ids
                .length
            }
            {" / "}
            {
              snapshot
                .selected_analysis_dataset_ids
                .length
            }
          </strong>
        </article>
      </div>


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
            ready
              ? "Le readiness gate autorise maintenant l’analyse."
              : canValidate
                ? "La validation finale peut être exécutée côté serveur."
                : snapshot.next_stage
                  ? `Prochaine étape : ${snapshot.next_stage}.`
                  : "La préparation n’est pas encore validable."
          }
        </p>


        <button
          type="button"
          onClick={
            onValidate
          }
          disabled={
            ready ||
            loading ||
            !canValidate
          }
          style={{
            minWidth:
              "220px",

            padding:
              "9px 12px",

            border:
              ready
                ? "1px solid rgba(122,203,160,0.2)"
                : "1px solid rgba(126,177,255,0.2)",

            borderRadius:
              "9px",

            background:
              ready
                ? "rgba(122,203,160,0.09)"
                : "rgba(72,121,200,0.08)",

            color:
              "inherit",

            cursor:
              ready ||
              loading ||
              !canValidate
                ? "default"
                : "pointer",

            font:
              "inherit",

            fontSize:
              "0.64rem",

            fontWeight:
              750,

            opacity:
              !ready &&
              canValidate
                ? 1
                : 0.58,
          }}
        >
          {
            ready
              ? "Préparation validée"
              : loading
                ? "Validation…"
                : "Valider la préparation"
          }
        </button>
      </div>
    </section>
  );
}
