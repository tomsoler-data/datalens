"use client";

import type {
  PreparationSessionView,
  PreparationStageRecord,
} from "./preparationTypes";

import styles from "./PreparationSubstepNavigation.module.css";


export type PreparationSubstep =
  | "diagnostic"
  | "cleaning"
  | "semantic"
  | "transform"
  | "validation";


type PreparationSubstepNavigationProps = {
  session: PreparationSessionView | null;
  activeStep: PreparationSubstep;
  onStepChange: (
    step: PreparationSubstep
  ) => void;
  qualityReady: boolean;
  cleaningPlanReady: boolean;
  cleaningActionCount: number;
  cleaningApplied: boolean;
  semanticReviewReady: boolean;
  semanticDecisionCount: number;
  semanticConfirmed: boolean;
};


type VisualStatus =
  | "done"
  | "current"
  | "attention"
  | "skipped"
  | "waiting"
  | "locked";


type StepDefinition = {
  id: PreparationSubstep;
  index: string;
  label: string;
  shortLabel: string;
  description: string;
};


const STEPS: StepDefinition[] = [
  {
    id: "diagnostic",
    index: "01",
    label: "Diagnostic",
    shortLabel: "Diagnostic",
    description:
      "Comprendre la structure des données et contrôler leur qualité avant toute modification.",
  },
  {
    id: "cleaning",
    index: "02",
    label: "Nettoyage",
    shortLabel: "Nettoyage",
    description:
      "Examiner puis appliquer uniquement les corrections déterministes, sûres et traçables.",
  },
  {
    id: "semantic",
    index: "03",
    label: "Revue sémantique",
    shortLabel: "Revue IA",
    description:
      "Contextualiser les cas ambigus avec le modèle local, puis conserver la décision finale côté analyste.",
  },
  {
    id: "transform",
    index: "04",
    label: "Transformer & combiner",
    shortLabel: "Transformer",
    description:
      "Préparer les agrégations, variables dérivées et combinaisons de datasets nécessaires au plan analytique.",
  },
  {
    id: "validation",
    index: "05",
    label: "Validation",
    shortLabel: "Validation",
    description:
      "Vérifier les preuves de préparation et autoriser explicitement l’entrée dans le moteur analytique.",
  },
];


function findStage(
  session: PreparationSessionView | null,
  stageName: PreparationStageRecord["stage"]
): PreparationStageRecord | null {
  if (
    session ===
    null
  ) {
    return null;
  }


  return (
    session.snapshot.stages.find(
      (
        stage
      ) =>
        stage.stage ===
        stageName
    ) ??
    null
  );
}


function isResolved(
  stage: PreparationStageRecord | null
): boolean {
  return (
    stage?.status ===
      "passed" ||
    stage?.status ===
      "skipped"
  );
}


function hasCombineDiscoveryEvidence(
  stage:
    PreparationStageRecord |
    null
): boolean {
  return (
    stage
      ?.evidence_refs
      .some(
        (
          reference
        ) =>
          reference.startsWith(
            "combine_service:"
          )
      ) ??
    false
  );
}


function statusLabel(
  status: VisualStatus,
  step: PreparationSubstep
): string {
  switch (
    status
  ) {
    case "done":
      return step ===
        "cleaning"
        ? "Terminé"
        : "Validé";

    case "current":
      return "En cours";

    case "attention":
      return "Action requise";

    case "skipped":
      return "Non requis";

    case "waiting":
      return "À faire";

    default:
      return "Verrouillé";
  }
}


function statusSymbol(
  status: VisualStatus
): string {
  switch (
    status
  ) {
    case "done":
      return "✓";

    case "attention":
      return "!";

    case "skipped":
      return "○";

    case "current":
      return "●";

    case "waiting":
      return "·";

    default:
      return "–";
  }
}


export default function PreparationSubstepNavigation({
  session,
  activeStep,
  onStepChange,
  qualityReady,
  cleaningPlanReady,
  cleaningActionCount,
  cleaningApplied,
  semanticReviewReady,
  semanticDecisionCount,
  semanticConfirmed,
}: PreparationSubstepNavigationProps) {
  const understand =
    findStage(
      session,
      "understand"
    );

  const quality =
    findStage(
      session,
      "quality"
    );

  const clean =
    findStage(
      session,
      "clean"
    );

  const transform =
    findStage(
      session,
      "transform"
    );

  const combine =
    findStage(
      session,
      "combine"
    );

  const validate =
    findStage(
      session,
      "validate"
    );


  const diagnosticDone =
    qualityReady &&
    understand?.status ===
      "passed" &&
    quality?.status ===
      "passed";


  const cleaningDone =
    cleaningPlanReady &&
    (
      cleaningActionCount ===
        0 ||
      cleaningApplied
    );


  const cleanResolved =
    isResolved(
      clean
    );


  const validationDone =
    session?.snapshot
      .ready_for_analysis ===
      true ||
    validate?.status ===
      "passed";


  const multipleSourceDatasets =
    (
      session
        ?.selected_analysis_dataset_ids
        .length ??
      0
    ) >
    1;


  const combineDiscoveryRecorded =
    hasCombineDiscoveryEvidence(
      combine
    );


  const combineDiscoveryPending =
    multipleSourceDatasets &&
    !combineDiscoveryRecorded &&
    !validationDone;


  const transformResolved =
    !combineDiscoveryPending &&
    isResolved(
      transform
    ) &&
    isResolved(
      combine
    );


  const transformSkipped =
    !combineDiscoveryPending &&
    transform?.status ===
      "skipped" &&
    combine?.status ===
      "skipped";


  const semanticSkipped =
    cleanResolved &&
    !semanticReviewReady;


  const semanticDone =
    semanticConfirmed ||
    (
      cleanResolved &&
      semanticReviewReady &&
      semanticDecisionCount ===
        0
    );


  const semanticReviewRequired =
    cleaningDone &&
    clean?.status ===
      "review_required" &&
    !semanticDone;


  const statusByStep: Record<
    PreparationSubstep,
    VisualStatus
  > = {
    diagnostic:
      diagnosticDone
        ? "done"
        : activeStep ===
            "diagnostic"
          ? "current"
          : "waiting",

    cleaning:
      !diagnosticDone
        ? "locked"
        : cleaningDone
          ? "done"
          : cleaningPlanReady &&
              cleaningActionCount >
                0
            ? "attention"
            : activeStep ===
                "cleaning"
              ? "current"
              : "waiting",

    semantic:
      !cleaningDone
        ? "locked"
        : semanticDone
          ? "done"
          : semanticReviewRequired
            ? "attention"
            : semanticSkipped
              ? "skipped"
              : semanticReviewReady
                ? "attention"
                : activeStep ===
                    "semantic"
                  ? "current"
                  : "waiting",

    transform:
      !cleanResolved
        ? "locked"
        : combineDiscoveryPending
          ? activeStep ===
              "transform"
            ? "current"
            : "waiting"
          : transformSkipped
            ? "skipped"
            : transformResolved
              ? "done"
              : (
                  transform?.status ===
                    "blocked" ||
                  transform?.status ===
                    "review_required" ||
                  combine?.status ===
                    "blocked" ||
                  combine?.status ===
                    "review_required"
                )
                ? "attention"
                : activeStep ===
                    "transform"
                  ? "current"
                  : "waiting",

    validation:
      validationDone
        ? "done"
        : combineDiscoveryPending
          ? "locked"
          : session?.snapshot
              .next_stage ===
              "validate"
            ? activeStep ===
                "validation"
              ? "current"
              : "waiting"
            : "locked",
  };


  const availabilityByStep: Record<
    PreparationSubstep,
    boolean
  > = {
    diagnostic:
      true,

    cleaning:
      diagnosticDone,

    semantic:
      cleaningDone,

    transform:
      cleanResolved,

    validation:
      validationDone ||
      (
        !combineDiscoveryPending &&
        session?.snapshot
          .next_stage ===
          "validate"
      ),
  };


  const activeDefinition =
    STEPS.find(
      (
        step
      ) =>
        step.id ===
        activeStep
    ) ??
    STEPS[
      0
    ];


  return (
    <section
      className={
        styles.wrapper
      }
      aria-label="Étapes de préparation"
    >
      <div
        className={
          styles.header
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Workflow de préparation
          </span>

          <strong
            className={
              styles.currentTitle
            }
          >
            {
              activeDefinition.index
            }
            {" · "}
            {
              activeDefinition.label
            }
          </strong>

          <p
            className={
              styles.currentDescription
            }
          >
            {
              activeDefinition.description
            }
          </p>
        </div>

        <span
          className={
            styles.progress
          }
        >
          {
            STEPS.findIndex(
              (
                step
              ) =>
                step.id ===
                activeStep
            ) +
            1
          }
          /5
        </span>
      </div>


      <div
        className={
          styles.scroller
        }
      >
        <div
          className={
            styles.steps
          }
        >
          {
            STEPS.map(
              (
                step
              ) => {
                const status =
                  statusByStep[
                    step.id
                  ];

                const available =
                  availabilityByStep[
                    step.id
                  ];

                const active =
                  activeStep ===
                  step.id;


                return (
                  <button
                    key={
                      step.id
                    }
                    className={
                      `${styles.step} ${
                        styles[
                          status
                        ]
                      } ${
                        active
                          ? styles.active
                          : ""
                      }`
                    }
                    type="button"
                    disabled={
                      !available
                    }
                    aria-current={
                      active
                        ? "step"
                        : undefined
                    }
                    onClick={
                      () =>
                        onStepChange(
                          step.id
                        )
                    }
                  >
                    <span
                      className={
                        styles.stepTop
                      }
                    >
                      <span
                        className={
                          styles.index
                        }
                      >
                        {
                          step.index
                        }
                      </span>

                      <span
                        className={
                          styles.symbol
                        }
                        aria-hidden="true"
                      >
                        {
                          statusSymbol(
                            status
                          )
                        }
                      </span>
                    </span>

                    <strong
                      className={
                        styles.label
                      }
                    >
                      <span
                        className={
                          styles.desktopLabel
                        }
                      >
                        {
                          step.label
                        }
                      </span>

                      <span
                        className={
                          styles.mobileLabel
                        }
                      >
                        {
                          step.shortLabel
                        }
                      </span>
                    </strong>

                    <span
                      className={
                        styles.status
                      }
                    >
                      {
                        statusLabel(
                          status,
                          step.id
                        )
                      }
                    </span>
                  </button>
                );
              }
            )
          }
        </div>
      </div>
    </section>
  );
}
