"use client";

import type {
  PreparationSessionView,
  PreparationStageRecord,
} from "./preparationTypes";

import styles from "./PreparationSubstepNavigation.module.css";


export type PreparationSubstep =
  | "understand"
  | "quality"
  | "cleaning"
  | "transform"
  | "combine"
  | "finalization";


type PreparationSubstepNavigationProps = {
  session:
    PreparationSessionView |
    null;

  activeStep:
    PreparationSubstep;

  onStepChange:
    (
      step:
        PreparationSubstep
    ) => void;

  qualityReady:
    boolean;

  cleaningPlanReady:
    boolean;

  cleaningActionCount:
    number;

  cleaningApplied:
    boolean;

  semanticReviewReady:
    boolean;

  semanticReviewExpectedCount:
    number;

  semanticDecisionCount:
    number;

  semanticConfirmed:
    boolean;
};


type VisualStatus =
  | "done"
  | "current"
  | "attention"
  | "skipped"
  | "waiting"
  | "locked";


type StepDefinition = {
  id:
    PreparationSubstep;

  index:
    string;

  label:
    string;

  shortLabel:
    string;

  description:
    string;
};


const STEPS:
  StepDefinition[] = [
    {
      id:
        "understand",

      index:
        "01",

      label:
        "Comprendre",

      shortLabel:
        "Comprendre",

      description:
        "Examiner les jeux de données, leur structure et leur grain avant toute décision de préparation.",
    },

    {
      id:
        "quality",

      index:
        "02",

      label:
        "Qualité",

      shortLabel:
        "Qualité",

      description:
        "Mesurer les valeurs manquantes, doublons, incohérences et autres signaux de qualité sans modifier les données.",
    },

    {
      id:
        "cleaning",

      index:
        "03",

      label:
        "Nettoyer",

      shortLabel:
        "Nettoyer",

      description:
        "Appliquer les corrections déterministes et traiter les cas ambigus avec une revue sémantique contrôlée.",
    },

    {
      id:
        "transform",

      index:
        "04",

      label:
        "Transformer",

      shortLabel:
        "Transformer",

      description:
        "Créer les variables dérivées, conversions, classes, extractions temporelles et agrégations utiles à l’analyse.",
    },

    {
      id:
        "combine",

      index:
        "05",

      label:
        "Assembler",

      shortLabel:
        "Assembler",

      description:
        "Vérifier les clés des tables et leurs relations avant de les assembler.",
    },

    {
      id:
        "finalization",

      index:
        "06",

      label:
        "Finaliser",

      shortLabel:
        "Finaliser",

      description:
        "Sélectionner la sortie analytique finale, vérifier les preuves de préparation et autoriser l’analyse.",
    },
  ];


function findStage(
  session:
    PreparationSessionView |
    null,

  stageName:
    PreparationStageRecord[
      "stage"
    ]
): PreparationStageRecord | null {
  if (
    session ===
    null
  ) {
    return null;
  }


  return (
    session
      .snapshot
      .stages
      .find(
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
  stage:
    PreparationStageRecord |
    null
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
  status:
    VisualStatus,

  step:
    PreparationSubstep
): string {
  switch (
    status
  ) {
    case "done":
      if (
        step ===
        "cleaning"
      ) {
        return "Terminé";
      }

      if (
        step ===
        "finalization"
      ) {
        return "Validé";
      }

      return "Terminé";

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
  status:
    VisualStatus
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
  semanticReviewExpectedCount,
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


  const understandResolved =
    isResolved(
      understand
    );


  /*
   * The server-owned Preparation stage is authoritative.
   *
   * `qualityReady` is volatile React UI state and may be false
   * immediately after F5 even though QUALITY is already passed.
   * Keep the UI flag only as a compatibility signal while the
   * current page is still mounted.
   */
  const qualityResolved =
    isResolved(
      quality
    ) ||
    (
      qualityReady &&
      quality?.status ===
        "passed"
    );


  /*
   * CLEAN is also server-owned.
   *
   * The detailed cleaning / semantic objects are useful while
   * the stage is unresolved, but they must not re-lock a stage
   * that the backend has already marked passed or skipped.
   */
  const cleanStageResolved =
    isResolved(
      clean
    );


  const semanticReviewPending =
    !cleanStageResolved &&
    (
      (
        semanticReviewExpectedCount >
          0 &&
        !semanticReviewReady
      ) ||
      (
        semanticReviewReady &&
        semanticDecisionCount >
          0 &&
        !semanticConfirmed
      )
    );


  const cleaningActionPending =
    !cleanStageResolved &&
    cleaningPlanReady &&
    cleaningActionCount >
      0 &&
    !cleaningApplied;


  const cleaningNeedsAttention =
    !cleanStageResolved &&
    (
      clean?.status ===
        "review_required" ||
      clean?.status ===
        "blocked" ||
      cleaningActionPending ||
      semanticReviewPending
    );


  const cleanResolved =
    cleanStageResolved;


  const transformResolved =
    isResolved(
      transform
    );


  const combineResolved =
    isResolved(
      combine
    );


  const finalizationDone =
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
    !finalizationDone;


  const statusByStep:
    Record<
      PreparationSubstep,
      VisualStatus
    > = {
      understand:
        understandResolved
          ? "done"
          : activeStep ===
              "understand"
            ? "current"
            : "waiting",

      quality:
        !understandResolved
          ? "locked"
          : qualityResolved
            ? "done"
            : quality?.status ===
                "blocked" ||
              quality?.status ===
                "review_required"
              ? "attention"
              : activeStep ===
                  "quality"
                ? "current"
                : "waiting",

      cleaning:
        !qualityResolved
          ? "locked"
          : cleaningNeedsAttention
            ? "attention"
            : clean?.status ===
                "skipped"
              ? "skipped"
              : cleanResolved
                ? "done"
                : activeStep ===
                    "cleaning"
                  ? "current"
                  : "waiting",

      transform:
        !cleanResolved
          ? "locked"
          : transform?.status ===
              "skipped"
            ? "skipped"
            : transformResolved
              ? "done"
              : transform?.status ===
                    "blocked" ||
                  transform?.status ===
                    "review_required"
                ? "attention"
                : activeStep ===
                    "transform"
                  ? "current"
                  : "waiting",

      combine:
        !transformResolved
          ? "locked"
          : combine?.status ===
              "skipped" &&
            !combineDiscoveryPending
            ? "skipped"
            : combineResolved &&
                !combineDiscoveryPending
              ? "done"
              : combine?.status ===
                    "blocked" ||
                  combine?.status ===
                    "review_required" ||
                  combineDiscoveryPending
                ? "attention"
                : activeStep ===
                    "combine"
                  ? "current"
                  : "waiting",

      finalization:
        finalizationDone
          ? "done"
          : combineResolved &&
              !combineDiscoveryPending
            ? activeStep ===
                "finalization"
              ? "current"
              : "waiting"
            : "locked",
    };


  const availabilityByStep:
    Record<
      PreparationSubstep,
      boolean
    > = {
      understand:
        true,

      quality:
        understandResolved,

      cleaning:
        qualityResolved,

      transform:
        cleanResolved,

      combine:
        transformResolved,

      finalization:
        finalizationDone ||
        (
          combineResolved &&
          !combineDiscoveryPending
        ) ||
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
        <div
          className={
            styles.headerContent
          }
        >
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
          /6
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
