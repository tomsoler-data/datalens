"use client";

import Link from "next/link";

import type { WorkspaceStep }
  from "./workspaceNavigationTypes";

import styles
  from "../../app/page.module.css";


type WorkspaceStepDefinition = {
  id:
    WorkspaceStep;

  label:
    string;

  shortLabel:
    string;
};


const WORKSPACE_STEPS:
  WorkspaceStepDefinition[] = [
    {
      id:
        "data",

      label:
        "Données",

      shortLabel:
        "1",
    },

    {
      id:
        "documents",

      label:
        "Documents",

      shortLabel:
        "2",
    },

    {
      id:
        "preparation",

      label:
        "Préparation",

      shortLabel:
        "3",
    },

    {
      id:
        "analyses",

      label:
        "Analyses",

      shortLabel:
        "4",
    },

    {
      id:
        "report",

      label:
        "Rapport",

      shortLabel:
        "5",
    },
  ];


export default function WorkspaceNavigation({
  activeStep,
  onStepChange,
  dataReady,
  reportReady,
  interventionCount,
}: {
  activeStep:
    WorkspaceStep;

  onStepChange:
    (
      step:
        WorkspaceStep
    ) => void;

  dataReady:
    boolean;

  reportReady:
    boolean;

  interventionCount:
    number;
}) {
  function isEnabled(
    step:
      WorkspaceStep
  ): boolean {
    if (
      step ===
      "data"
    ) {
      return true;
    }


    if (
      step ===
      "documents"
    ) {
      return dataReady;
    }


    if (
      step ===
      "preparation"
    ) {
      return dataReady;
    }


    return reportReady;
  }


  return (
    <nav
      aria-label="Navigation du workspace"
      className={
        styles.workspaceNav
      }
    >
      <div
        role="group"
        aria-label="Étapes d’analyse"
        className={
          styles.workspaceSteps
        }
      >
        {
          WORKSPACE_STEPS.map(
            (
              step
            ) => {
              const active =
                step.id ===
                activeStep;

              const enabled =
                isEnabled(
                  step.id
                );

              const showIntervention =
                step.id ===
                  "analyses" &&
                interventionCount >
                  0;


              return (
                <button
                  key={
                    step.id
                  }
                  type="button"
                  aria-disabled={
                    !enabled
                  }
                  aria-current={
                    active
                      ? "step"
                      : undefined
                  }
                  className={
                    `${styles.workspaceStep} ${
                      active
                        ? styles.workspaceStepActive
                        : ""
                    } ${
                      !enabled
                        ? styles.workspaceStepDisabled
                        : ""
                    }`
                  }
                  onClick={
                    () => {
                      if (
                        !enabled
                      ) {
                        return;
                      }


                      onStepChange(
                        step.id
                      );
                    }
                  }
                >
                  <span
                    aria-hidden="true"
                    className={
                      styles.workspaceStepNumber
                    }
                  >
                    {
                      step.shortLabel
                    }
                  </span>

                  <span
                    className={
                      styles.workspaceStepLabel
                    }
                  >
                    {
                      step.label
                    }
                  </span>

                  {
                    showIntervention
                      ? (
                          <span
                            className={
                              styles.workspaceInterventionDot
                            }
                            title={
                              `${interventionCount} intervention(s) requise(s)`
                            }
                          />
                        )
                      : null
                  }
                </button>
              );
            }
          )
        }
      </div>


      <Link
        href="/observability"
        target="_blank"
        rel="noreferrer"
        title="Ouvrir l’observabilité IA locale dans un nouvel onglet"
        className={
          styles.observabilityLink
        }
      >
        <span
          aria-hidden="true"
          className={
            styles.observabilityIcon
          }
        >
          AI
        </span>

        <span>
          Observabilité
          {" ↗"}
        </span>
      </Link>
    </nav>
  );
}
