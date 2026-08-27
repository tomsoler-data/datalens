import type {
  PreparationSessionView,
  PreparationStageRecord,
} from "./preparationTypes";

import type { PreparationSubstep }
  from "./PreparationSubstepNavigation";


function hasCombineDiscoveryEvidence(
  session:
    PreparationSessionView |
    null
): boolean {
  if (
    session ===
    null
  ) {
    return false;
  }


  const combine =
    session
      .snapshot
      .stages
      .find(
        (
          stage
        ) =>
          stage.stage ===
          "combine"
      );


  return (
    combine
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


export function requiresCombineDiscoveryBeforeValidation(
  session:
    PreparationSessionView |
    null
): boolean {
  return (
    session !==
      null &&
    session
      .selected_analysis_dataset_ids
      .length >
      1 &&
    !hasCombineDiscoveryEvidence(
      session
    )
  );
}


export function findPreparationStage(
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


export function preparationSubstepFromSession(
  session:
    PreparationSessionView |
    null
): PreparationSubstep {
  if (
    session ===
    null
  ) {
    return "understand";
  }


  if (
    requiresCombineDiscoveryBeforeValidation(
      session
    )
  ) {
    return "combine";
  }


  switch (
    session.snapshot.next_stage
  ) {
    case "understand":
      return "understand";

    case "quality":
      return "quality";

    case "clean":
      return "cleaning";

    case "transform":
      return "transform";

    case "combine":
      return "combine";

    case "validate":
      return "finalization";

    default:
      return session
        .snapshot
        .ready_for_analysis
        ? "finalization"
        : "understand";
  }
}


export function preparationStageResolved(
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
