type NullSetter =
  (value: null) => void;


type FalseSetter =
  (value: false) => void;


type EmptyStringListSetter =
  (value: string[]) => void;


export type PreparationPipelineResetOptions = {
  resetIngestionLoading?: boolean;

  resetQualityLoading?: boolean;

  resetCleaningPlanLoading?: boolean;

  resetCleaningApplyLoading?: boolean;

  resetPreparedExportError?: boolean;

  resetPreparedExportLoading?: boolean;
};


export function createPreparationPipelineReset({
  setIngestion,
  setIngestionLoading,
  setPreparationSession,
  setPreparationSessionError,
  setPreparationSessionLoading,
  setQualityReport,
  setQualityError,
  setQualityLoading,
  setCleaningPlan,
  setCleaningPlanError,
  setCleaningPlanLoading,
  setSelectedCleaningActionIds,
  setCleaningExecution,
  setAppliedCleaningActionIds,
  setCleaningApplyError,
  setCleaningApplyLoading,
  setPreparedExportError,
  setPreparedExportLoading,
}: {
  setIngestion: NullSetter;

  setIngestionLoading: FalseSetter;

  setPreparationSession: NullSetter;

  setPreparationSessionError: NullSetter;

  setPreparationSessionLoading: FalseSetter;

  setQualityReport: NullSetter;

  setQualityError: NullSetter;

  setQualityLoading: FalseSetter;

  setCleaningPlan: NullSetter;

  setCleaningPlanError: NullSetter;

  setCleaningPlanLoading: FalseSetter;

  setSelectedCleaningActionIds:
    EmptyStringListSetter;

  setCleaningExecution: NullSetter;

  setAppliedCleaningActionIds:
    EmptyStringListSetter;

  setCleaningApplyError: NullSetter;

  setCleaningApplyLoading: FalseSetter;

  setPreparedExportError: NullSetter;

  setPreparedExportLoading: FalseSetter;
}) {
  return function resetPreparationPipelineState({
    resetIngestionLoading = false,
    resetQualityLoading = false,
    resetCleaningPlanLoading = false,
    resetCleaningApplyLoading = false,
    resetPreparedExportError = false,
    resetPreparedExportLoading = false,
  }: PreparationPipelineResetOptions = {}) {
    setIngestion(
      null
    );


    if (
      resetIngestionLoading
    ) {
      setIngestionLoading(
        false
      );
    }


    setPreparationSession(
      null
    );


    setPreparationSessionError(
      null
    );


    setPreparationSessionLoading(
      false
    );


    setQualityReport(
      null
    );


    setQualityError(
      null
    );


    if (
      resetQualityLoading
    ) {
      setQualityLoading(
        false
      );
    }


    setCleaningPlan(
      null
    );


    setCleaningPlanError(
      null
    );


    if (
      resetCleaningPlanLoading
    ) {
      setCleaningPlanLoading(
        false
      );
    }


    setSelectedCleaningActionIds(
      []
    );


    setCleaningExecution(
      null
    );


    setAppliedCleaningActionIds(
      []
    );


    setCleaningApplyError(
      null
    );


    if (
      resetCleaningApplyLoading
    ) {
      setCleaningApplyLoading(
        false
      );
    }


    if (
      resetPreparedExportError
    ) {
      setPreparedExportError(
        null
      );
    }


    if (
      resetPreparedExportLoading
    ) {
      setPreparedExportLoading(
        false
      );
    }
  };
}
