from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.estimator_contracts import (
    estimator_problem_type,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_COMPARISON_CONTRACT_RULE_VERSION = (
    "ml_model_comparison_contract_v0.1"
)


# ============================================================
# RANKING POLICY
# ============================================================


MLModelComparisonPrimaryMetric = Literal[
    "rmse",
    "f1_macro",
]


MLModelComparisonRankingPolicy = Literal[
    "regression_rmse_v0.1",
    "classification_f1_macro_v0.1",
]


REGRESSION_RANKING_KEYS = (
    "rmse:asc",
    "mae:asc",
    "r2:desc",
    "estimator_key:asc",
)


CLASSIFICATION_RANKING_KEYS = (
    "f1_macro:desc",
    "accuracy:desc",
    "estimator_key:asc",
)


# ============================================================
# MODEL COMPARISON CONTRACT
# ============================================================


class MLModelComparisonContract(
    BaseModel
):
    """
    Server-validatable contract for deterministic comparison of
    multiple fixed Classical ML estimators.

    v0.1 deliberately compares already-specified estimator
    configurations.

    It does NOT perform:

    - grid search;
    - random search;
    - Bayesian optimization;
    - automatic hyperparameter tuning;
    - cross-validation;
    - time-series validation.

    Every candidate MUST operate on exactly the same:

    - Preparation workflow;
    - validated dataset;
    - problem type;
    - target;
    - ordered feature surface;
    - categorical feature roles;
    - preprocessing policy;
    - train/test split policy.

    This ensures that candidate metrics are directly comparable.

    Estimator keys must be unique in v0.1. Comparing several
    hyperparameter variants of the same estimator belongs to a
    later experiment/search contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    candidates: list[
        MLTrainingContract
    ] = Field(
        min_length=2,
        max_length=5,
    )


    rule_version: Literal[
        "ml_model_comparison_contract_v0.1"
    ] = ML_MODEL_COMPARISON_CONTRACT_RULE_VERSION


    # ========================================================
    # REFERENCE CANDIDATE
    # ========================================================


    @property
    def reference_candidate(
        self,
    ) -> MLTrainingContract:
        return (
            self.candidates[
                0
            ]
        )


    # ========================================================
    # SHARED AUTHORITY
    # ========================================================


    @property
    def workflow_id(
        self,
    ) -> str:
        return (
            self
            .reference_candidate
            .workflow_id
        )


    @property
    def dataset_id(
        self,
    ) -> str:
        return (
            self
            .reference_candidate
            .dataset_id
        )


    @property
    def problem_type(
        self,
    ) -> Literal[
        "regression",
        "classification",
    ]:
        return (
            self
            .reference_candidate
            .problem_type
        )


    @property
    def target_column(
        self,
    ) -> str:
        return (
            self
            .reference_candidate
            .target_column
        )


    @property
    def feature_columns(
        self,
    ) -> list[
        str
    ]:
        return list(
            self
            .reference_candidate
            .feature_columns
        )


    @property
    def categorical_feature_columns(
        self,
    ) -> list[
        str
    ]:
        return list(
            self
            .reference_candidate
            .categorical_feature_columns
        )


    # ========================================================
    # RANKING AUTHORITY
    # ========================================================


    @property
    def primary_metric(
        self,
    ) -> MLModelComparisonPrimaryMetric:

        if (
            self.problem_type
            ==
            "regression"
        ):
            return "rmse"


        return "f1_macro"


    @property
    def ranking_policy(
        self,
    ) -> MLModelComparisonRankingPolicy:

        if (
            self.problem_type
            ==
            "regression"
        ):
            return (
                "regression_rmse_v0.1"
            )


        return (
            "classification_f1_macro_v0.1"
        )


    @property
    def ranking_keys(
        self,
    ) -> tuple[
        str,
        ...,
    ]:

        if (
            self.problem_type
            ==
            "regression"
        ):
            return (
                REGRESSION_RANKING_KEYS
            )


        return (
            CLASSIFICATION_RANKING_KEYS
        )


    # ========================================================
    # CROSS-CANDIDATE VALIDATION
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_comparison_contract(
        self,
    ) -> "MLModelComparisonContract":

        reference = (
            self.candidates[
                0
            ]
        )


        # ----------------------------------------------------
        # SUPPORTED ESTIMATORS + PROBLEM TYPE
        # ----------------------------------------------------


        seen_estimator_keys: set[
            str
        ] = set()


        for (
            index,
            candidate,
        ) in enumerate(
            self.candidates
        ):

            expected_problem_type = (
                estimator_problem_type(
                    candidate
                    .estimator_key
                )
            )


            if (
                expected_problem_type
                is None
            ):
                raise ValueError(
                    (
                        "Model comparison candidate uses "
                        "an unsupported estimator. "
                        f"candidate_index={index}, "
                        "estimator_key="
                        f"{candidate.estimator_key}"
                    )
                )


            if (
                expected_problem_type
                !=
                candidate.problem_type
            ):
                raise ValueError(
                    (
                        "Model comparison candidate "
                        "estimator/problem type mismatch. "
                        f"candidate_index={index}, "
                        "estimator_key="
                        f"{candidate.estimator_key}, "
                        "estimator_problem_type="
                        f"{expected_problem_type}, "
                        "candidate_problem_type="
                        f"{candidate.problem_type}"
                    )
                )


            if (
                candidate.estimator_key
                in
                seen_estimator_keys
            ):
                raise ValueError(
                    (
                        "Model Comparison v0.1 requires "
                        "unique estimator_key values. "
                        "Duplicate estimator_key="
                        f"{candidate.estimator_key}"
                    )
                )


            seen_estimator_keys.add(
                candidate
                .estimator_key
            )


        # ----------------------------------------------------
        # SAME WORKFLOW
        # ----------------------------------------------------


        for (
            index,
            candidate,
        ) in enumerate(
            self.candidates[
                1:
            ],
            start=1,
        ):

            if (
                candidate.workflow_id
                !=
                reference.workflow_id
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same workflow_id. "
                        f"candidate_index={index}"
                    )
                )


            # ------------------------------------------------
            # SAME DATASET
            # ------------------------------------------------


            if (
                candidate.dataset_id
                !=
                reference.dataset_id
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same dataset_id. "
                        f"candidate_index={index}"
                    )
                )


            # ------------------------------------------------
            # SAME PROBLEM TYPE
            # ------------------------------------------------


            if (
                candidate.problem_type
                !=
                reference.problem_type
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same problem_type. "
                        f"candidate_index={index}"
                    )
                )


            # ------------------------------------------------
            # SAME TARGET
            # ------------------------------------------------


            if (
                candidate.target_column
                !=
                reference.target_column
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same target_column. "
                        f"candidate_index={index}"
                    )
                )


            # ------------------------------------------------
            # SAME ORDERED FEATURE SURFACE
            # ------------------------------------------------


            if (
                candidate.feature_columns
                !=
                reference.feature_columns
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same ordered "
                        "feature_columns. "
                        f"candidate_index={index}"
                    )
                )


            # ------------------------------------------------
            # SAME CATEGORICAL ROLES
            # ------------------------------------------------


            if (
                candidate
                .categorical_feature_columns
                !=
                reference
                .categorical_feature_columns
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same "
                        "categorical_feature_columns. "
                        f"candidate_index={index}"
                    )
                )


            # ------------------------------------------------
            # SAME PREPROCESSING
            # ------------------------------------------------


            if (
                candidate.preprocessing
                !=
                reference.preprocessing
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same preprocessing "
                        "contract. "
                        f"candidate_index={index}"
                    )
                )


            # ------------------------------------------------
            # SAME HOLDOUT SPLIT
            # ------------------------------------------------


            if (
                candidate.split
                !=
                reference.split
            ):
                raise ValueError(
                    (
                        "All comparison candidates must "
                        "use the same split contract. "
                        f"candidate_index={index}"
                    )
                )


        return self