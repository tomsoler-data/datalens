from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


from app.ml.estimator_contracts import (
    MLEstimatorHyperparameters,
    default_estimator_hyperparameters,
)


# ============================================================
# VERSION
# ============================================================


ML_TRAINING_CONTRACT_RULE_VERSION = (
    "ml_training_contract_v0.1"
)


ML_PREPROCESSING_CONTRACT_RULE_VERSION = (
    "ml_preprocessing_contract_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLProblemType = Literal[
    "regression",
    "classification",
]


MLSplitStrategy = Literal[
    "holdout",
]


MLNumericImputationStrategy = Literal[
    "error",
    "median",
]


MLCategoricalImputationStrategy = Literal[
    "error",
    "most_frequent",
]


MLCategoricalEncodingStrategy = Literal[
    "one_hot",
]


MLUnknownCategoryStrategy = Literal[
    "ignore",
]


# ============================================================
# SPLIT CONTRACT
# ============================================================


class MLSplitContract(
    BaseModel
):
    """
    Deterministic train/test split contract.

    DataLens v0.1 intentionally supports only a holdout split.

    Cross-validation and time-aware splitting belong to later
    contract versions.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    strategy: MLSplitStrategy = (
        "holdout"
    )


    test_size: float = Field(
        default=0.20,
        gt=0.0,
        lt=0.5,
    )


    random_seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )


    shuffle: bool = True


    stratify: bool = False


class MLGroupHoldoutSplitContract(
    BaseModel
):
    """
    Deterministic entity-aware train/test split.

    group_column identifies repeated entities whose observations
    must remain entirely inside one side of the holdout.

    The group column is split metadata, never a model feature.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    strategy: Literal[
        "group_holdout",
    ] = "group_holdout"


    group_column: str = Field(
        min_length=1,
    )


    test_size: float = Field(
        default=0.20,
        gt=0.0,
        lt=0.5,
    )


    random_seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )


    shuffle: Literal[
        True
    ] = True


    stratify: Literal[
        False
    ] = False


    @field_validator(
        "group_column",
        mode="before",
    )
    @classmethod
    def normalize_group_column(
        cls,
        value: object,
    ) -> str:

        normalized = str(
            value
            if value is not None
            else ""
        ).strip()


        if not normalized:

            raise ValueError(
                "group_column cannot be empty"
            )


        return normalized


MLTrainingSplitContract = (
    MLSplitContract
    |
    MLGroupHoldoutSplitContract
)


# ============================================================
# PREPROCESSING CONTRACT
# ============================================================


class MLPreprocessingContract(
    BaseModel
):
    """
    Deterministic preprocessing policy for Classical ML.

    This contract describes learned preprocessing behavior only.

    Data-dependent statistics such as:

    - numeric medians;
    - categorical modes;
    - one-hot vocabularies;
    - scaling means / variances;

    MUST NOT be supplied by callers.

    Those values are learned by the scikit-learn pipeline from
    the training split only.

    This prevents train/test leakage and keeps preprocessing
    reproducible inside the persisted Model Artifact.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    numeric_imputation: (
        MLNumericImputationStrategy
    ) = "error"


    categorical_imputation: (
        MLCategoricalImputationStrategy
    ) = "error"


    categorical_encoding: (
        MLCategoricalEncodingStrategy
    ) = "one_hot"


    handle_unknown_categories: (
        MLUnknownCategoryStrategy
    ) = "ignore"


    scale_numeric: bool = True


    rule_version: Literal[
        "ml_preprocessing_contract_v0.1"
    ] = ML_PREPROCESSING_CONTRACT_RULE_VERSION


# ============================================================
# TRAINING CONTRACT
# ============================================================


class MLTrainingContract(
    BaseModel
):
    """
    Server-validatable contract describing one deterministic
    classical machine-learning training request.

    The contract contains configuration and provenance only.

    It must never contain:
    - raw dataset rows;
    - trained model bytes;
    - predictions;
    - secrets;
    - arbitrary executable code;
    - learned preprocessing statistics;
    - arbitrary estimator kwargs.

    feature_columns defines the complete ordered model feature
    surface.

    categorical_feature_columns explicitly marks the subset that
    must be treated as categorical.

    Every remaining feature is therefore numeric.

    estimator_hyperparameters is optional for backward
    compatibility.

    When omitted, DataLens can resolve one server-owned default
    contract for supported estimator keys.

    Unknown estimator keys remain constructible at this layer so
    that the existing executor continues to own its fail-closed
    unsupported-estimator behavior until the executor milestone
    is upgraded.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    problem_type: MLProblemType


    target_column: str = Field(
        min_length=1,
    )


    feature_columns: list[
        str
    ] = Field(
        min_length=1,
    )


    categorical_feature_columns: list[
        str
    ] = Field(
        default_factory=list,
    )


    estimator_key: str = Field(
        min_length=1,
    )


    estimator_hyperparameters: (
        MLEstimatorHyperparameters
        |
        None
    ) = None


    preprocessing: MLPreprocessingContract = Field(
        default_factory=MLPreprocessingContract,
    )


    split: MLTrainingSplitContract = Field(
        default_factory=MLSplitContract,
    )


    rule_version: Literal[
        "ml_training_contract_v0.1"
    ] = ML_TRAINING_CONTRACT_RULE_VERSION


    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================


    @field_validator(
        "workflow_id",
        "dataset_id",
        "target_column",
        "estimator_key",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
    ) -> str:
        normalized = str(
            value
            if value is not None
            else ""
        ).strip()


        if not normalized:
            raise ValueError(
                "value cannot be empty"
            )


        return normalized


    # ========================================================
    # FEATURE NORMALIZATION
    # ========================================================


    @field_validator(
        "feature_columns",
        "categorical_feature_columns",
        mode="before",
    )
    @classmethod
    def normalize_feature_column_lists(
        cls,
        value: object,
        info,
    ) -> list[
        str
    ]:
        field_name = str(
            info.field_name
        )


        if (
            value is None
            and
            field_name
            ==
            "categorical_feature_columns"
        ):
            return []


        if not isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            raise ValueError(
                (
                    f"{field_name} must be "
                    "a list of column names"
                )
            )


        normalized: list[
            str
        ] = []


        seen: set[
            str
        ] = set()


        for raw_column in value:
            column = str(
                raw_column
                if raw_column is not None
                else ""
            ).strip()


            if not column:
                raise ValueError(
                    (
                        f"{field_name} cannot "
                        "contain an empty name"
                    )
                )


            if column in seen:
                raise ValueError(
                    (
                        f"{field_name} cannot "
                        "contain duplicates: "
                        f"{column!r}"
                    )
                )


            seen.add(
                column
            )


            normalized.append(
                column
            )


        if (
            field_name
            ==
            "feature_columns"
            and
            not normalized
        ):
            raise ValueError(
                (
                    "feature_columns must contain "
                    "at least one column"
                )
            )


        return normalized


    # ========================================================
    # DERIVED FEATURE ROLES
    # ========================================================


    @property
    def numeric_feature_columns(
        self,
    ) -> list[
        str
    ]:
        categorical = set(
            self.categorical_feature_columns
        )


        return [
            column

            for column
            in self.feature_columns

            if column
            not in
            categorical
        ]


    # ========================================================
    # EFFECTIVE ESTIMATOR CONFIGURATION
    # ========================================================


    @property
    def effective_estimator_hyperparameters(
        self,
    ) -> (
        MLEstimatorHyperparameters
        |
        None
    ):
        """
        Return an explicitly supplied estimator configuration or
        the DataLens-owned deterministic default.

        Unknown estimator keys return None.

        This property calculates configuration only. It does not
        instantiate or execute sklearn objects.
        """

        if (
            self.estimator_hyperparameters
            is not None
        ):
            return (
                self.estimator_hyperparameters
            )


        return (
            default_estimator_hyperparameters(
                self.estimator_key
            )
        )


    # ========================================================
    # CROSS-FIELD CONTRACT
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_training_contract(
        self,
    ) -> "MLTrainingContract":

        # ----------------------------------------------------
        # TARGET LEAKAGE
        # ----------------------------------------------------

        if (
            self.target_column
            in
            self.feature_columns
        ):
            raise ValueError(
                (
                    "target_column cannot also "
                    "be present in feature_columns"
                )
            )


        # ----------------------------------------------------
        # FEATURE ROLE AUTHORITY
        # ----------------------------------------------------

        feature_columns = set(
            self.feature_columns
        )


        unknown_categorical_columns = [
            column

            for column
            in self.categorical_feature_columns

            if column
            not in
            feature_columns
        ]


        if unknown_categorical_columns:
            raise ValueError(
                (
                    "categorical_feature_columns must "
                    "be a subset of feature_columns. "
                    "Unknown categorical features: "
                    +
                    ", ".join(
                        unknown_categorical_columns
                    )
                )
            )


        # ----------------------------------------------------
        # ESTIMATOR HYPERPARAMETER IDENTITY
        # ----------------------------------------------------

        if (
            self.estimator_hyperparameters
            is not None
            and
            self.estimator_hyperparameters.kind
            !=
            self.estimator_key
        ):
            raise ValueError(
                (
                    "estimator_hyperparameters kind "
                    "must match estimator_key. "
                    "estimator_key="
                    f"{self.estimator_key}, "
                    "hyperparameters_kind="
                    f"{self.estimator_hyperparameters.kind}"
                )
            )


        # ----------------------------------------------------
        # ENTITY-AWARE GROUP ROLE
        # ----------------------------------------------------

        if isinstance(
            self.split,
            MLGroupHoldoutSplitContract,
        ):

            if (
                self.split.group_column
                ==
                self.target_column
            ):

                raise ValueError(
                    (
                        "group_column cannot also "
                        "be target_column"
                    )
                )


            if (
                self.split.group_column
                in
                self.feature_columns
            ):

                raise ValueError(
                    (
                        "group_column cannot also "
                        "be present in feature_columns"
                    )
                )


        # ----------------------------------------------------
        # REGRESSION SPLIT
        # ----------------------------------------------------

        if (
            self.problem_type
            ==
            "regression"
            and
            self.split.stratify
        ):
            raise ValueError(
                (
                    "stratify=True is not supported "
                    "for regression contracts"
                )
            )


        return self