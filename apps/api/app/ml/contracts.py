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


# ============================================================
# VERSION
# ============================================================


ML_TRAINING_CONTRACT_RULE_VERSION = (
    "ml_training_contract_v0.1"
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
    - arbitrary executable code.

    The future ML executor is responsible for resolving the
    server-owned Preparation artifact identified by workflow_id
    and dataset_id.
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


    estimator_key: str = Field(
        min_length=1,
    )


    split: MLSplitContract = Field(
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
        mode="before",
    )
    @classmethod
    def normalize_feature_columns(
        cls,
        value: object,
    ) -> list[
        str
    ]:
        if not isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            raise ValueError(
                (
                    "feature_columns must be "
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
                        "feature_columns cannot "
                        "contain an empty name"
                    )
                )


            if column in seen:
                raise ValueError(
                    (
                        "feature_columns cannot "
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


        if not normalized:
            raise ValueError(
                (
                    "feature_columns must contain "
                    "at least one column"
                )
            )


        return normalized


    # ========================================================
    # CROSS-FIELD CONTRACT
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_training_contract(
        self,
    ) -> "MLTrainingContract":

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