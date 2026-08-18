from __future__ import annotations

from pathlib import Path


# ============================================================
# VERSION
# ============================================================

PROMOTION_VERSION = (
    "analytical_v1_production_promotion_v1.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parent
)


SOURCE_DIR = (
    BASE_DIR
    / "app"
    / "evals"
)


TARGET_DIR = (
    BASE_DIR
    / "app"
    / "planning"
    / "analytical_v1"
)


# ============================================================
# SOURCE Ã¢â€ â€™ PRODUCTION FILE MAP
#
# We intentionally do not copy:
#
#     app/evals/schemas.py
#
# because it also contains benchmark-specific classes such as
# AnalyticalEvalCase and AnalyticalExpectation.
#
# Production gets a minimal dataset_context.py instead.
# ============================================================

SOURCE_FILE_MAP = {
    "routing_relationships_v0_8.py":
        "relationships.py",

    "dataset_dependency_contract_v0_8.py":
        "dependency.py",

    "relationship_path_resolver_v0_8.py":
        "relationship_paths.py",

    "planner_structural_handoff_v0_9.py":
        "structural_handoff.py",

    "analytical_planner_context_v0_9.py":
        "context.py",

    "analytical_planner_input_v0_9.py":
        "input.py",

    "analytical_planner_contract_v0_9.py":
        "contract.py",

    "analytical_planner_validator_v0_9.py":
        "validator_base.py",

    "analytical_planner_validator_v0_9_1.py":
        "validator.py",

    "analytical_reference_canonicalizer_v1_0.py":
        "reference_canonicalizer.py",

    "analytical_planner_safety_pipeline_v1_0.py":
        "safety.py",
}


# ============================================================
# IMPORT REWRITES
#
# Only package boundaries change.
# Analytical logic remains exactly the tested implementation.
# ============================================================

IMPORT_REWRITES = {
    "from app.evals.schemas import (":
        (
            "from app.planning.analytical_v1."
            "dataset_context import ("
        ),

    "from app.evals.routing_relationships_v0_8 import (":
        (
            "from app.planning.analytical_v1."
            "relationships import ("
        ),

    "from app.evals.dataset_dependency_contract_v0_8 import (":
        (
            "from app.planning.analytical_v1."
            "dependency import ("
        ),

    "from app.evals.relationship_path_resolver_v0_8 import (":
        (
            "from app.planning.analytical_v1."
            "relationship_paths import ("
        ),

    "from app.evals.planner_structural_handoff_v0_9 import (":
        (
            "from app.planning.analytical_v1."
            "structural_handoff import ("
        ),

    "from app.evals.analytical_planner_context_v0_9 import (":
        (
            "from app.planning.analytical_v1."
            "context import ("
        ),

    "from app.evals.analytical_planner_input_v0_9 import (":
        (
            "from app.planning.analytical_v1."
            "input import ("
        ),

    "from app.evals.analytical_planner_contract_v0_9 import (":
        (
            "from app.planning.analytical_v1."
            "contract import ("
        ),

    "from app.evals.analytical_planner_validator_v0_9 import (":
        (
            "from app.planning.analytical_v1."
            "validator_base import ("
        ),

    "from app.evals.analytical_planner_validator_v0_9_1 import (":
        (
            "from app.planning.analytical_v1."
            "validator import ("
        ),

    "from app.evals.analytical_reference_canonicalizer_v1_0 import (":
        (
            "from app.planning.analytical_v1."
            "reference_canonicalizer import ("
        ),

    "from app.evals.analytical_planner_safety_pipeline_v1_0 import (":
        (
            "from app.planning.analytical_v1."
            "safety import ("
        ),
}


# ============================================================
# PACKAGE INIT
# ============================================================

INIT_CONTENT = '''\
"""
Production analytical planning stack for DataLens.

This package contains production-promoted versions of the
analytical planning components validated through the eval
harness.

Important architectural rule:

    app.planning.analytical_v1

must never import from:

    the evaluation package

Evaluation code may depend on production code in future
versions, but production code must not depend on evaluation
artifacts.
"""

ANALYTICAL_V1_PACKAGE_VERSION = "analytical_v1"
'''


# ============================================================
# PRODUCTION DATASET CONTEXT
#
# This is the only small schema promoted manually instead of
# copying app/evals/schemas.py, because the latter also
# contains benchmark-only models.
# ============================================================

DATASET_CONTEXT_CONTENT = '''\
from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_DATASET_CONTEXT_VERSION = (
    "analytical_dataset_context_v1.0"
)


# ============================================================
# DATASET COLUMN
# ============================================================

class DatasetColumnSpec(
    BaseModel
):
    """
    Minimal production representation of a dataset column
    exposed to structural analytical planning.

    analytical_type remains a string because the upstream
    profiling layer owns analytical type inference and the
    planning stack consumes the resulting trusted value.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: str = Field(
        min_length=1,
    )

    analytical_type: str = Field(
        min_length=1,
    )

    semantic_role: (
        str
        | None
    ) = None


# ============================================================
# DATASET CONTEXT
# ============================================================

class DatasetContext(
    BaseModel
):
    """
    Trusted structural description of one dataset.

    No raw dataset values are carried here.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    dataset_id: str = Field(
        min_length=1,
    )

    filename: str = Field(
        min_length=1,
    )

    grain: str = Field(
        min_length=1,
    )

    entity_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    columns: list[
        DatasetColumnSpec
    ] = Field(
        min_length=1,
    )


    @model_validator(
        mode="after",
    )
    def validate_entity_columns(
        self,
    ) -> "DatasetContext":

        column_names = {
            column.name

            for column
            in self.columns
        }


        unknown_entity_columns = [
            entity_column

            for entity_column
            in self.entity_columns

            if (
                entity_column
                not in column_names
            )
        ]


        if unknown_entity_columns:

            raise ValueError(
                "Dataset entity_columns must reference "
                "existing dataset columns. "
                f"dataset_id={self.dataset_id}, "
                "unknown="
                f"{unknown_entity_columns}"
            )


        return self
'''


# ============================================================
# SAFETY HELPERS
# ============================================================

def require_source_files() -> None:

    missing = [
        source_name

        for source_name
        in SOURCE_FILE_MAP

        if not (
            SOURCE_DIR
            / source_name
        ).exists()
    ]


    if missing:

        raise FileNotFoundError(
            "Cannot promote analytical_v1 because "
            "validated source files are missing: "
            f"{missing}"
        )


def require_unused_target() -> None:
    """
    First promotion is intentionally append-only.

    We refuse to overwrite an existing production package.
    """

    if TARGET_DIR.exists():

        existing_files = list(
            TARGET_DIR.glob(
                "*.py"
            )
        )


        if existing_files:

            raise FileExistsError(
                "Production analytical_v1 package already "
                "contains Python files. Refusing to overwrite "
                "an existing promoted implementation.\n"
                f"Target: {TARGET_DIR}"
            )


# ============================================================
# REWRITE
# ============================================================

def rewrite_production_imports(
    content: str,
) -> str:

    rewritten = (
        content
    )


    for (
        source_import,
        production_import,
    ) in IMPORT_REWRITES.items():

        rewritten = (
            rewritten.replace(
                source_import,
                production_import,
            )
        )


    return (
        rewritten
    )


# ============================================================
# COPY ONE VALIDATED MODULE
# ============================================================

def promote_module(
    *,
    source_name: str,
    target_name: str,
) -> None:

    source_path = (
        SOURCE_DIR
        / source_name
    )


    target_path = (
        TARGET_DIR
        / target_name
    )


    content = (
        source_path.read_text(
            encoding="utf-8",
        )
    )


    rewritten = (
        rewrite_production_imports(
            content
        )
    )


    # ========================================================
    # HARD ARCHITECTURAL BOUNDARY
    # ========================================================

    if (
        "app.evals"
        in rewritten
    ):

        lines = [
            (
                index,
                line,
            )

            for (
                index,
                line,
            ) in enumerate(
                rewritten.splitlines(),
                start=1,
            )

            if (
                "app.evals"
                in line
            )
        ]


        raise ValueError(
            "Promotion left an app.evals dependency in "
            f"{source_name}: {lines}"
        )


    target_path.write_text(
        rewritten,
        encoding="utf-8",
    )


    print(
        source_name,
        "->",
        target_name,
    )


# ============================================================
# VERIFY PACKAGE
# ============================================================

def verify_promoted_package() -> None:

    expected_files = {
        "__init__.py",
        "dataset_context.py",
        *SOURCE_FILE_MAP.values(),
    }


    actual_files = {
        path.name

        for path
        in TARGET_DIR.glob(
            "*.py"
        )
    }


    if (
        actual_files
        != expected_files
    ):

        raise ValueError(
            "Unexpected promoted package contents. "
            f"Expected={sorted(expected_files)}, "
            f"actual={sorted(actual_files)}"
        )


    violations: list[
        tuple[
            str,
            int,
            str,
        ]
    ] = []


    for path in (
        TARGET_DIR.glob(
            "*.py"
        )
    ):

        for (
            line_number,
            line,
        ) in enumerate(
            path
            .read_text(
                encoding="utf-8",
            )
            .splitlines(),
            start=1,
        ):

            if (
                "app.evals"
                in line
            ):

                violations.append(
                    (
                        path.name,
                        line_number,
                        line,
                    )
                )


    if violations:

        raise ValueError(
            "Production analytical_v1 contains "
            "app.evals references: "
            f"{violations}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL v1 PRODUCTION PROMOTION ==="
    )


    print(
        "Promotion:",
        PROMOTION_VERSION,
    )


    print()


    require_source_files()

    require_unused_target()


    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # CLEAN PRODUCTION-ONLY SUPPORT FILES
    # ========================================================

    (
        TARGET_DIR
        / "__init__.py"
    ).write_text(
        INIT_CONTENT,
        encoding="utf-8",
    )


    (
        TARGET_DIR
        / "dataset_context.py"
    ).write_text(
        DATASET_CONTEXT_CONTENT,
        encoding="utf-8",
    )


    print(
        "[production]",
        "__init__.py",
    )


    print(
        "[production]",
        "dataset_context.py",
    )


    # ========================================================
    # PROMOTE TESTED IMPLEMENTATIONS
    # ========================================================

    for (
        source_name,
        target_name,
    ) in SOURCE_FILE_MAP.items():

        promote_module(
            source_name=(
                source_name
            ),

            target_name=(
                target_name
            ),
        )


    # ========================================================
    # FINAL STATIC VERIFICATION
    # ========================================================

    verify_promoted_package()


    print()


    print(
        "Production files:",
        len(
            list(
                TARGET_DIR.glob(
                    "*.py"
                )
            )
        ),
    )


    print(
        "app.evals production dependencies: 0"
    )


    print()


    print(
        "Analytical v1 production promotion: COMPLETE"
    )


if __name__ == "__main__":
    main()