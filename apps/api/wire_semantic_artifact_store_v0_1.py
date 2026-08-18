from __future__ import annotations

from pathlib import (
    Path,
)

import shutil


# ============================================================
# PATHS
# ============================================================

TARGET = Path(
    "app/api/preparation_semantic.py"
)

BACKUP = Path(
    "app/api/"
    "preparation_semantic.before_artifact_store_v0_1.py.bak"
)


# ============================================================
# IMPORT
# ============================================================

IMPORT_ANCHOR = """from app.preparation.semantic_confirmation import (
"""

IMPORT_BLOCK = """from app.preparation.semantic_cleaning_artifacts import (
    materialize_semantic_cleaning_artifacts,
)

"""


# ============================================================
# CONFIRM FUNCTION
# ============================================================

CONFIRM_FUNCTION_MARKER = """def confirm_uploaded_semantic_review(
"""


OLD_EXECUTION_ASSIGNMENT = """        (
            _,
            execution,
        ) = (
            execute_semantic_cleaning_plan(
"""


NEW_EXECUTION_ASSIGNMENT = """        (
            semantic_frames,
            execution,
        ) = (
            execute_semantic_cleaning_plan(
"""


RECORD_ANCHOR = """        _record_semantic_confirmation_passed(
"""


MATERIALIZATION_BLOCK = """        # ====================================================
        # MATERIALIZE FINAL CONFIRMED SEMANTIC STATE
        #
        # Trust boundary:
        #
        #   semantic execution
        #       ↓
        #   Python confirmation
        #       ↓
        #   Artifact Store
        #       ↓
        #   CLEAN = PASSED
        #
        # The PreparationSession must never mark CLEAN as
        # completed before the exact confirmed DataFrames
        # exist in the server-owned Artifact Store.
        # ====================================================

        materialize_semantic_cleaning_artifacts(
            workflow_id=
                workflow_id,

            deterministic_frames=
                deterministic_frames,

            derived_frames=
                semantic_frames,

            semantic_plan=
                semantic_plan,

            execution=
                execution,
        )


"""


# ============================================================
# HELPERS
# ============================================================

def require_target() -> None:

    if not TARGET.exists():

        raise FileNotFoundError(
            f"Target file not found: {TARGET}"
        )


def insert_import(
    text: str,
) -> str:

    if IMPORT_BLOCK in text:

        print(
            "Semantic artifact import: already present"
        )

        return text


    count = (
        text.count(
            IMPORT_ANCHOR
        )
    )


    if count != 1:

        raise RuntimeError(
            "Semantic artifact import: expected exactly "
            f"one anchor, found {count}."
        )


    print(
        "Semantic artifact import: inserted"
    )


    return (
        text.replace(
            IMPORT_ANCHOR,
            (
                IMPORT_BLOCK
                +
                IMPORT_ANCHOR
            ),
            1,
        )
    )


def modify_confirmation_function(
    text: str,
) -> str:

    function_start = (
        text.find(
            CONFIRM_FUNCTION_MARKER
        )
    )


    if function_start < 0:

        raise RuntimeError(
            "Semantic confirmation function was not found."
        )


    prefix = (
        text[
            :function_start
        ]
    )


    function_text = (
        text[
            function_start:
        ]
    )


    # ========================================================
    # CAPTURE DERIVED SEMANTIC FRAMES
    # ========================================================

    if (
        NEW_EXECUTION_ASSIGNMENT
        in function_text
    ):

        print(
            "Semantic confirmation frame capture: "
            "already present"
        )


    else:

        count = (
            function_text.count(
                OLD_EXECUTION_ASSIGNMENT
            )
        )


        if count != 1:

            raise RuntimeError(
                "Semantic confirmation frame capture: "
                "expected exactly one execution assignment "
                f"inside confirmation function, found {count}."
            )


        function_text = (
            function_text.replace(
                OLD_EXECUTION_ASSIGNMENT,
                NEW_EXECUTION_ASSIGNMENT,
                1,
            )
        )


        print(
            "Semantic confirmation frame capture: inserted"
        )


    # ========================================================
    # MATERIALIZE BEFORE CLEAN PASSED
    # ========================================================

    if (
        MATERIALIZATION_BLOCK
        in function_text
    ):

        print(
            "Semantic confirmation materialization: "
            "already present"
        )


    else:

        count = (
            function_text.count(
                RECORD_ANCHOR
            )
        )


        if count != 1:

            raise RuntimeError(
                "Semantic confirmation materialization: "
                "expected exactly one PASSED recording "
                f"anchor, found {count}."
            )


        function_text = (
            function_text.replace(
                RECORD_ANCHOR,
                (
                    MATERIALIZATION_BLOCK
                    +
                    RECORD_ANCHOR
                ),
                1,
            )
        )


        print(
            "Semantic confirmation materialization: inserted"
        )


    return (
        prefix
        +
        function_text
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== WIRE SEMANTIC CLEANING → ARTIFACT STORE v0.1 ==="
    )

    print()


    require_target()


    original = (
        TARGET.read_text(
            encoding="utf-8"
        )
    )


    if not BACKUP.exists():

        shutil.copy2(
            TARGET,
            BACKUP,
        )


        print(
            "Backup created:",
            BACKUP,
        )


    else:

        print(
            "Backup already exists:",
            BACKUP,
        )


    updated = (
        insert_import(
            original
        )
    )


    updated = (
        modify_confirmation_function(
            updated
        )
    )


    required_tokens = [
        "materialize_semantic_cleaning_artifacts",
        "semantic_frames",
        "derived_frames=",
        "_record_semantic_confirmation_passed",
    ]


    for token in required_tokens:

        if token not in updated:

            raise RuntimeError(
                "Generated Semantic API is missing "
                f"required token: {token}"
            )


    TARGET.write_text(
        updated,
        encoding="utf-8",
    )


    print()

    print(
        "Target:",
        TARGET,
    )

    print(
        "Semantic Cleaning → Artifact Store wiring: COMPLETE"
    )


if __name__ == "__main__":
    main()