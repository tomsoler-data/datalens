from __future__ import annotations

from pathlib import Path
import shutil


# ============================================================
# PATHS
# ============================================================

TARGET = Path(
    "app/api/preparation_cleaning.py"
)

BACKUP = Path(
    "app/api/"
    "preparation_cleaning.before_artifact_store_v0_1.py.bak"
)


# ============================================================
# BLOCKS
# ============================================================

IMPORT_ANCHOR = """from app.preparation.data_quality import (
"""

IMPORT_BLOCK = """from app.preparation.cleaning_artifacts import (
    materialize_cleaning_execution_artifacts,
    materialize_skipped_cleaning_artifacts,
)

"""


PLAN_ANCHOR = """        _record_cleaning_plan_stage(
"""

PLAN_BLOCK = """        # ====================================================
        # MATERIALIZE SOURCE WHEN DETERMINISTIC CLEANING
        # HAS NOTHING TO EXECUTE
        #
        # This may still happen while protected semantic
        # issues remain. Artifact existence does NOT mean
        # CLEAN is authorized for downstream use.
        # PreparationSession remains the readiness authority.
        # ====================================================

        if (
            cleaning_plan.action_count
            ==
            0
        ):
            materialize_skipped_cleaning_artifacts(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,

                cleaning_plan=
                    cleaning_plan,
            )


"""


APPLY_ANCHOR = """        _record_cleaning_execution_stage(
"""

APPLY_BLOCK = """        # ====================================================
        # MATERIALIZE THE EXACT CLEANING RESULT BEFORE
        # CHANGING THE LOGICAL WORKFLOW STATE
        #
        # Important trust boundary:
        #
        #     material artifact
        #         BEFORE
        #     PreparationSession stage update
        #
        # Therefore CLEAN can never become completed while its
        # corresponding server-owned DataFrame was lost.
        #
        # A protected semantic issue may still keep CLEAN in
        # REVIEW_REQUIRED. The artifact remains useful as the
        # input to the semantic-cleaning substage.
        # ====================================================

        if (
            cleaning_plan.action_count
            ==
            0
        ):
            materialize_skipped_cleaning_artifacts(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,

                cleaning_plan=
                    cleaning_plan,
            )

        elif (
            execution.blocked_action_count
            ==
            0
        ):
            materialize_cleaning_execution_artifacts(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,

                cleaning_plan=
                    cleaning_plan,

                derived_frames=
                    derived_frames,

                execution=
                    execution,
            )


"""


OLD_NOTE = """                    (
                        "Derived datasets are not yet "
                        "wired into /analysis/run in v0.1."
                    ),
"""

NEW_NOTE = """                    (
                        "Derived cleaning datasets are "
                        "materialized server-side for "
                        "downstream preparation stages."
                    ),
"""


# ============================================================
# HELPERS
# ============================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Target file not found: {path}"
        )


def insert_once(
    *,
    text: str,
    anchor: str,
    block: str,
    label: str,
) -> str:

    if block in text:

        print(
            f"{label}: already present"
        )

        return text


    count = (
        text.count(
            anchor
        )
    )


    if count != 1:

        raise RuntimeError(
            f"{label}: expected exactly one anchor, "
            f"found {count}."
        )


    updated = (
        text.replace(
            anchor,
            block + anchor,
            1,
        )
    )


    print(
        f"{label}: inserted"
    )


    return updated


def replace_note(
    text: str,
) -> str:

    if NEW_NOTE in text:

        print(
            "Cleaning response note: already updated"
        )

        return text


    count = (
        text.count(
            OLD_NOTE
        )
    )


    if count != 1:

        raise RuntimeError(
            "Cleaning response note: expected exactly one "
            f"legacy note, found {count}."
        )


    updated = (
        text.replace(
            OLD_NOTE,
            NEW_NOTE,
            1,
        )
    )


    print(
        "Cleaning response note: updated"
    )


    return updated


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== WIRE CLEANING → PREPARATION ARTIFACT STORE v0.1 ==="
    )

    print()


    require_file(
        TARGET
    )


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
        original
    )


    # ========================================================
    # IMPORT
    # ========================================================

    updated = (
        insert_once(
            text=(
                updated
            ),

            anchor=(
                IMPORT_ANCHOR
            ),

            block=(
                IMPORT_BLOCK
            ),

            label=(
                "Cleaning artifact imports"
            ),
        )
    )


    # ========================================================
    # CLEANING PLAN
    # ========================================================

    updated = (
        insert_once(
            text=(
                updated
            ),

            anchor=(
                PLAN_ANCHOR
            ),

            block=(
                PLAN_BLOCK
            ),

            label=(
                "Cleaning-plan materialization"
            ),
        )
    )


    # ========================================================
    # CLEANING APPLY
    # ========================================================

    updated = (
        insert_once(
            text=(
                updated
            ),

            anchor=(
                APPLY_ANCHOR
            ),

            block=(
                APPLY_BLOCK
            ),

            label=(
                "Cleaning-apply materialization"
            ),
        )
    )


    # ========================================================
    # RESPONSE NOTE
    # ========================================================

    updated = (
        replace_note(
            updated
        )
    )


    # ========================================================
    # FINAL SANITY
    # ========================================================

    required_tokens = [
        (
            "materialize_skipped_cleaning_artifacts"
        ),

        (
            "materialize_cleaning_execution_artifacts"
        ),

        (
            "Derived cleaning datasets are "
        ),
    ]


    for token in required_tokens:

        if token not in updated:

            raise RuntimeError(
                "Generated Cleaning API is missing "
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
        "Cleaning → Artifact Store wiring: COMPLETE"
    )


if __name__ == "__main__":
    main()