from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from app.core.utils import (
    utc_now_iso,
)


# ============================================================
# TEMPORARY IN-MEMORY STORAGE
# ============================================================

WORKSPACES: dict[
    str,
    dict[str, Any],
] = {}


# ============================================================
# CREATE WORKSPACE
# ============================================================

def create_workspace_record(
    name: str,
) -> dict[str, Any]:
    workspace_id = str(
        uuid4()
    )

    workspace = {
        "workspace_id":
            workspace_id,

        "name":
            name,

        "created_at":
            utc_now_iso(),

        "datasets":
            {},
    }

    WORKSPACES[
        workspace_id
    ] = workspace

    return workspace


# ============================================================
# GET WORKSPACE
# ============================================================

def get_workspace_or_404(
    workspace_id: str,
) -> dict[str, Any]:
    workspace = (
        WORKSPACES.get(
            workspace_id
        )
    )

    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Workspace not found."
            ),
        )

    return workspace


# ============================================================
# GET DATASET
# ============================================================

def get_dataset_or_404(
    workspace: dict[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    dataset = (
        workspace[
            "datasets"
        ].get(
            dataset_id
        )
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset not found."
            ),
        )

    return dataset


# ============================================================
# WORKSPACE SUMMARY
# ============================================================

def build_workspace_summary(
    workspace: dict[str, Any],
) -> dict[str, Any]:
    datasets = []

    for (
        dataset_id,
        dataset,
    ) in (
        workspace[
            "datasets"
        ].items()
    ):
        profile_raw = (
            dataset[
                "profile_raw"
            ]
        )

        profile_cleaned = (
            dataset[
                "profile_cleaned"
            ]
        )

        cleaning_report = (
            dataset[
                "cleaning_report"
            ]
        )

        datasets.append(
            {
                "dataset_id":
                    dataset_id,

                "filename":
                    dataset[
                        "filename"
                    ],

                "raw_rows":
                    profile_raw[
                        "rows"
                    ],

                "cleaned_rows":
                    profile_cleaned[
                        "rows"
                    ],

                "columns":
                    profile_cleaned[
                        "columns"
                    ],

                "candidate_keys":
                    profile_cleaned[
                        "candidate_keys"
                    ],

                "cleaning_operations":
                    cleaning_report[
                        "summary"
                    ][
                        "operations_count"
                    ],

                "review_items":
                    cleaning_report[
                        "summary"
                    ][
                        "review_items_count"
                    ],
            }
        )

    return {
        "workspace_id":
            workspace[
                "workspace_id"
            ],

        "name":
            workspace[
                "name"
            ],

        "created_at":
            workspace[
                "created_at"
            ],

        "datasets_count":
            len(
                workspace[
                    "datasets"
                ]
            ),

        "datasets":
            datasets,
    }