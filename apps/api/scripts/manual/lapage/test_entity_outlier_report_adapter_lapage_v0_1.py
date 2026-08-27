from __future__ import annotations


from pathlib import Path


from starlette.datastructures import (
    UploadFile,
)


from app.analysis.entity_outlier_requests import (
    run_entity_outlier_request,
)

from app.api.routes import (
    load_uploaded_dataset_bundle,
)

from app.reporting.entity_outlier_adapter import (
    ENTITY_OUTLIER_ADAPTER_RULE_VERSION,
    adapt_entity_outlier_request_to_finding,
)


# ============================================================
# REAL LAPAGE DATA
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\tomas\Documents"
    r"\openclassrooms_data_analyst"
    r"\Projet_9"
    r"\Documents"
)


DATA_ROOT = (
    PROJECT_ROOT
    /
    "Data"
)


DATASET_PATHS = [
    DATA_ROOT
    /
    "customers.csv",

    DATA_ROOT
    /
    "products.csv",

    DATA_ROOT
    /
    "Transactions.csv",
]


EXPECTED_CLIENTS = {
    "c_1609",
    "c_4958",
    "c_3454",
    "c_6714",
}


# ============================================================
# UPLOAD HELPERS
# ============================================================


def build_uploads(
) -> tuple[
    list[
        UploadFile
    ],
    list,
]:
    uploads = []

    handles = []


    for path in (
        DATASET_PATHS
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}"
            )


        handle = path.open(
            "rb"
        )


        handles.append(
            handle
        )


        uploads.append(
            UploadFile(
                file=
                    handle,

                filename=
                    path.name,
            )
        )


    return (
        uploads,
        handles,
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print()
    print(
        "=========================================================="
    )

    print(
        "DataLens Entity Outlier Report Adapter · Lapage v0.1"
    )

    print(
        "=========================================================="
    )


    uploads, handles = (
        build_uploads()
    )


    try:
        # ====================================================
        # INGESTION
        # ====================================================

        (
            _,
            source_records,
        ) = load_uploaded_dataset_bundle(
            uploads
        )


        # ====================================================
        # INTERNAL ENTITY ANALYSIS
        # ====================================================

        internal_report = (
            run_entity_outlier_request(
                objective=
                    (
                        "Détecte les clients "
                        "atypiques."
                    ),

                source_dataset_records=
                    source_records,

                top_profile_limit=
                    50,
            )
        )


        assert (
            internal_report.status
            ==
            "ready"
        )


        # ====================================================
        # REPORT ADAPTER
        # ====================================================

        finding = (
            adapt_entity_outlier_request_to_finding(
                internal_report
            )
        )


        print()
        print(
            "=== FINDING ==="
        )


        print(
            (
                "Analysis ID : "
                f"{finding.analysis_id}"
            )
        )

        print(
            (
                "Status      : "
                f"{finding.status}"
            )
        )

        print(
            (
                "Title       : "
                f"{finding.title}"
            )
        )

        print(
            (
                "Family      : "
                f"{finding.family}"
            )
        )

        print(
            (
                "Kind        : "
                f"{finding.kind}"
            )
        )

        print(
            (
                "Dataset     : "
                f"{finding.dataset_filename}"
            )
        )

        print(
            (
                "Customers   : "
                f"{finding.entity_count}"
            )
        )

        print(
            (
                "IQR flags   : "
                f"{finding.raw_flagged_entity_count}"
            )
        )

        print(
            (
                "Priority    : "
                f"{finding.priority_profile_count}"
            )
        )

        print(
            (
                "Secondary   : "
                f"{finding.behavioral_signal_count}"
            )
        )


        # ====================================================
        # CONTRACT
        # ====================================================

        assert (
            finding.status
            ==
            "ready"
        )


        assert (
            finding.family
            ==
            "entity_outlier"
        )


        assert (
            finding.kind
            ==
            (
                "customer_entity_"
                "outlier_detection"
            )
        )


        assert (
            finding.dataset_filename
            ==
            (
                "Transactions"
                "__customers_price.derived"
            )
        )


        assert (
            finding.entity_column
            ==
            "client_id"
        )


        assert (
            finding.entity_count
            ==
            8600
        )


        assert (
            finding.raw_flagged_entity_count
            ==
            1422
        )


        assert (
            finding.priority_profile_count
            ==
            4
        )


        assert (
            finding.behavioral_signal_count
            ==
            1418
        )


        assert (
            finding.adapter_rule_version
            ==
            ENTITY_OUTLIER_ADAPTER_RULE_VERSION
        )


        # ====================================================
        # PRIORITY PROFILES
        # ====================================================

        profile_ids = {
            profile.entity

            for profile
            in finding
            .priority_profiles
        }


        assert (
            profile_ids
            ==
            EXPECTED_CLIENTS
        )


        print()
        print(
            "=== USER-FACING PRIORITY PROFILES ==="
        )


        for (
            rank,
            profile,
        ) in enumerate(
            finding
            .priority_profiles,
            start=1,
        ):
            print(
                (
                    f"{rank:02d}. "
                    f"{profile.entity:<10} "
                    f"{profile.dominant_family_label} "
                    f"· {profile.signal_count} signaux"
                )
            )


            for evidence in (
                profile.evidence
            ):
                print(
                    (
                        "    - "
                        f"{evidence.metric_label:<28} "
                        f"{evidence.value:>12.2f} "
                        f"{evidence.direction}"
                    )
                )


        # ====================================================
        # INTERNAL SCORE MUST NOT LEAK
        # ====================================================

        serialized = (
            finding.model_dump(
                mode="json"
            )
        )


        assert (
            "anomaly_score"
            not in
            str(
                serialized
            )
        )


        print()
        print(
            "Internal anomaly score exposed : NO"
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print(
            "=== SUMMARY ==="
        )


        for sentence in (
            finding.summary
        ):
            print(
                (
                    "- "
                    f"{sentence}"
                )
            )


        assert (
            finding.summary
        )


        assert any(
            "4 clients"
            in sentence

            for sentence
            in finding.summary
        )


        # ====================================================
        # SAFETY LANGUAGE
        # ====================================================

        caveat_text = (
            " ".join(
                finding.caveats
            )
            .casefold()
        )


        assert (
            "fraude"
            in
            caveat_text
        )


        assert (
            "b2b"
            in
            caveat_text
        )


        assert (
            "supprim"
            in
            caveat_text
        )


        print()
        print(
            "=== SAFETY ==="
        )

        print(
            "Fraud inference       : NO"
        )

        print(
            "BtoB inference        : NO"
        )

        print(
            "Automatic deletion    : NO"
        )

        print(
            "Raw internal score UI : NO"
        )


        # ====================================================
        # PASS
        # ====================================================

        print()
        print(
            "=========================================================="
        )

        print(
            "PASS - entity outlier report adapter Lapage v0.1"
        )

        print(
            "=========================================================="
        )


    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()