from __future__ import annotations


from pathlib import Path


from starlette.datastructures import (
    UploadFile,
)


from app.analysis.analytical_views import (
    build_analytical_views,
)

from app.analysis.entity_outliers import (
    detect_entity_outliers,
)

from app.analysis.entity_outlier_profiles import (
    ENTITY_OUTLIER_PROFILE_RULE_VERSION,
    build_entity_outlier_profiles,
)

from app.api.routes import (
    load_uploaded_dataset_bundle,
)


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "entity_outlier_profiles_lapage_v0.1"
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


EXPECTED_PRIORITY_CLIENTS = {
    "c_1609",
    "c_4958",
    "c_3454",
    "c_6714",
}


# ============================================================
# UPLOADS
# ============================================================


def build_uploads(
) -> tuple[
    list[
        UploadFile
    ],
    list,
]:
    handles = []

    uploads: list[
        UploadFile
    ] = []


    for path in (
        DATASET_PATHS
    ):
        if not path.exists():
            raise FileNotFoundError(
                (
                    "Missing Lapage dataset: "
                    f"{path}"
                )
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
# CUSTOMER PROFILE RESULT
# ============================================================


def find_customer_profile_result(
    report,
):
    candidates = [
        result

        for result
        in report.results

        if (
            result.entity_column
            ==
            "client_id"
        )

        and

        (
            "customers"
            in
            result.dataset_filename
            .casefold()
        )
    ]


    if not candidates:
        raise AssertionError(
            (
                "No customer-grain profile result "
                "was produced."
            )
        )


    candidates.sort(
        key=lambda result:
            result.entity_count,
        reverse=True,
    )


    return (
        candidates[
            0
        ]
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
        "DataLens Entity Outlier Profiles · Lapage v0.1"
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


        print()
        print(
            "=== SOURCE DATASETS ==="
        )


        for record in (
            source_records
        ):
            dataframe = (
                record[
                    "dataframe"
                ]
            )


            print(
                (
                    f"{record['filename']:<20} "
                    f"{len(dataframe):>8} rows"
                )
            )


        # ====================================================
        # ANALYTICAL VIEWS
        # ====================================================

        view_build = (
            build_analytical_views(
                source_records
            )
        )


        print()
        print(
            "=== ANALYTICAL VIEWS ==="
        )


        print(
            (
                "Derived datasets : "
                f"{len(view_build.derived_datasets)}"
            )
        )


        # ====================================================
        # RAW ENTITY OUTLIERS
        #
        # We deliberately request enough entities to classify
        # every flagged customer.
        # ====================================================

        raw_report = (
            detect_entity_outliers(
                datasets=
                    view_build
                    .all_datasets,

                top_limit=
                    10_000,
            )
        )


        print()
        print(
            "=== RAW ENTITY OUTLIERS ==="
        )


        print(
            (
                "Views evaluated : "
                f"{raw_report.evaluated_view_count}"
            )
        )

        print(
            (
                "Signals flagged : "
                f"{raw_report.total_flagged_entity_count}"
            )
        )


        # ====================================================
        # PROFILE CLASSIFICATION
        # ====================================================

        profile_report = (
            build_entity_outlier_profiles(
                raw_report,

                top_limit=
                    50,
            )
        )


        assert (
            profile_report.rule_version
            ==
            ENTITY_OUTLIER_PROFILE_RULE_VERSION
        )


        customer_result = (
            find_customer_profile_result(
                profile_report
            )
        )


        print()
        print(
            "=== CUSTOMER PROFILE CLASSIFICATION ==="
        )


        print(
            (
                "Dataset            : "
                f"{customer_result.dataset_filename}"
            )
        )

        print(
            (
                "Entity grain       : "
                f"{customer_result.entity_column}"
            )
        )

        print(
            (
                "Customers          : "
                f"{customer_result.entity_count}"
            )
        )

        print(
            (
                "Raw IQR signals    : "
                f"{customer_result.source_flagged_entity_count}"
            )
        )

        print(
            (
                "Classified         : "
                f"{customer_result.classified_entity_count}"
            )
        )

        print(
            (
                "Priority profiles  : "
                f"{customer_result.priority_profile_count}"
            )
        )

        print(
            (
                "Other signals      : "
                f"{customer_result.behavioral_signal_count}"
            )
        )

        print(
            (
                "Unclassified       : "
                f"{customer_result.unclassified_flagged_entity_count}"
            )
        )


        # ====================================================
        # ALL FLAGGED CUSTOMERS SHOULD BE CLASSIFIED
        # ====================================================

        assert (
            customer_result
            .classified_entity_count
            ==
            customer_result
            .source_flagged_entity_count
        )


        assert (
            customer_result
            .unclassified_flagged_entity_count
            ==
            0
        )


        # ====================================================
        # PRIORITY PROFILES
        # ====================================================

        priority_ids = {
            profile.entity

            for profile
            in customer_result
            .priority_profiles
        }


        print()
        print(
            "=== PRIORITY CLIENT PROFILES ==="
        )


        for (
            rank,
            profile,
        ) in enumerate(
            customer_result
            .priority_profiles,
            start=1,
        ):
            print(
                (
                    f"{rank:02d}. "
                    f"{profile.entity:<12} "
                    f"severity="
                    f"{profile.severity:<8} "
                    f"signals="
                    f"{profile.signal_count} "
                    f"family="
                    f"{profile.dominant_family:<10} "
                    f"max="
                    f"{profile.max_distance_iqr:.2f} IQR"
                )
            )


            print(
                (
                    "    "
                    f"{profile.title}"
                )
            )


            for evidence in (
                profile.evidence
            ):
                print(
                    (
                        "    - "
                        f"{evidence.metric:<28} "
                        f"{evidence.value:>12.2f} "
                        f"{evidence.distance_iqr:>8.2f} IQR "
                        f"[{evidence.family}]"
                    )
                )


        print()
        print(
            "=== EXPECTED PRIORITY CLIENTS ==="
        )


        for client_id in sorted(
            EXPECTED_PRIORITY_CLIENTS
        ):
            found = (
                client_id
                in
                priority_ids
            )


            print(
                (
                    f"{client_id:<10} : "
                    f"{'PRIORITY' if found else 'NOT PRIORITY'}"
                )
            )


        missing = (
            EXPECTED_PRIORITY_CLIENTS
            -
            priority_ids
        )


        assert not missing, (
            "Expected extreme Lapage clients "
            "were not classified as priority: "
            f"{sorted(missing)}"
        )


        # ====================================================
        # IMPORTANT REGRESSION
        #
        # Based on the already validated real-data ranking,
        # the next strongest client is c_2369 with roughly
        # 11 IQR beyond its strongest boundary.
        #
        # It should remain a behavioural signal rather than
        # being promoted to an extreme priority profile.
        # ====================================================

        assert (
            "c_2369"
            not in
            priority_ids
        )


        behavioral_ids = {
            profile.entity

            for profile
            in customer_result
            .behavioral_signals
        }


        assert (
            "c_2369"
            in
            behavioral_ids
        )


        # ====================================================
        # EXPECT EXACTLY FOUR PRIORITY PROFILES
        #
        # This is now safe to validate because the previous
        # real integration run showed the fifth highest score
        # far below the conservative 20-IQR priority boundary.
        # ====================================================

        assert (
            customer_result
            .priority_profile_count
            ==
            4
        ), (
            "Expected exactly four extreme priority "
            "customer profiles on the current Lapage data, "
            "but got "
            f"{customer_result.priority_profile_count}."
        )


        assert (
            priority_ids
            ==
            EXPECTED_PRIORITY_CLIENTS
        )


        # ====================================================
        # PROFILE SEMANTICS
        # ====================================================

        for profile in (
            customer_result
            .priority_profiles
        ):
            assert (
                profile.profile_kind
                ==
                "priority_profile"
            )


            assert (
                profile.severity
                ==
                "extreme"
            )


            assert (
                profile.signal_count
                >=
                2
            )


            assert (
                profile.max_distance_iqr
                >=
                20.0
            )


            assert (
                profile.dominant_family
                ==
                "volume"
            )


            assert (
                "volume"
                in
                profile.signal_families
            )


        # ====================================================
        # ACCOUNTING
        # ====================================================

        assert (
            (
                customer_result
                .priority_profile_count
            )
            +
            (
                customer_result
                .behavioral_signal_count
            )
            ==
            customer_result
            .classified_entity_count
        )


        print()
        print(
            "=== PRODUCT INTERPRETATION ==="
        )


        print(
            (
                f"{customer_result.source_flagged_entity_count} "
                "clients crossed at least one raw IQR boundary."
            )
        )


        print(
            (
                f"{customer_result.priority_profile_count} "
                "clients are classified as extreme "
                "priority profiles."
            )
        )


        print(
            (
                f"{customer_result.behavioral_signal_count} "
                "clients remain behavioural signals "
                "for secondary exploration."
            )
        )


        print()
        print(
            "Automatic fraud label : NO"
        )

        print(
            "Automatic BtoB label  : NO"
        )

        print(
            "Automatic deletion    : NO"
        )


        print()
        print(
            "=========================================================="
        )

        print(
            "PASS - entity outlier profiles Lapage v0.1"
        )

        print(
            "=========================================================="
        )


    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()