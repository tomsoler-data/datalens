from __future__ import annotations


from pathlib import Path


from starlette.datastructures import (
    UploadFile,
)


from app.analysis.analytical_views import (
    ANALYTICAL_VIEW_RULE_VERSION,
    build_analytical_views,
)

from app.analysis.entity_outliers import (
    ENTITY_OUTLIER_RULE_VERSION,
    detect_entity_outliers,
)

from app.api.routes import (
    load_uploaded_dataset_bundle,
)


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "entity_outliers_lapage_integration_v0.1"
)


# ============================================================
# REAL LAPAGE PATHS
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


CUSTOMERS_PATH = (
    DATA_ROOT
    /
    "customers.csv"
)


PRODUCTS_PATH = (
    DATA_ROOT
    /
    "products.csv"
)


TRANSACTIONS_PATH = (
    DATA_ROOT
    /
    "Transactions.csv"
)


# ============================================================
# EXPECTED BUSINESS SIGNALS
#
# We do NOT assert the exact numerical anomaly score.
#
# The engine is free to evolve its scoring system.
#
# What matters here is that the known extreme customer
# profiles remain detectable from the real analytical view.
# ============================================================


EXPECTED_ATYPICAL_CLIENTS = {
    "c_1609",
    "c_4958",
    "c_6714",
    "c_3454",
}


# ============================================================
# FILE VALIDATION
# ============================================================


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            (
                "Required Lapage dataset "
                "was not found: "
                f"{path}"
            )
        )


    if not path.is_file():
        raise FileNotFoundError(
            (
                "Expected a file but found "
                "something else: "
                f"{path}"
            )
        )


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
    """
    Build Starlette UploadFile objects from the real local
    Lapage files.

    The second returned list contains the underlying file
    handles so the test can always close them explicitly.
    """

    paths = [
        CUSTOMERS_PATH,
        PRODUCTS_PATH,
        TRANSACTIONS_PATH,
    ]


    for path in paths:
        require_file(
            path
        )


    handles = []


    uploads: list[
        UploadFile
    ] = []


    for path in paths:
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
# CUSTOMER VIEW HELPER
# ============================================================


def find_customer_view(
    datasets,
):
    candidates = []


    for dataset in datasets:
        provenance = (
            dataset.get(
                "provenance"
            )
            or {}
        )


        operation = str(
            provenance.get(
                "operation"
            )
            or ""
        ).strip()


        derivation_type = str(
            dataset.get(
                "derivation_type"
            )
            or ""
        ).strip()


        dataframe = (
            dataset.get(
                "dataframe"
            )
        )


        if dataframe is None:
            continue


        columns = {
            str(
                column
            )

            for column
            in dataframe.columns
        }


        looks_like_customer_view = (
            operation
            ==
            "customer_behavior_materialization"
        )


        has_customer_behaviour_shape = (
            "client_id"
            in columns

            and

            (
                "total_spend"
                in columns

                or

                "purchase_sessions"
                in columns

                or

                "average_basket"
                in columns
            )
        )


        if (
            looks_like_customer_view
            or
            (
                derivation_type
                ==
                "entity_additive_measure"

                and
                has_customer_behaviour_shape
            )
        ):
            candidates.append(
                dataset
            )


    if not candidates:
        raise AssertionError(
            (
                "The Analytical View Builder "
                "did not produce a customer-grain "
                "behavioural view."
            )
        )


    candidates.sort(
        key=lambda dataset:
            (
                str(
                    (
                        dataset.get(
                            "provenance"
                        )
                        or {}
                    ).get(
                        "operation"
                    )
                    or ""
                )
                !=
                "customer_behavior_materialization",

                -len(
                    dataset[
                        "dataframe"
                    ].columns
                ),
            )
    )


    return (
        candidates[
            0
        ]
    )


# ============================================================
# MAIN TEST
# ============================================================


def main(
) -> None:
    print()
    print(
        "=========================================================="
    )

    print(
        "DataLens Entity Outliers · Lapage Integration v0.1"
    )

    print(
        "=========================================================="
    )


    uploads, handles = (
        build_uploads()
    )


    try:
        # ====================================================
        # 1. REAL DATALENS INGESTION
        # ====================================================

        (
            ingestion,
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
                    f"{record['dataset_id']} | "
                    f"{record['filename']} | "
                    f"{len(dataframe)} rows | "
                    f"{len(dataframe.columns)} columns"
                )
            )


        assert (
            len(
                source_records
            )
            ==
            3
        )


        assert {
            record[
                "filename"
            ]

            for record
            in source_records
        } == {
            "customers.csv",
            "products.csv",
            "Transactions.csv",
        }


        # ====================================================
        # 2. REAL ANALYTICAL VIEW BUILDER
        # ====================================================

        view_build = (
            build_analytical_views(
                source_records
            )
        )


        print()
        print(
            "=== ANALYTICAL VIEW BUILDER ==="
        )


        print(
            (
                "Rule version    : "
                f"{view_build.rule_version}"
            )
        )

        print(
            (
                "Source datasets : "
                f"{len(view_build.original_datasets)}"
            )
        )

        print(
            (
                "Derived datasets: "
                f"{len(view_build.derived_datasets)}"
            )
        )

        print(
            (
                "Join audits     : "
                f"{len(view_build.join_audits)}"
            )
        )


        assert (
            view_build.rule_version
            ==
            ANALYTICAL_VIEW_RULE_VERSION
        )


        assert (
            len(
                view_build.derived_datasets
            )
            >
            0
        )


        customer_view = (
            find_customer_view(
                view_build
                .derived_datasets
            )
        )


        customer_dataframe = (
            customer_view[
                "dataframe"
            ]
        )


        customer_provenance = (
            customer_view.get(
                "provenance"
            )
            or {}
        )


        print()
        print(
            "=== CUSTOMER ANALYTICAL VIEW ==="
        )


        print(
            (
                "Dataset ID : "
                f"{customer_view['dataset_id']}"
            )
        )

        print(
            (
                "Filename   : "
                f"{customer_view['filename']}"
            )
        )

        print(
            (
                "Rows       : "
                f"{len(customer_dataframe)}"
            )
        )

        print(
            (
                "Columns    : "
                f"{list(customer_dataframe.columns)}"
            )
        )

        print(
            (
                "Operation  : "
                f"{customer_provenance.get('operation')}"
            )
        )

        print(
            (
                "Entity     : "
                f"{customer_provenance.get('entity_column')}"
            )
        )


        # Previous DataLens integration tests have produced
        # approximately 8,600 customer-grain observations.
        #
        # Use a range rather than an exact count so the test
        # does not become coupled to a small cleaning change.

        assert (
            8_000
            <=
            len(
                customer_dataframe
            )
            <=
            9_000
        )


        assert (
            "client_id"
            in
            customer_dataframe.columns
        )


        # ====================================================
        # IMPORTANT
        #
        # pandas.Series.any() may return numpy.bool_, so:
        #
        #     series.any() is False
        #
        # is not a safe assertion even when the logical value
        # is actually False.
        #
        # Count duplicate entity IDs explicitly instead.
        # This also gives us a much better diagnostic if the
        # analytical view ever violates the customer grain.
        # ====================================================

        duplicate_client_count = int(
            customer_dataframe[
                "client_id"
            ]
            .dropna()
            .duplicated()
            .sum()
        )


        print(
            (
                "Duplicate client IDs : "
                f"{duplicate_client_count}"
            )
        )


        assert (
            duplicate_client_count
            ==
            0
        ), (
            "The customer analytical view must contain "
            "exactly one row per client, but "
            f"{duplicate_client_count} duplicate client "
            "ID(s) were found."
        )


        assert (
            "total_spend"
            in
            customer_dataframe.columns
        )


        # ====================================================
        # 3. ENTITY OUTLIER ENGINE
        # ====================================================

        entity_report = (
            detect_entity_outliers(
                datasets=
                    view_build
                    .all_datasets,

                top_limit=
                    50,
            )
        )


        print()
        print(
            "=== ENTITY OUTLIER REPORT ==="
        )


        print(
            (
                "Rule version   : "
                f"{entity_report.rule_version}"
            )
        )

        print(
            (
                "Candidate views: "
                f"{entity_report.candidate_view_count}"
            )
        )

        print(
            (
                "Evaluated views: "
                f"{entity_report.evaluated_view_count}"
            )
        )

        print(
            (
                "Total flagged  : "
                f"{entity_report.total_flagged_entity_count}"
            )
        )


        assert (
            entity_report.rule_version
            ==
            ENTITY_OUTLIER_RULE_VERSION
        )


        assert (
            entity_report
            .candidate_view_count
            >=
            1
        )


        assert (
            entity_report
            .evaluated_view_count
            >=
            1
        )


        # ====================================================
        # 4. FIND CUSTOMER OUTLIER RESULT
        # ====================================================

        customer_results = [
            result

            for result
            in entity_report.results

            if (
                result.entity_column
                ==
                "client_id"
            )

            and

            (
                result.operation
                ==
                "customer_behavior_materialization"

                or

                "total_spend"
                in
                result.evaluated_metrics
            )
        ]


        if not customer_results:
            raise AssertionError(
                (
                    "Entity outlier engine did not "
                    "evaluate the customer-grain "
                    "analytical view."
                )
            )


        customer_result = max(
            customer_results,

            key=lambda result:
                len(
                    result.evaluated_metrics
                ),
        )


        print()
        print(
            "=== CUSTOMER OUTLIER RESULT ==="
        )


        print(
            (
                "Dataset       : "
                f"{customer_result.dataset_filename}"
            )
        )

        print(
            (
                "Entity grain  : "
                f"{customer_result.entity_column}"
            )
        )

        print(
            (
                "Entity count  : "
                f"{customer_result.entity_count}"
            )
        )

        print(
            (
                "Primary metric: "
                f"{customer_result.primary_metric}"
            )
        )

        print(
            (
                "Metrics       : "
                f"{customer_result.evaluated_metrics}"
            )
        )

        print(
            (
                "Flagged       : "
                f"{customer_result.flagged_entity_count}"
            )
        )


        assert (
            customer_result
            .primary_metric
            ==
            "total_spend"
        )


        assert (
            "total_spend"
            in
            customer_result
            .evaluated_metrics
        )


        assert (
            customer_result
            .flagged_entity_count
            >
            0
        )


        # ====================================================
        # 5. REAL LAPAGE ATYPICAL CLIENTS
        # ====================================================

        detected_entities = {
            candidate.entity

            for candidate
            in customer_result
            .top_entities
        }


        print()
        print(
            "=== TOP ATYPICAL CLIENTS ==="
        )


        for (
            rank,
            candidate,
        ) in enumerate(
            customer_result
            .top_entities[
                :15
            ],
            start=1,
        ):
            print(
                (
                    f"{rank:02d}. "
                    f"{candidate.entity:<12} "
                    f"score="
                    f"{candidate.anomaly_score:.2f} "
                    f"signals="
                    f"{candidate.outlier_metric_count}"
                )
            )


            for evidence in (
                candidate.evidence[
                    :4
                ]
            ):
                print(
                    (
                        "    - "
                        f"{evidence.metric}: "
                        f"{evidence.value:.2f} "
                        f"[{evidence.direction}] "
                        f"{evidence.distance_iqr:.2f} "
                        "IQR beyond threshold"
                    )
                )


        print()
        print(
            "=== EXPECTED LAPAGE CLIENTS ==="
        )


        for client_id in sorted(
            EXPECTED_ATYPICAL_CLIENTS
        ):
            found = (
                client_id
                in
                detected_entities
            )


            print(
                (
                    f"{client_id:<10} : "
                    f"{'FOUND' if found else 'NOT FOUND'}"
                )
            )


        missing_expected = (
            EXPECTED_ATYPICAL_CLIENTS
            -
            detected_entities
        )


        assert not (
            missing_expected
        ), (
            "Known atypical Lapage clients "
            "were not all found in the top "
            "entity-outlier results: "
            f"{sorted(missing_expected)}"
        )


        # ====================================================
        # 6. SOURCE DATASET SAFETY
        # ====================================================

        assert all(
            not bool(
                record.get(
                    "is_derived",
                    False,
                )
            )

            for record
            in source_records
        )


        print()
        print(
            "=== SAFETY ==="
        )

        print(
            "Raw transaction rows treated as entities : NO"
        )

        print(
            "Entity grain derived by DataLens          : YES"
        )

        print(
            "Automatic customer deletion               : NO"
        )

        print(
            "Automatic BtoB classification             : NO"
        )


        # ====================================================
        # PASS
        # ====================================================

        print()
        print(
            "=========================================================="
        )

        print(
            "PASS - entity outliers Lapage integration v0.1"
        )

        print(
            "=========================================================="
        )


    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()