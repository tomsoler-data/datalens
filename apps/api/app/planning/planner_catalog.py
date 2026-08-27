from __future__ import annotations


from typing import (
    Any,
)


import pandas as pd


from app.planning.ai_analytical_planner import (
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
)

from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# VERSION
# ============================================================


PLANNER_CATALOG_RULE_VERSION = (
    "planner_catalog_v0.3"
)


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if (
        denominator
        <=
        0
    ):
        return 0.0


    return float(
        numerator
        /
        denominator
    )


def _normalized_record_text(
    record: dict[
        str,
        Any,
    ],
    key: str,
) -> str:
    return str(
        record.get(
            key
        )
        or
        ""
    ).strip()


def _duplicate_column_names(
    dataframe: pd.DataFrame,
) -> list[
    str
]:
    names = [
        str(
            column
        )

        for column
        in dataframe.columns
    ]


    return sorted(
        {
            name

            for name
            in names

            if (
                names.count(
                    name
                )
                >
                1
            )
        }
    )


def _provenance_dict(
    record: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    provenance = (
        record.get(
            "provenance"
        )
    )


    return (
        provenance
        if isinstance(
            provenance,
            dict,
        )
        else {}
    )


def _normalized_optional_text(
    value: Any,
) -> str | None:
    normalized = str(
        value
        or
        ""
    ).strip()


    return (
        normalized
        or
        None
    )


def _analytical_measure_aliases(
    *,
    provenance: dict[
        str,
        Any,
    ],
) -> list[
    str
]:
    """
    Build a small deterministic semantic alias set for the declared
    target measure of a server-owned analytical view.

    The special revenue aliases are allowed only for the strict
    analytical line-amount derivation already guarded by the
    Analytical View Builder:

        quantity * unit_price -> gross_amount

    This does not invent a new physical column and does not mutate the
    Preparation output. It exposes controlled semantic vocabulary to
    the planner and to Python's metric-fidelity checks.
    """

    aliases: list[
        str
    ] = []


    for key in [
        "source_measure_column",
        "target_measure_column",
    ]:
        value = _normalized_optional_text(
            provenance.get(
                key
            )
        )


        if value:
            aliases.append(
                value
            )


    derivation = (
        provenance.get(
            "source_measure_derivation"
        )
    )


    if not isinstance(
        derivation,
        dict,
    ):
        return list(
            dict.fromkeys(
                aliases
            )
        )


    operation = _normalized_optional_text(
        derivation.get(
            "operation"
        )
    )


    derived_column = _normalized_optional_text(
        derivation.get(
            "derived_column"
        )
    )


    quantity_column = _normalized_optional_text(
        derivation.get(
            "source_quantity_column"
        )
    )


    unit_price_column = _normalized_optional_text(
        derivation.get(
            "source_unit_price_column"
        )
    )


    if (
        operation
        ==
        "analytical_line_amount_derivation"
        and
        derived_column
        ==
        "gross_amount"
        and
        quantity_column
        and
        unit_price_column
    ):
        aliases.extend(
            [
                "gross_amount",
                "gross_sales_amount",
                "sales_amount",
                "revenue",
                "turnover",
                "chiffre_affaires",
                "ca",
            ]
        )


    return list(
        dict.fromkeys(
            alias
            for alias
            in aliases
            if alias
        )
    )


# ============================================================
# CENTRAL ANALYTICAL CATALOG
# ============================================================


def planner_catalog_from_dataset_records(
    dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> PlannerCatalog:
    """
    Build the AI planner catalog directly from the DataFrames
    that will actually be analyzed.

    Analytical typing is delegated to the central DataLens
    profiler:

        app.profiling.types.infer_analytical_type

    This prevents the AI planner from maintaining a second,
    weaker interpretation of analytical variable types.

    Important architecture rule:

        physical dtype
            !=
        analytical type

    Examples:

        int64 birth year
            -> temporal

        int64 category code
            -> categorical

        float64 business measure
            -> quantitative

        string customer identifier
            -> identifier

    Only schema-level and server-owned analytical metadata are returned
    in PlannerCatalog. Raw rows are never copied into the planner
    catalog.
    """

    if not dataset_records:
        raise ValueError(
            (
                "At least one dataset record "
                "is required to build the "
                "planner catalog."
            )
        )


    datasets: list[
        PlannerDatasetProfile
    ] = []


    seen_dataset_ids: set[
        str
    ] = set()


    for record in (
        dataset_records
    ):
        # ====================================================
        # DATASET IDENTITY
        # ====================================================

        dataset_id = (
            _normalized_record_text(
                record,
                "dataset_id",
            )
        )


        filename = (
            _normalized_record_text(
                record,
                "filename",
            )
        )


        if not dataset_id:
            raise ValueError(
                (
                    "A dataset record is missing "
                    "its dataset_id."
                )
            )


        if (
            dataset_id
            in
            seen_dataset_ids
        ):
            raise ValueError(
                (
                    "Duplicate dataset_id in "
                    "planner catalog input: "
                    f"{dataset_id}."
                )
            )


        if not filename:
            raise ValueError(
                (
                    f"Dataset {dataset_id} "
                    "is missing its filename."
                )
            )


        dataframe = (
            record.get(
                "dataframe"
            )
        )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                (
                    f"Dataset {dataset_id} "
                    "does not contain a pandas "
                    "DataFrame."
                )
            )


        duplicate_columns = (
            _duplicate_column_names(
                dataframe
            )
        )


        if duplicate_columns:
            raise ValueError(
                (
                    f"Dataset {dataset_id} contains "
                    "duplicate column names that "
                    "cannot be represented safely "
                    "in the analytical planner "
                    "catalog: "
                    +
                    ", ".join(
                        duplicate_columns
                    )
                    +
                    "."
                )
            )


        seen_dataset_ids.add(
            dataset_id
        )


        # ====================================================
        # DATASET METRICS
        # ====================================================

        row_count = int(
            dataframe.shape[
                0
            ]
        )


        column_count = int(
            dataframe.shape[
                1
            ]
        )


        columns: list[
            PlannerColumnProfile
        ] = []


        # ====================================================
        # CENTRAL COLUMN TYPING
        # ====================================================

        for (
            column_index,
            raw_column_name,
        ) in enumerate(
            dataframe.columns
        ):
            column_name = str(
                raw_column_name
            )


            series = (
                dataframe.iloc[
                    :,
                    column_index,
                ]
            )


            analytical_type = (
                infer_analytical_type(
                    column_name,
                    series,
                )
            )


            analysis_kind = str(
                analytical_type.get(
                    "type"
                )
                or
                "unknown"
            ).strip()


            if not analysis_kind:
                analysis_kind = (
                    "unknown"
                )


            missing_count = int(
                series
                .isna()
                .sum()
            )


            unique_count = int(
                series
                .nunique(
                    dropna=True
                )
            )


            non_missing_count = (
                row_count
                -
                missing_count
            )


            missing_ratio = (
                _safe_ratio(
                    missing_count,
                    row_count,
                )
            )


            # ------------------------------------------------
            # IMPORTANT
            #
            # A quantitative variable can legitimately contain
            # one distinct value per observation.
            #
            # Example:
            #   transaction_amount
            #   sensor_measurement
            #   price
            #
            # Therefore:
            #
            #   unique_count == row_count
            #
            # is NOT sufficient to classify a variable as an
            # analytical identifier.
            #
            # `unique_candidate` remains a conservative helper
            # for identifier protection, but only when the
            # central profiler has already established
            # identifier semantics.
            # ------------------------------------------------

            unique_candidate = bool(
                analysis_kind
                ==
                "identifier"

                and
                non_missing_count
                >
                0

                and
                missing_count
                ==
                0

                and
                unique_count
                ==
                row_count
            )


            columns.append(
                PlannerColumnProfile(
                    name=
                        column_name,

                    dtype=str(
                        series.dtype
                    ),

                    analysis_kind=
                        analysis_kind,

                    missing_ratio=
                        missing_ratio,

                    unique_count=
                        unique_count,

                    unique_candidate=
                        unique_candidate,
                )
            )


        # ====================================================
        # SERVER-OWNED ANALYTICAL METADATA
        # ====================================================

        provenance = (
            _provenance_dict(
                record
            )
        )


        derivation = (
            provenance.get(
                "source_measure_derivation"
            )
        )


        source_measure_formula = (
            _normalized_optional_text(
                derivation.get(
                    "formula"
                )
            )
            if isinstance(
                derivation,
                dict,
            )
            else None
        )


        analytical_grain = (
            _normalized_optional_text(
                provenance.get(
                    "grain"
                )
            )
            or
            _normalized_optional_text(
                record.get(
                    "analytical_grain"
                )
            )
        )


        # ====================================================
        # DATASET PROFILE
        # ====================================================

        datasets.append(
            PlannerDatasetProfile(
                dataset_id=
                    dataset_id,

                filename=
                    filename,

                row_count=
                    row_count,

                column_count=
                    column_count,

                columns=
                    columns,

                is_derived=bool(
                    record.get(
                        "is_derived",
                        False,
                    )
                ),

                derivation_type=(
                    _normalized_optional_text(
                        record.get(
                            "derivation_type"
                        )
                    )
                ),

                analytical_grain=
                    analytical_grain,

                operation=(
                    _normalized_optional_text(
                        provenance.get(
                            "operation"
                        )
                    )
                ),

                aggregation=(
                    _normalized_optional_text(
                        provenance.get(
                            "aggregation"
                        )
                    )
                ),

                group_column=(
                    _normalized_optional_text(
                        provenance.get(
                            "group_column"
                        )
                    )
                ),

                entity_column=(
                    _normalized_optional_text(
                        provenance.get(
                            "entity_column"
                        )
                    )
                ),

                source_time_column=(
                    _normalized_optional_text(
                        provenance.get(
                            "source_time_column"
                        )
                    )
                ),

                target_time_column=(
                    _normalized_optional_text(
                        provenance.get(
                            "target_time_column"
                        )
                    )
                ),

                source_measure_column=(
                    _normalized_optional_text(
                        provenance.get(
                            "source_measure_column"
                        )
                    )
                ),

                target_measure_column=(
                    _normalized_optional_text(
                        provenance.get(
                            "target_measure_column"
                        )
                    )
                ),

                source_measure_formula=(
                    source_measure_formula
                ),

                metric_semantics=(
                    _normalized_optional_text(
                        provenance.get(
                            "metric_semantics"
                        )
                    )
                ),

                measure_semantic_aliases=(
                    _analytical_measure_aliases(
                        provenance=
                            provenance,
                    )
                ),
            )
        )


    # ========================================================
    # FINAL CATALOG
    # ========================================================

    return (
        PlannerCatalog(
            datasets=
                datasets
        )
    )
