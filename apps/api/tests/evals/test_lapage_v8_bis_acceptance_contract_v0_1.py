from __future__ import annotations

import hashlib
import json
import os
import re

from pathlib import Path

import pandas as pd


FIXTURE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    / "fixtures"
)


INPUT_PATH = (
    FIXTURE_DIR
    / "lapage_v8_bis_input_contract_v0_1.json"
)


GOLD_PATH = (
    FIXTURE_DIR
    / "lapage_v8_bis_gold_v0_1.json"
)


EXPECTED_CASE_IDS = {
    "lapage_total_revenue",
    "lapage_category_revenue_leader",
    "lapage_average_basket",
    "lapage_gender_category_association",
    "lapage_age_total_spend",
    "lapage_age_purchase_sessions",
    "lapage_age_average_basket",
    "lapage_age_band_category",
    "lapage_customer_outliers",
    "lapage_customer_revenue_gini",
    "lapage_unsold_products",
}


KNOWN_GOLD_TOKENS = [
    "158.254",
    "-0.185",
    "0.212",
    "-0.701",
    "0.390",
    "0.442",
    "c_1609",
    "c_3454",
    "c_4958",
    "c_6714",
]


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:

    print(
        "=== DATALENS V8-BIS LAPAGE "
        "ACCEPTANCE CONTRACT v0.1 ==="
    )


    model_input = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )


    evaluator_gold = json.loads(
        GOLD_PATH.read_text(
            encoding="utf-8"
        )
    )


    assert (
        model_input[
            "frozen"
        ]
        is True
    )


    assert (
        evaluator_gold[
            "frozen"
        ]
        is True
    )


    input_ids = {
        case[
            "case_id"
        ]
        for case
        in model_input[
            "cases"
        ]
    }


    gold_ids = {
        case[
            "case_id"
        ]
        for case
        in evaluator_gold[
            "cases"
        ]
    }


    assert (
        input_ids
        ==
        EXPECTED_CASE_IDS
    )


    assert (
        gold_ids
        ==
        EXPECTED_CASE_IDS
    )


    assert (
        input_ids
        ==
        gold_ids
    )


    serialized_input = json.dumps(
        model_input,
        ensure_ascii=False,
        sort_keys=True,
    )


    for token in KNOWN_GOLD_TOKENS:

        assert (
            token
            not in
            serialized_input
        ), (
            "Gold leaked into model-visible contract: "
            f"{token}"
        )


    print(
        "[PASS] model-visible contract is label-blind"
    )


    authority = (
        evaluator_gold[
            "canonicalization_authority"
        ]
    )


    assert (
        authority[
            "transactions_input"
        ]
        ==
        1048575
    )


    assert (
        authority[
            "transactions_all_null_removed"
        ]
        ==
        361041
    )


    assert (
        authority[
            "transactions_canonical"
        ]
        ==
        687534
    )


    assert (
        authority[
            "rule"
        ]
        ==
        "pandas.DataFrame.dropna(how='all')"
    )


    assert (
        authority[
            "delimiter"
        ]
        ==
        ";"
    )


    assert (
        authority[
            "canonical_duplicate_count"
        ]
        ==
        0
    )


    print(
        "[PASS] P9 canonicalization authority frozen"
    )


    fingerprints = (
        evaluator_gold[
            "source_dataset_fingerprints"
        ]
    )


    assert (
        len(
            fingerprints
        )
        ==
        3
    )


    for item in fingerprints:

        assert re.fullmatch(
            r"[0-9a-f]{64}",
            item[
                "sha256_raw_file"
            ],
        )


    print(
        "[PASS] raw source SHA-256 fingerprints frozen"
    )


    configured = (
        os.getenv(
            "DATALENS_LAPAGE_DATA_DIR",
            "",
        )
        .strip()
    )


    if configured:

        data_dir = Path(
            configured
        )


        by_filename = {
            item[
                "filename"
            ]:
                item

            for item
            in fingerprints
        }


        for filename in [
            "customers.csv",
            "products.csv",
            "Transactions.csv",
        ]:

            frozen = (
                by_filename[
                    filename
                ]
            )


            path = (
                data_dir
                / filename
            )


            assert path.is_file()


            assert (
                sha256_file(
                    path
                )
                ==
                frozen[
                    "sha256_raw_file"
                ]
            )


            dataframe = pd.read_csv(
                path,
                sep=frozen[
                    "delimiter"
                ],
                low_memory=False,
            )


            assert (
                len(
                    dataframe
                )
                ==
                frozen[
                    "raw_row_count"
                ]
            )


            assert (
                int(
                    dataframe
                    .isna()
                    .all(
                        axis=1
                    )
                    .sum()
                )
                ==
                frozen[
                    "all_null_row_count"
                ]
            )


            if (
                frozen[
                    "canonicalization"
                ]
                ==
                "dropna_how_all"
            ):

                canonical = (
                    dataframe
                    .dropna(
                        how="all"
                    )
                    .copy()
                )

            else:

                canonical = (
                    dataframe.copy()
                )


            assert (
                len(
                    canonical
                )
                ==
                frozen[
                    "canonical_row_count"
                ]
            )


            assert (
                int(
                    canonical
                    .duplicated()
                    .sum()
                )
                ==
                frozen[
                    "canonical_duplicate_count"
                ]
            )


        print(
            "[PASS] private local CSVs match frozen authority"
        )


    else:

        print(
            "[INFO] private local source verification skipped"
        )


    print(
        (
            "Input SHA256                            "
            +
            sha256_file(
                INPUT_PATH
            )
        )
    )


    print(
        (
            "Gold SHA256                             "
            +
            sha256_file(
                GOLD_PATH
            )
        )
    )


    print()
    print(
        "PASS - V8-BIS Lapage acceptance contract v0.1"
    )


if __name__ == "__main__":
    main()
