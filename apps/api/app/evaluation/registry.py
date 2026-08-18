from __future__ import annotations

from app.evaluation.registry_schemas import (
    BenchmarkRegistrySnapshot,
    BenchmarkSplit,
    SemanticBenchmarkSuite,
)


# ============================================================
# REGISTRY
# ============================================================

class SemanticBenchmarkRegistry:
    def __init__(
        self,
    ) -> None:
        self._suites: dict[
            str,
            SemanticBenchmarkSuite,
        ] = {}


    # ========================================================
    # REGISTER
    # ========================================================

    def register(
        self,
        suite: SemanticBenchmarkSuite,
    ) -> None:
        if (
            suite.benchmark_id
            in self._suites
        ):
            raise ValueError(
                "Benchmark already registered: "
                f"{suite.benchmark_id}"
            )


        self._suites[
            suite.benchmark_id
        ] = suite


    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        benchmark_id: str,
    ) -> SemanticBenchmarkSuite:
        try:
            return self._suites[
                benchmark_id
            ]

        except KeyError as exc:
            raise KeyError(
                "Unknown benchmark: "
                f"{benchmark_id}"
            ) from exc


    # ========================================================
    # EXISTS
    # ========================================================

    def contains(
        self,
        benchmark_id: str,
    ) -> bool:
        return (
            benchmark_id
            in self._suites
        )


    # ========================================================
    # LIST
    # ========================================================

    def list_suites(
        self,
        *,
        split: BenchmarkSplit | None = None,
        domain: str | None = None,
    ) -> list[
        SemanticBenchmarkSuite
    ]:
        suites = list(
            self._suites.values()
        )


        if (
            split
            is not None
        ):
            suites = [
                suite
                for suite
                in suites
                if (
                    suite.split
                    ==
                    split
                )
            ]


        if (
            domain
            is not None
        ):
            normalized_domain = (
                domain
                .strip()
                .casefold()
            )


            suites = [
                suite
                for suite
                in suites
                if (
                    suite.domain
                    .strip()
                    .casefold()
                    ==
                    normalized_domain
                )
            ]


        return sorted(
            suites,
            key=lambda suite:
                suite.benchmark_id,
        )


    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(
        self,
    ) -> BenchmarkRegistrySnapshot:
        suites = self.list_suites()


        development_count = sum(
            suite.split
            ==
            "development"

            for suite
            in suites
        )


        regression_count = sum(
            suite.split
            ==
            "regression"

            for suite
            in suites
        )


        holdout_count = sum(
            suite.split
            ==
            "holdout"

            for suite
            in suites
        )


        dataset_count = sum(
            len(
                suite.datasets
            )

            for suite
            in suites
        )


        domains = sorted(
            {
                suite.domain
                for suite
                in suites
            }
        )


        benchmark_ids = [
            suite.benchmark_id
            for suite
            in suites
        ]


        return BenchmarkRegistrySnapshot(
            suite_count=
                len(
                    suites
                ),

            development_count=
                development_count,

            regression_count=
                regression_count,

            holdout_count=
                holdout_count,

            dataset_count=
                dataset_count,

            domains=
                domains,

            benchmark_ids=
                benchmark_ids,
        )


# ============================================================
# DEFAULT REGISTRY
# ============================================================

def build_default_benchmark_registry(
) -> SemanticBenchmarkRegistry:
    from app.evaluation.benchmarks import (
        build_builtin_semantic_benchmarks,
    )


    registry = (
        SemanticBenchmarkRegistry()
    )


    for suite in (
        build_builtin_semantic_benchmarks()
    ):
        registry.register(
            suite
        )


    return registry
