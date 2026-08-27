from __future__ import annotations


import ast

from pathlib import (
    Path,
)


TEST_RULE_VERSION = (
    "offline_evaluation_boundary_test_v0.1"
)


APP_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
    /
    "app"
)


OFFLINE_PREFIXES = (
    "app/evals/",
    "app/evaluation/",
)


EXPECTED_DIRECT_MODEL_CALLERS = {
    (
        "app/evals/"
        "analytical_planner_frozen_runner_v1_0.py"
    ),

    (
        "app/evals/"
        "analytical_planner_model_runner_v0_9.py"
    ),

    (
        "app/evals/"
        "dataset_dependency_extractor_v0_8.py"
    ),

    (
        "app/evals/"
        "decision_router_runner_v0_7.py"
    ),

    (
        "app/evals/"
        "decision_router_runner_v0_7_1.py"
    ),

    (
        "app/evals/"
        "decision_router_runner_v0_7_2.py"
    ),

    (
        "app/evals/"
        "frozen_runner_v0_6.py"
    ),

    (
        "app/evals/"
        "ollama_baseline.py"
    ),

    (
        "app/evals/"
        "ollama_baseline_v0_2.py"
    ),

    (
        "app/evals/"
        "ollama_baseline_v0_3.py"
    ),

    (
        "app/evaluation/"
        "rag_relevance_runner.py"
    ),
}


ROUTE_METHODS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "options",
    "head",
    "websocket",
}


FORBIDDEN_NETWORK_IMPORTS = {
    "requests",
    "httpx",
    "urllib.request",
    "aiohttp",
}


FORBIDDEN_NETWORK_CALLS = {
    "urlopen",
    "build_opener",
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.patch",
    "requests.delete",
    "httpx.get",
    "httpx.post",
    "httpx.Client",
    "httpx.AsyncClient",
    "ollama.Client",
    "ollama.AsyncClient",
}


def relative_path(
    path: Path,
) -> str:

    return (
        "app/"
        +
        path.relative_to(
            APP_ROOT
        )
        .as_posix()
    )


def is_offline_path(
    relative: str,
) -> bool:

    return any(
        relative.startswith(
            prefix
        )

        for prefix
        in OFFLINE_PREFIXES
    )


def python_files(
):

    for path in sorted(
        APP_ROOT.rglob(
            "*.py"
        )
    ):

        if ".before_" in path.name:
            continue

        yield path


def offline_python_files(
):

    for path in python_files():

        if is_offline_path(
            relative_path(
                path
            )
        ):
            yield path


def production_python_files(
):

    for path in python_files():

        if not is_offline_path(
            relative_path(
                path
            )
        ):
            yield path


def parse_file(
    path: Path,
) -> ast.Module:

    return ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        ),
        filename=str(
            path
        ),
    )


def dotted_name(
    node: ast.AST,
) -> (
    str
    | None
):

    if isinstance(
        node,
        ast.Name,
    ):
        return node.id


    if isinstance(
        node,
        ast.Attribute,
    ):

        parent = (
            dotted_name(
                node.value
            )
        )


        if parent:
            return (
                f"{parent}.{node.attr}"
            )


        return node.attr


    return None


def imports_provider_client(
    tree: ast.Module,
) -> bool:

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.ImportFrom,
        ):
            continue


        if (
            node.module
            !=
            "app.ai.provider"
        ):
            continue


        if any(
            alias.name
            ==
            "client"

            for alias
            in node.names
        ):
            return True


    return False


def test_rule_version(
) -> None:

    assert (
        TEST_RULE_VERSION
        ==
        "offline_evaluation_boundary_test_v0.1"
    )


def test_production_cannot_import_evaluation(
) -> None:

    violations = []


    for path in production_python_files():

        tree = (
            parse_file(
                path
            )
        )

        relative = (
            relative_path(
                path
            )
        )


        for node in ast.walk(
            tree
        ):

            if isinstance(
                node,
                ast.ImportFrom,
            ):

                module = (
                    node.module
                    or
                    ""
                )


                if (
                    module.startswith(
                        "app.evals"
                    )
                    or
                    module.startswith(
                        "app.evaluation"
                    )
                ):

                    violations.append(
                        (
                            relative,
                            node.lineno,
                            module,
                        )
                    )


            elif isinstance(
                node,
                ast.Import,
            ):

                for alias in node.names:

                    if (
                        alias.name.startswith(
                            "app.evals"
                        )
                        or
                        alias.name.startswith(
                            "app.evaluation"
                        )
                    ):

                        violations.append(
                            (
                                relative,
                                node.lineno,
                                alias.name,
                            )
                        )


    assert (
        violations
        ==
        []
    ), (
        "Production code imports offline "
        f"evaluation code: {violations}"
    )


def test_evaluation_exposes_no_http_routes(
) -> None:

    violations = []


    for path in offline_python_files():

        tree = (
            parse_file(
                path
            )
        )

        relative = (
            relative_path(
                path
            )
        )


        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue


            for decorator in (
                node.decorator_list
            ):

                if not isinstance(
                    decorator,
                    ast.Call,
                ):
                    continue


                name = (
                    dotted_name(
                        decorator.func
                    )
                )


                if not name:
                    continue


                parts = (
                    name.split(
                        "."
                    )
                )


                if (
                    len(
                        parts
                    )
                    >=
                    2
                    and
                    parts[
                        -1
                    ]
                    in
                    ROUTE_METHODS
                    and
                    parts[
                        -2
                    ]
                    in {
                        "router",
                        "app",
                    }
                ):

                    violations.append(
                        (
                            relative,
                            node.lineno,
                            name,
                        )
                    )


    assert (
        violations
        ==
        []
    ), (
        "Offline evaluation code exposes "
        f"HTTP routes: {violations}"
    )


def test_direct_eval_model_calls_are_allowlisted(
) -> None:

    callers = set()

    unexpected_calls = []


    for path in offline_python_files():

        tree = (
            parse_file(
                path
            )
        )

        relative = (
            relative_path(
                path
            )
        )


        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue


            name = (
                dotted_name(
                    node.func
                )
            )


            if not name:
                continue


            if (
                name
                ==
                "client.chat"
            ):

                callers.add(
                    relative
                )

                continue


            if (
                name.startswith(
                    "client."
                )
                and
                name.split(
                    "."
                )[
                    -1
                ]
                in {
                    "chat",
                    "embed",
                    "generate",
                    "embeddings",
                }
            ):

                unexpected_calls.append(
                    (
                        relative,
                        node.lineno,
                        name,
                    )
                )


    assert (
        unexpected_calls
        ==
        []
    ), (
        "Unexpected direct evaluation "
        f"model calls: {unexpected_calls}"
    )


    assert (
        callers
        ==
        EXPECTED_DIRECT_MODEL_CALLERS
    ), (
        "Direct evaluation model caller "
        "allowlist changed.\n"
        f"Expected: {sorted(EXPECTED_DIRECT_MODEL_CALLERS)}\n"
        f"Actual: {sorted(callers)}"
    )


def test_eval_model_calls_use_hardened_provider(
) -> None:

    violations = []


    for path in offline_python_files():

        tree = (
            parse_file(
                path
            )
        )

        relative = (
            relative_path(
                path
            )
        )


        has_direct_model_call = (
            False
        )


        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue


            if (
                dotted_name(
                    node.func
                )
                ==
                "client.chat"
            ):

                has_direct_model_call = (
                    True
                )

                break


        if (
            has_direct_model_call
            and
            not imports_provider_client(
                tree
            )
        ):

            violations.append(
                relative
            )


    assert (
        violations
        ==
        []
    ), (
        "Evaluation model callers bypass "
        "app.ai.provider.client: "
        f"{violations}"
    )


def test_eval_owns_no_independent_network_transport(
) -> None:

    violations = []


    for path in offline_python_files():

        tree = (
            parse_file(
                path
            )
        )

        relative = (
            relative_path(
                path
            )
        )


        for node in ast.walk(
            tree
        ):

            if isinstance(
                node,
                ast.Import,
            ):

                for alias in node.names:

                    if (
                        alias.name
                        in
                        FORBIDDEN_NETWORK_IMPORTS
                    ):

                        violations.append(
                            (
                                relative,
                                node.lineno,
                                (
                                    "import "
                                    f"{alias.name}"
                                ),
                            )
                        )


            elif isinstance(
                node,
                ast.ImportFrom,
            ):

                module = (
                    node.module
                    or
                    ""
                )


                if (
                    module
                    in
                    FORBIDDEN_NETWORK_IMPORTS
                ):

                    violations.append(
                        (
                            relative,
                            node.lineno,
                            (
                                "from "
                                f"{module}"
                            ),
                        )
                    )


            elif isinstance(
                node,
                ast.Call,
            ):

                name = (
                    dotted_name(
                        node.func
                    )
                )


                if (
                    name
                    in
                    FORBIDDEN_NETWORK_CALLS
                ):

                    violations.append(
                        (
                            relative,
                            node.lineno,
                            name,
                        )
                    )


    assert (
        violations
        ==
        []
    ), (
        "Offline evaluation code owns "
        "independent network transport: "
        f"{violations}"
    )


def test_offline_packages_are_exact(
) -> None:

    assert (
        OFFLINE_PREFIXES
        ==
        (
            "app/evals/",
            "app/evaluation/",
        )
    )


def main(
) -> None:

    print(
        "=== DATALENS OFFLINE EVALUATION "
        "BOUNDARY v0.1 ==="
    )

    print()


    tests = [
        (
            "Offline boundary rule version",
            test_rule_version,
        ),
        (
            "Production cannot import evaluation",
            test_production_cannot_import_evaluation,
        ),
        (
            "Evaluation exposes no HTTP routes",
            test_evaluation_exposes_no_http_routes,
        ),
        (
            "Direct model callers exactly allowlisted",
            test_direct_eval_model_calls_are_allowlisted,
        ),
        (
            "Eval model calls use hardened provider",
            test_eval_model_calls_use_hardened_provider,
        ),
        (
            "Eval owns no independent network transport",
            test_eval_owns_no_independent_network_transport,
        ),
        (
            "Offline package scope is exact",
            test_offline_packages_are_exact,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - 7/7 offline evaluation "
        "boundary checks"
    )

    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
