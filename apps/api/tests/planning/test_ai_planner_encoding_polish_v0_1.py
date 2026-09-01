from __future__ import annotations

import inspect


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def U(
    value: str,
) -> str:
    return (
        value
        .encode(
            "ascii"
        )
        .decode(
            "unicode_escape"
        )
    )


def main() -> None:
    print(
        "=== DATALENS AI PLANNER "
        "ENCODING POLISH v0.1 ==="
    )


    # ========================================================
    # VERSION
    # ========================================================

    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        "ai_analytical_planner_v0.34"
    )


    print(
        "[PASS] planner remains v0.34"
    )


    # ========================================================
    # SYSTEM PROMPT CORRUPTION ABSENT
    # ========================================================

    corrupt_tokens = [
        "?tre",
        "pr?sents",
        "demand?",
        "autoris?",
        "m?trique",
        "agr?g?e",
        "m?me",
        "calcul?e",
        "Op?rateurs",
        "Sup?rieur",
        "sup?rieur",
        "?gal",
        "inf?rieur",
        "S?lection",
        "r?utilise",
        "agr?gation",
        "r?f?rence",
        "group?e",
        "d?partements",
    ]


    for token in (
        corrupt_tokens
    ):
        assert (
            token
            not in
            SYSTEM_PROMPT
        )


    assert (
        " ? "
        not in
        SYSTEM_PROMPT
    )


    print(
        "[PASS] SYSTEM_PROMPT corruption removed"
    )


    # ========================================================
    # CORRECT UNICODE PRESENT
    # ========================================================

    expected = [
        U(
            "TOUJOURS "
            "\\u00eatre pr\\u00e9sents"
        ),
        U(
            "benchmark demand\\u00e9"
        ),
        U(
            "benchmark actif est "
            "autoris\\u00e9"
        ),
        U(
            "Op\\u00e9rateurs"
        ),
        U(
            "S\\u00e9lection"
        ),
        U(
            "Sup\\u00e9rieur "
            "\\u00e0 la moyenne"
        ),
        U(
            "Quels d\\u00e9partements "
            "ont un salaire moyen "
            "sup\\u00e9rieur "
            "\\u00e0 la moyenne globale"
        ),
    ]


    for value in (
        expected
    ):
        assert (
            value
            in
            SYSTEM_PROMPT
        )


    print(
        "[PASS] correct Unicode present"
    )


    # ========================================================
    # LEGITIMATE QUESTION MARKS
    # ========================================================

    assert (
        SYSTEM_PROMPT.count(
            ' ?"'
        )
        >=
        2
    )


    print(
        "[PASS] legitimate question marks preserved"
    )


    # ========================================================
    # build_user_prompt SOURCE
    # ========================================================

    source = (
        inspect.getsource(
            build_user_prompt
        )
    )


    corrupt_source_tokens = [
        "agr?gation",
        "group?e",
        "m?me",
        "calcul?e",
        "sup?rieur",
        " ? ",
    ]


    for token in (
        corrupt_source_tokens
    ):
        assert (
            token
            not in
            source
        )


    assert (
        U(
            "sup\\u00e9rieur "
            "\\u00e0 la moyenne"
        )
        in
        source
    )


    assert (
        U(
            "agr\\u00e9gation "
            "group\\u00e9e"
        )
        in
        source
    )


    assert (
        U(
            "forme \\u00e9quivalente"
        )
        in
        source
    )


    print(
        "[PASS] build_user_prompt Unicode repaired"
    )


    print()
    print(
        "PASS - AI Planner encoding polish v0.1"
    )


if __name__ == "__main__":
    main()
