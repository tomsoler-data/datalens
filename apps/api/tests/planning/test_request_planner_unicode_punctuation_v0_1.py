from types import (
    SimpleNamespace,
)

from app.planning.request_planner import (
    classify_request,
    normalize_text,
)


print(
    "=== DATALENS REQUEST PLANNER "
    "UNICODE PUNCTUATION v0.1 ==="
)

print()


objective = (
    "\u00c9volution du chiffre "
    "d\u2019affaires / moyenne mobile."
)


normalized = (
    normalize_text(
        objective
    )
)


assert (
    normalized
    ==
    "evolution du chiffre d affaires moyenne mobile"
)


print(
    "[PASS] curly apostrophe preserves the word boundary"
)


claim = SimpleNamespace(
    statement=
        objective,

    evidence_quote=
        objective,

    context_quote=
        None,
)


kind = (
    classify_request(
        claim
    )
)


assert (
    kind
    ==
    "revenue_moving_average"
)


print(
    "[PASS] Unicode revenue prompt routes to revenue_moving_average"
)


ascii_objective = (
    "Evolution du chiffre "
    "d'affaires / moyenne mobile."
)


assert (
    normalize_text(
        ascii_objective
    )
    ==
    normalized
)


print(
    "[PASS] ASCII and Unicode apostrophes normalize consistently"
)

print()
print(
    "PASS - request planner Unicode punctuation v0.1"
)
