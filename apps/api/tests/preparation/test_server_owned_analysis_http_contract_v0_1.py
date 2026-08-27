from __future__ import annotations

from typing import Any

from main import app


ROUTES = (
    "/planning/ai-preview",
    "/planning/ai-tool-run",
    "/planning/ai-native-run",
    "/analysis/run",
    "/analysis/run-contextualized",
)


def _resolve_schema(
    openapi: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    reference = schema.get("$ref")

    if not reference:
        return schema

    prefix = "#/components/schemas/"

    if not str(reference).startswith(prefix):
        raise AssertionError(
            f"Unsupported OpenAPI schema reference: {reference}"
        )

    schema_name = str(reference)[len(prefix):]

    return openapi["components"]["schemas"][schema_name]


def _multipart_schema(
    openapi: dict[str, Any],
    path: str,
) -> dict[str, Any]:
    request_body = openapi["paths"][path]["post"]["requestBody"]
    content = request_body["content"]

    assert "multipart/form-data" in content, (
        f"{path} must remain a multipart endpoint."
    )

    schema = content["multipart/form-data"]["schema"]

    return _resolve_schema(openapi, schema)


def main() -> None:
    openapi = app.openapi()

    print("=== DATALENS SERVER-OWNED ANALYSIS HTTP CONTRACT v0.1 ===")

    for path in ROUTES:
        schema = _multipart_schema(openapi, path)
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})

        assert "workflow_id" in properties, (
            f"{path} must expose workflow_id."
        )

        assert "workflow_id" in required, (
            f"{path} must require workflow_id."
        )

        assert "dataset_files" in properties, (
            f"{path} keeps dataset_files only as an optional "
            "backward-compatibility field."
        )

        assert "dataset_files" not in required, (
            f"{path} must not require browser dataset files after VALIDATE."
        )

        print(
            f"[PASS] {path} accepts workflow_id without dataset_files"
        )

    contextual_schema = _multipart_schema(
        openapi,
        "/analysis/run-contextualized",
    )
    contextual_required = set(
        contextual_schema.get("required", [])
    )

    assert "document_files" in contextual_required, (
        "Contextualized analysis must still require the document "
        "uploads used for the current RAG request."
    )

    print(
        "[PASS] contextualized analysis still requires document_files"
    )

    print("PASS - server-owned analysis HTTP contract v0.1")


if __name__ == "__main__":
    main()
