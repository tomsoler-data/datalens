from app.reporting.selected_report_pdf import (
    clean_text,
    normalize_pdf_ligatures,
)


print()
print(
    "===== PDF LIGATURE NORMALIZATION v0.1 ====="
)
print()


cases = {
    "les \ufb02ops":
        "les flops",

    "\ufb01nal":
        "final",

    "e\ufb00ectif":
        "effectif",

    "\ufb03":
        "ffi",

    "\ufb04":
        "ffl",
}


for source, expected in cases.items():
    result = normalize_pdf_ligatures(
        source
    )

    print(
        repr(source),
        "->",
        repr(result),
    )

    assert (
        result
        ==
        expected
    )


assert (
    clean_text(
        "les \ufb02ops"
    )
    ==
    "les flops"
)


print()
print(
    "[PASS] fl ligature -> fl"
)
print(
    "[PASS] fi ligature -> fi"
)
print(
    "[PASS] ff / ffi / ffl ligatures normalized"
)
print(
    "[PASS] clean_text uses presentation normalization"
)

print()
print(
    "PASS - PDF ligature normalization v0.1"
)
