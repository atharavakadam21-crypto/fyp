"""Deterministic quality checks for generated PlantVision guidance.

These checks validate structure, citation identifiers, and obvious pesticide-dose
patterns. They do not prove that every natural-language claim is factually supported.
"""
import re


REQUIRED_HEADINGS = (
    "What the prediction means",
    "Symptoms to check",
    "Immediate actions",
    "Prevention measures",
    "Treatment and ongoing care",
    "When to contact an agricultural expert",
    "Important cautions",
)
_CITATION_PATTERN = re.compile(r"\[(PV-[A-Z0-9-]+)\]")
_DOSE_PATTERN = re.compile(
    r"\b(?:apply|mix|spray|dose|dosage|dilute|dilution)\b"
    r".{0,100}\b\d+(?:\.\d+)?\s*"
    r"(?:ml|mL|g|kg|mg|litres?|liters?|oz|lbs?|gallons?|teaspoons?|tablespoons?)\b",
    re.IGNORECASE,
)


def validate_report(report, retrieved_docs):
    """Return a list of actionable warnings; an empty list means checks passed."""
    issues = []
    if not isinstance(report, str) or not report.strip():
        return ["The generated report is empty."]

    lowered = report.casefold()
    for heading in REQUIRED_HEADINGS:
        if heading.casefold() not in lowered:
            issues.append(f"Missing required report section: {heading}.")

    available_ids = {
        doc.get("id") for doc in retrieved_docs
        if isinstance(doc, dict) and doc.get("id")
    }
    cited_ids = set(_CITATION_PATTERN.findall(report))

    unknown_ids = sorted(cited_ids - available_ids)
    if unknown_ids:
        issues.append(
            "The report cites passage IDs that were not retrieved: "
            + ", ".join(unknown_ids)
            + "."
        )
    if available_ids and not cited_ids:
        issues.append(
            "Retrieved evidence was available, but the report contains no passage citations."
        )

    if _DOSE_PATTERN.search(report):
        issues.append(
            "Possible pesticide/application dosage detected; review and remove unsupported "
            "chemical instructions before presenting this advice."
        )

    if "confirmed diagnosis" not in lowered and "not a diagnosis" not in lowered:
        issues.append(
            "The report does not clearly remind the reader that the model output is not a confirmed diagnosis."
        )

    return issues
