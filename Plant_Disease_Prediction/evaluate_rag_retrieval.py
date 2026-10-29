"""Evaluate lexical RAG retrieval against all 38 PlantVision classifier labels.

Run from Plant_Disease_Prediction with:
    python evaluate_rag_retrieval.py

This evaluates the deterministic lexical baseline without API keys. It reports which
classes have source-backed passages and ensures unsupported labels do not get a
misleading, unrelated source. Semantic retrieval is tested separately with mocked
vectors in the unit tests; live provider evaluation requires configured API keys.
"""
from rag_knowledge import retrieve_knowledge


# (classifier label, retrieval query, expected top passage ID or None when not covered)
CLASS_EVALUATION_CASES = [
    ("Apple___Apple_scab", "Apple apple scab", "PV-APPLE-SCAB-01"),
    ("Apple___Black_rot", "Apple black rot", "PV-APPLE-BLACK-ROT-01"),
    ("Apple___Cedar_apple_rust", "Apple cedar apple rust", "PV-APPLE-CEDAR-RUST-01"),
    ("Apple___healthy", "Apple healthy", "PV-GENERAL-01"),
    ("Blueberry___healthy", "Blueberry healthy", "PV-GENERAL-01"),
    ("Cherry_(including_sour)___Powdery_mildew", "Cherry powdery mildew", "PV-CHERRY-POWDERY-01"),
    ("Cherry_(including_sour)___healthy", "Cherry healthy", "PV-GENERAL-01"),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Cercospora gray leaf spot", "PV-CORN-LEAF-DISEASES-01"),
    ("Corn_(maize)___Common_rust_", "Corn common rust", "PV-CORN-RUST-01"),
    ("Corn_(maize)___Northern_Leaf_Blight", "Corn northern leaf blight", "PV-CORN-LEAF-DISEASES-01"),
    ("Corn_(maize)___healthy", "Corn healthy", "PV-GENERAL-01"),
    ("Grape___Black_rot", "Grape black rot", "PV-GRAPE-BLACK-ROT-01"),
    ("Grape___Esca_(Black_Measles)", "Grape Esca Black Measles", "PV-GRAPE-ESCA-01"),
    ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape Isariopsis leaf blight spot", "PV-GRAPE-LEAF-BLIGHT-01"),
    ("Grape___healthy", "Grape healthy", "PV-GENERAL-01"),
    ("Orange___Haunglongbing_(Citrus_greening)", "Orange Huanglongbing citrus greening", "PV-CITRUS-HLB-01"),
    ("Peach___Bacterial_spot", "Peach bacterial spot", "PV-PEACH-BACTERIAL-SPOT-01"),
    ("Peach___healthy", "Peach healthy", "PV-GENERAL-01"),
    ("Pepper,_bell___Bacterial_spot", "Pepper bacterial spot", "PV-PEPPER-BACTERIAL-SPOT-01"),
    ("Pepper,_bell___healthy", "Pepper healthy", "PV-GENERAL-01"),
    ("Potato___Early_blight", "Potato early blight", "PV-POTATO-EARLY-01"),
    ("Potato___Late_blight", "Potato late blight", "PV-POTATO-LATE-01"),
    ("Potato___healthy", "Potato healthy", "PV-GENERAL-01"),
    ("Raspberry___healthy", "Raspberry healthy", "PV-GENERAL-01"),
    ("Soybean___healthy", "Soybean healthy", "PV-GENERAL-01"),
    ("Squash___Powdery_mildew", "Squash powdery mildew", "PV-SQUASH-POWDERY-01"),
    ("Strawberry___Leaf_scorch", "Strawberry leaf scorch", "PV-STRAWBERRY-LEAF-SCORCH-01"),
    ("Strawberry___healthy", "Strawberry healthy", "PV-GENERAL-01"),
    ("Tomato___Bacterial_spot", "Tomato bacterial spot", "PV-TOMATO-BACTERIAL-SPOT-01"),
    ("Tomato___Early_blight", "Tomato early blight", "PV-TOMATO-EARLY-BLIGHT-01"),
    ("Tomato___Late_blight", "Tomato late blight", "PV-TOMATO-LATE-BLIGHT-01"),
    ("Tomato___Leaf_Mold", "Tomato leaf mold", "PV-TOMATO-LEAF-MOLD-01"),
    ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot", "PV-TOMATO-SEPTORIA-01"),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "Tomato spider mites two spotted mite", "PV-SPIDER-MITES-01"),
    ("Tomato___Target_Spot", "Tomato target spot", "PV-TOMATO-TARGET-SPOT-01"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato yellow leaf curl virus", "PV-TOMATO-TYLCV-01"),
    ("Tomato___Tomato_mosaic_virus", "Tomato mosaic virus", "PV-TOMATO-MOSAIC-01"),
    ("Tomato___healthy", "Tomato healthy", "PV-GENERAL-01"),
]


def evaluate():
    if len(CLASS_EVALUATION_CASES) != 38:
        raise RuntimeError(
            f"Expected 38 classifier classes, found {len(CLASS_EVALUATION_CASES)}."
        )

    supported = 0
    supported_pass = 0
    unsupported = 0
    unsupported_false_positives = 0
    failures = []

    for label, query, expected_id in CLASS_EVALUATION_CASES:
        results = retrieve_knowledge(query, top_k=3)
        top_id = results[0]["id"] if results else None

        if expected_id is None:
            unsupported += 1
            passed = not results
            if not passed:
                unsupported_false_positives += 1
        else:
            supported += 1
            passed = top_id == expected_id
            if passed:
                supported_pass += 1

        status = "PASS" if passed else "FAIL"
        expected = expected_id or "NO_SOURCE_SPECIFIC_PASSAGE"
        actual = top_id or "NO_RESULT"
        print(f"{status:4} | {label:52} | expected={expected:32} | top={actual}")
        if not passed:
            failures.append((label, expected, actual))

    coverage = (supported / len(CLASS_EVALUATION_CASES)) * 100
    print("\nRAG retrieval evaluation (lexical baseline)")
    print(f"Classifier classes evaluated: {len(CLASS_EVALUATION_CASES)}")
    print(f"Classes with source-backed guidance: {supported}/38 ({coverage:.1f}%)")
    print(f"Supported-query top-1 accuracy: {supported_pass}/{supported}")
    print(f"Unsupported classes returning a passage: {unsupported_false_positives}/{unsupported}")

    if failures:
        print("\nFailures:")
        for label, expected, actual in failures:
            print(f"- {label}: expected {expected!r}, got {actual!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
