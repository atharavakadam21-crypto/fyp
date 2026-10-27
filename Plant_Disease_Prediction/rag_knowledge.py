"""Small dependency-free retrieval knowledge base for PlantVision AI.

This is a starter RAG corpus, not an authoritative pesticide guide. Expand it with
reviewed agricultural-extension documents before relying on it for field decisions.
"""
import re
from collections import Counter

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "PV-GENERAL-01",
        "title": "General plant disease prevention",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "General prevention: inspect plants regularly; avoid moving from infected "
            "plants to healthy plants without cleaning hands and tools; remove diseased "
            "plant material where appropriate and dispose of it according to local guidance; "
            "use clean planting material; improve spacing and airflow where suitable; water "
            "the root zone when practical and avoid prolonged leaf wetness; clean containers "
            "and tools; rotate crops only when the disease organism and crop rotation make "
            "that useful. These are general practices, not a diagnosis-specific prescription."
        ),
    },
    {
        "id": "PV-FUNGAL-01",
        "title": "General fungal leaf disease management",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "For suspected fungal leaf diseases, check whether spots expand, whether lesions "
            "have distinctive margins or growth, and whether symptoms appear on new leaves. "
            "Avoid overhead watering when practical, allow foliage to dry, maintain suitable "
            "airflow, sanitize tools, and manage infected debris where recommended for that "
            "crop and disease. Several disorders look alike, so confirm uncertain cases. "
            "Do not apply fungicides without checking local registered guidance and labels."
        ),
    },
    {
        "id": "PV-BACTERIAL-01",
        "title": "General bacterial plant disease precautions",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "For suspected bacterial plant disease, avoid handling wet plants, clean tools "
            "between plants, use clean planting material, reduce splashing water where "
            "practical, and remove affected material only when recommended for the crop. "
            "Bacterial and fungal spots can look similar. Seek local expert confirmation "
            "before choosing a treatment; some products do not cure infected plant tissue."
        ),
    },
    {
        "id": "PV-VIRAL-01",
        "title": "General viral plant disease precautions",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "For suspected viral disease, inspect for mosaic patterns, curling, distortion, "
            "or stunting, but these signs are not definitive. Control insect vectors only "
            "with locally appropriate integrated pest management advice, sanitize tools "
            "when relevant, use clean planting material, and ask an agricultural expert "
            "about removal of infected plants. There is generally no curative treatment "
            "for a plant once systemically infected by a virus."
        ),
    },
    {
        "id": "PV-POTATO-EARLY-01",
        "title": "Potato early blight: cautious field checks",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "Potato early blight is commonly associated with brown leaf lesions that may "
            "show concentric rings, often on older leaves. Similar symptoms can have other "
            "causes. Reduce plant stress with appropriate irrigation and nutrition, avoid "
            "prolonged leaf wetness where practical, maintain suitable spacing, remove crop "
            "residues according to local recommendations, and rotate crops where suitable. "
            "Use locally approved disease-management recommendations if disease is confirmed."
        ),
    },
    {
        "id": "PV-POTATO-LATE-01",
        "title": "Potato late blight: urgent escalation",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "Potato late blight can progress rapidly in favorable cool, wet conditions and "
            "may cause water-soaked lesions that turn brown or dark, sometimes with pale "
            "growth at lesion margins in humid conditions. This is not a visual diagnosis. "
            "If rapid spread or suspected late blight occurs, isolate work in affected areas "
            "where practical and contact local agricultural extension promptly. Avoid moving "
            "infected plant material. Follow locally registered guidance rather than "
            "improvised pesticide treatments."
        ),
    },
    {
        "id": "PV-APPLE-SCAB-01",
        "title": "Apple scab: cautious field checks",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "Apple scab can cause olive-green to dark scabby lesions on leaves and fruit; "
            "leaf symptoms and fruit marks vary with stage and conditions. Similar marks "
            "can have other causes. Monitor leaves and fruit, remove fallen infected leaves "
            "where recommended, improve airflow with appropriate pruning, and follow local "
            "orchard sanitation and disease-management guidance. Do not treat a model output "
            "as a confirmed diagnosis."
        ),
    },
    {
        "id": "PV-CORN-RUST-01",
        "title": "Corn common rust: cautious field checks",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "Corn common rust may produce raised rust-colored pustules on leaf surfaces. "
            "Other leaf diseases can resemble rust. Monitor spread across leaves, record "
            "conditions, and consult local crop advisers about resistant hybrids and "
            "region-specific management. Avoid recommending a chemical treatment solely "
            "from an image classifier."
        ),
    },
    {
        "id": "PV-TOMATO-LEAF-01",
        "title": "Tomato leaf spot and blight: general checks",
        "source": "PlantVision curated baseline guidance",
        "text": (
            "Tomato leaf spots and blights have overlapping visual symptoms. Check the "
            "location and pattern of lesions, lower versus upper leaves, stems and fruit, "
            "and whether symptoms are spreading. Water near the soil when practical, "
            "maintain airflow, sanitize tools, avoid working among wet plants, and manage "
            "plant debris according to local recommendations. Confirm the likely cause "
            "before selecting disease-specific treatment."
        ),
    },
]

def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())

def retrieve_knowledge(query, top_k=3):
    """Return top matching corpus passages using transparent lexical retrieval.

    This starter uses token overlap rather than vector embeddings; it is a simple,
    dependency-free RAG baseline and can later be upgraded to embedding retrieval.
    """
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return []
    scored = []
    for doc in KNOWLEDGE_DOCUMENTS:
        doc_tokens = set(_tokens(doc["title"] + " " + doc["text"]))
        overlap = len(query_tokens & doc_tokens)
        if overlap:
            # Normalize overlap to avoid favoring longer documents.
            score = overlap / max(1, len(query_tokens) ** 0.5 * len(doc_tokens) ** 0.25)
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {**doc, "score": round(score, 4)}
        for score, doc in scored[:max(1, int(top_k))]
    ]
