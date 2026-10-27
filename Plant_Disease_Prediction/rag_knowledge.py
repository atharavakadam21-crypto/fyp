"""Small dependency-free retrieval knowledge base for PlantVision AI.

This is a starter RAG corpus, not an authoritative pesticide guide. Expand it with
reviewed agricultural-extension documents before relying on it for field decisions.
"""
import re

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "PV-GENERAL-01",
        "title": "General plant disease prevention",
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
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
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
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
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
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
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
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
        "source": "University of Minnesota Extension",
        "source_title": "Potato leaf spots: early blight and late blight",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/potato/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
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
        "source": "University of Minnesota Extension",
        "source_title": "Potato leaf spots: early blight and late blight",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/potato/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
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
        "source": "University of Minnesota Extension",
        "source_title": "Apple scab of apples and crabapples",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/apple-scab",
        "source_publisher": "University of Minnesota Extension",
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
        "source": "Penn State Extension",
        "source_title": "Not every spot is the same: identifying corn leaf diseases",
        "source_url": "https://extension.psu.edu/not-every-spot-is-the-same-identifying-corn-leaf-diseases",
        "source_publisher": "Penn State Extension",
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
        "source": "UC Statewide IPM Program",
        "source_title": "Tomato pest management guidelines",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato leaf spots and blights have overlapping visual symptoms. Check the "
            "location and pattern of lesions, lower versus upper leaves, stems and fruit, "
            "and whether symptoms are spreading. Water near the soil when practical, "
            "maintain airflow, sanitize tools, avoid working among wet plants, and manage "
            "plant debris according to local recommendations. Confirm the likely cause "
            "before selecting disease-specific treatment."
        ),
    },

    {
        "id": "PV-APPLE-BLACK-ROT-01",
        "title": "Apple black rot: symptoms and field checks",
        "source": "University of Minnesota Extension",
        "source_title": "Apple leaves: spots and blotches",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/fruit/apple/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Apple black rot may cause round leaf spots with purple borders and tan centers. "
            "Fruit may develop larger brown spots with rings, remain firm while rotting, or "
            "become mummified on the tree. Branch cankers may also occur. These signs can "
            "overlap with other problems; confirm the diagnosis before choosing treatment. "
            "Consult local extension guidance for sanitation and pruning recommendations."
        ),
    },
    {
        "id": "PV-APPLE-CEDAR-RUST-01",
        "title": "Apple cedar rust: symptoms and field checks",
        "source": "University of Minnesota Extension",
        "source_title": "Apple leaves: spots and blotches",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/fruit/apple/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Cedar-apple rust can cause yellow spots that become bright orange and develop "
            "a red border. Small dark structures may appear on leaf surfaces, and fruit can "
            "develop rough green-to-brown marks. Similar symptoms have other causes; use "
            "the plant and nearby host context and seek local expert confirmation before "
            "taking disease-specific action."
        ),
    },
    {
        "id": "PV-GRAPE-BLACK-ROT-01",
        "title": "Grape black rot: symptoms and sanitation",
        "source": "Penn State Extension",
        "source_title": "Grape disease: black rot",
        "source_url": "https://extension.psu.edu/grape-disease-black-rot",
        "source_publisher": "Penn State Extension",
        "text": (
            "Grape black rot may cause leaf spots and fruit that shrivels into hard, black "
            "mummies. The fungus can survive in mummified fruit and cane lesions, and wet "
            "conditions can support infection. Extension guidance recommends removing "
            "infected prunings and mummified berries before new growth where appropriate. "
            "Local climate, cultivar, and production system affect management decisions."
        ),
    },
    {
        "id": "PV-CITRUS-HLB-01",
        "title": "Citrus greening (HLB): warning signs",
        "source": "UC Statewide IPM Program",
        "source_title": "Asian citrus psyllid and huanglongbing disease",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/asian-citrus-psyllid-and-huanglongbing-disease/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Huanglongbing, also called citrus greening, can cause uneven blotchy yellow "
            "mottling on leaves, small or lopsided fruit, premature fruit drop, and twig "
            "dieback. Nutrient deficiencies and other problems can also yellow leaves. "
            "A leaf-image prediction cannot confirm HLB; contact a local plant-health "
            "authority or agricultural extension service for appropriate testing and advice."
        ),
    },
    {
        "id": "PV-PEPPER-BACTERIAL-SPOT-01",
        "title": "Pepper bacterial spot: symptoms and prevention",
        "source": "UC Statewide IPM Program",
        "source_title": "Bacterial spot on peppers",
        "source_url": "https://ipm.ucanr.edu/agriculture/peppers/bacterial-spot/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Pepper bacterial spot may cause small, initially water-soaked angular leaf "
            "spots that later turn brown, raised stem lesions, and rough raised fruit spots. "
            "The pathogen can be seedborne and spread on transplants; moisture favors "
            "disease development. Use clean, pathogen-tested seed or healthy transplants, "
            "avoid prolonged leaf wetness where practical, and follow locally applicable "
            "extension guidance. Do not select a chemical treatment from an image alone."
        ),
    },
    {
        "id": "PV-TOMATO-BACTERIAL-SPOT-01",
        "title": "Tomato bacterial spot: symptoms and prevention",
        "source": "UC Statewide IPM Program",
        "source_title": "Bacterial spot on tomato",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/bacterial-spot/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato bacterial spot may cause water-soaked areas on older leaves that turn "
            "dark brown or black. Fruit can develop slightly sunken spots that later become "
            "brown and scabby. The pathogen can be seedborne and survive in crop debris or "
            "alternate hosts. Use clean seed and transplants, reduce splashing water where "
            "practical, and confirm the diagnosis before choosing treatment."
        ),
    },
    {
        "id": "PV-TOMATO-EARLY-BLIGHT-01",
        "title": "Tomato early blight: symptoms and prevention",
        "source": "UC Statewide IPM Program",
        "source_title": "Early blight on tomatoes",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/early-blight-on-tomatoes/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato early blight may cause brown or black spots with concentric rings, "
            "often beginning on older leaves; stems and fruit can also be affected. Leaf "
            "wetness favors infection. Avoid overhead irrigation where practical, maintain "
            "airflow, and rotate crops when suitable for the growing system. Confirm the "
            "cause of spots before disease-specific treatment."
        ),
    },
    {
        "id": "PV-TOMATO-LATE-BLIGHT-01",
        "title": "Tomato late blight: rapid-spread warning signs",
        "source": "UC Statewide IPM Program",
        "source_title": "Late blight on tomato",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/late-blight/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato late blight may cause water-soaked leaf areas that rapidly enlarge into "
            "dark purple-brown blotches. Pale growth may appear near lesions in humid "
            "conditions, and stems or fruit may be affected. The disease can spread rapidly "
            "under favorable conditions. If rapid spread is suspected, limit movement of "
            "plant material and contact local agricultural extension promptly."
        ),
    },
    {
        "id": "PV-TOMATO-TYLCV-01",
        "title": "Tomato yellow leaf curl virus: warning signs",
        "source": "UC Statewide IPM Program",
        "source_title": "Tomato yellow leaf curl",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato yellow leaf curl virus may cause upward-curling, crumpled leaves with "
            "yellowing between or along veins, shortened internodes, and stunted upright "
            "growth. Whiteflies can transmit the virus. Similar symptoms can have other "
            "causes; do not treat a classifier output as confirmation. Seek local expert "
            "advice about testing, resistant varieties, and vector management."
        ),
    },
    {
        "id": "PV-TOMATO-MOSAIC-01",
        "title": "Tomato mosaic virus: symptoms and caution",
        "source": "UC Statewide IPM Program",
        "source_title": "Plant diseases: virus symptoms you may see on tomato",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/diseases-virus-symptoms-you-may-see/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato mosaic virus symptoms vary by cultivar and virus strain. Mosaic "
            "coloring and leaf malformation can occur, but visual symptoms overlap with "
            "other viral, pest, and nutrient problems. Use clean planting material, clean "
            "tools where relevant, and seek local diagnostic advice before removing plants "
            "or selecting a treatment."
        ),
    },
    {
        "id": "PV-SPIDER-MITES-01",
        "title": "Tomato spider mites: signs and nonchemical checks",
        "source": "UC Statewide IPM Program",
        "source_title": "Spider mites",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/spider-mites/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Spider mites may cause fine pale stippling, bronzing, yellowing, leaf drop, "
            "and sometimes webbing. Check leaf undersides with a hand lens because mites "
            "are tiny. Water-stressed plants and hot, dusty conditions can worsen damage. "
            "Confirm mites are present before treating and protect beneficial predators; "
            "follow local guidance for any control product."
        ),
    },
    {
        "id": "PV-SQUASH-POWDERY-01",
        "title": "Squash powdery mildew: symptoms and prevention",
        "source": "University of Minnesota Extension",
        "source_title": "Powdery mildew of cucurbits",
        "source_url": "https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/powdery-mildew-of-cucurbits",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Cucurbit powdery mildew can begin as pale yellow spots and develop white-to-gray "
            "powdery patches on leaf surfaces. Severe infection weakens vines and can lead "
            "to premature fruit ripening. Use suitable spacing, improve air movement, "
            "control weeds, and consider resistant varieties. Confirm the disease before "
            "using a disease-specific treatment."
        ),
    },

    {
        "id": "PV-CORN-LEAF-DISEASES-01",
        "title": "Corn leaf diseases: gray leaf spot and northern leaf blight",
        "source": "Penn State Extension",
        "source_title": "Not every spot is the same: identifying corn leaf diseases",
        "source_url": "https://extension.psu.edu/not-every-spot-is-the-same-identifying-corn-leaf-diseases",
        "source_publisher": "Penn State Extension",
        "text": (
            "Corn gray leaf spot can begin as tan lesions with yellow halos and develop into "
            "longer rectangular gray-to-tan lesions aligned with leaf veins. Corn leaf "
            "diseases can resemble one another, and symptom appearance varies by hybrid and "
            "conditions. Compare the full symptom pattern and seek local crop-adviser "
            "confirmation before selecting disease-specific treatment, including for "
            "suspected northern leaf blight."
        ),
    },
]

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "what", "how",
    "plant", "plants", "disease", "diseases", "symptom", "symptoms",
    "prevention", "prevent", "treatment", "treatments", "management",
    "guidance", "general", "common", "check", "checks", "leaf", "leaves",
    "crop", "crops", "care", "condition", "conditions", "help", "advice", "two", "spotted",
}
_TOKEN_ALIASES = {
    "viral": "virus",
    "viruses": "virus",
    "blights": "blight",
    "spots": "spot",
    "lesions": "lesion",
    "leaves": "leaf",
    "mites": "mite",
}


def _tokens(text):
    """Normalize words and common label variants for deterministic retrieval."""
    raw_tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    normalized = [_TOKEN_ALIASES.get(token, token) for token in raw_tokens]
    return [token for token in normalized if token not in _STOPWORDS]


_EMBEDDING_CACHE = {}


def _retrieve_lexical(query, top_k):
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return []

    # Healthy predictions should not be paired with disease-specific guidance.
    if "healthy" in query_tokens:
        general = next(
            (doc for doc in KNOWLEDGE_DOCUMENTS if doc["id"] == "PV-GENERAL-01"),
            None,
        )
        return [{**general, "score": 1.0, "retrieval_method": "lexical"}] if general else []

    scored = []
    for doc in KNOWLEDGE_DOCUMENTS:
        title_tokens = set(_tokens(doc["title"]))
        body_tokens = set(_tokens(doc["text"]))
        title_matches = query_tokens & title_tokens
        body_matches = query_tokens & body_tokens
        matches = title_matches | body_matches

        if len(query_tokens) == 1:
            if not title_matches:
                continue
        elif len(matches) < 2 or (len(matches) / len(query_tokens)) < 0.7:
            continue

        score = (3 * len(title_matches)) + len(body_matches)
        coverage = len(matches) / len(query_tokens)
        normalized_score = score / (3 * len(query_tokens) + len(query_tokens))
        scored.append((normalized_score + coverage * 0.01, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {**doc, "score": round(score, 4), "retrieval_method": "lexical"}
        for score, doc in scored[:top_k]
    ]


def _retrieve_semantic(query, top_k, provider, api_key):
    import hashlib
    from rag_embeddings import cosine_similarity, embed_texts

    document_texts = [
        f"{doc['title']}\n{doc['text']}"
        for doc in KNOWLEDGE_DOCUMENTS
    ]
    corpus_hash = hashlib.sha256(
        "\n".join(document_texts).encode("utf-8")
    ).hexdigest()
    key_fingerprint = hashlib.sha256(str(api_key).encode("utf-8")).hexdigest()
    cache_key = (provider, key_fingerprint, corpus_hash)

    if cache_key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[cache_key] = embed_texts(provider, api_key, document_texts)
    document_vectors = _EMBEDDING_CACHE[cache_key]
    query_vector = embed_texts(provider, api_key, [query])[0]

    scored = []
    for doc, vector in zip(KNOWLEDGE_DOCUMENTS, document_vectors):
        score = cosine_similarity(query_vector, vector)
        if score >= 0.35:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            **doc,
            "score": round(score, 4),
            "retrieval_method": "semantic",
        }
        for score, doc in scored[:top_k]
    ]


def retrieve_knowledge(query, top_k=3, provider=None, api_key=None):
    """Retrieve source-backed passages semantically when credentials are available.

    If embedding retrieval fails or returns no sufficiently similar passages, the
    dependency-free lexical retriever is used. The result marks which method ran.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    if provider and api_key:
        try:
            results = _retrieve_semantic(query, top_k, provider, api_key)
            if results:
                return results
        except Exception:
            # Keep the assistant usable during provider outages, but label fallback.
            pass
        return [
            {**doc, "retrieval_method": "lexical_fallback"}
            for doc in _retrieve_lexical(query, top_k)
        ]
    return _retrieve_lexical(query, top_k)
