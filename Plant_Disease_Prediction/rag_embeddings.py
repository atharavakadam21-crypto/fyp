"""Small, dependency-free adapters for OpenAI and Gemini text embeddings."""
import json
import math
import urllib.error
import urllib.request


OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"


def _validate_vectors(vectors, expected_count):
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise RuntimeError("Embedding provider returned an unexpected number of vectors.")
    cleaned = []
    dimension = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Embedding provider returned an empty vector.")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as error:
            raise RuntimeError("Embedding provider returned non-numeric vector values.") from error
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Embedding provider returned non-finite vector values.")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise RuntimeError("Embedding provider returned inconsistent vector dimensions.")
        cleaned.append(values)
    return cleaned


def embed_texts(provider, api_key, texts, timeout=20):
    """Embed texts using the selected provider; raises safe, actionable errors."""
    if provider not in ("OpenAI", "Gemini"):
        raise ValueError("provider must be 'OpenAI' or 'Gemini'")
    if not api_key or not str(api_key).strip():
        raise RuntimeError(f"{provider} API key is not configured.")
    if not isinstance(texts, list) or not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("texts must be a non-empty list of non-empty strings")

    if provider == "OpenAI":
        endpoint = "https://api.openai.com/v1/embeddings"
        payload = {
            "model": OPENAI_EMBEDDING_MODEL,
            "input": texts,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_EMBEDDING_MODEL}:batchEmbedContents"
        )
        payload = {
            "requests": [
                {
                    "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts
            ]
        }
        headers = {
            "x-goog-api-key": str(api_key),
            "Content-Type": "application/json",
        }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"{provider} embedding API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach {provider} embedding API: {error.reason}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"{provider} embedding API returned invalid JSON.") from error

    if provider == "OpenAI":
        items = data.get("data", [])
        items.sort(key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in items]
    else:
        vectors = [
            item.get("values")
            for item in data.get("embeddings", [])
        ]
    return _validate_vectors(vectors, len(texts))


def cosine_similarity(left, right):
    """Return cosine similarity for two finite, equal-length vectors."""
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        raise ValueError("vectors must be lists or tuples")
    if not left or len(left) != len(right):
        raise ValueError("vectors must have the same non-zero dimension")
    a = [float(value) for value in left]
    b = [float(value) for value in right]
    if not all(math.isfinite(value) for value in a + b):
        raise ValueError("vectors must contain only finite numbers")
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
