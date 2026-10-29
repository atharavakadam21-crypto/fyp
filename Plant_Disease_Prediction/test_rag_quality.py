"""Offline regression tests for PlantVision RAG; no API keys or network needed."""
import unittest

from rag_knowledge import retrieve_knowledge
from rag_evaluation import validate_report
from rag_embeddings import cosine_similarity


class RetrievalTests(unittest.TestCase):
    def test_healthy_query_uses_only_general_guidance(self):
        docs = retrieve_knowledge("Corn healthy", top_k=3)
        self.assertEqual([doc["id"] for doc in docs], ["PV-GENERAL-01"])

    def test_crop_disease_query_returns_scored_passages(self):
        docs = retrieve_knowledge("corn common rust", top_k=3)
        self.assertTrue(docs)
        self.assertTrue(any("CORN-RUST" in doc["id"] for doc in docs))
        self.assertTrue(all(isinstance(doc.get("score"), float) for doc in docs))
        self.assertTrue(all(doc.get("source_url", "").startswith("https://") for doc in docs))

    def test_top_k_must_be_positive_integer(self):
        with self.assertRaises(ValueError):
            retrieve_knowledge("tomato leaf spot", top_k=0)


class EvaluationTests(unittest.TestCase):
    def test_missing_sections_are_reported(self):
        issues = validate_report("## 1. What the prediction means\nTentative prediction, not a confirmed diagnosis.", [])
        self.assertTrue(any("Missing required report section" in issue for issue in issues))

    def test_unknown_citation_is_reported(self):
        report = "This is not a confirmed diagnosis [PV-FAKE-999]."
        issues = validate_report(report, [{"id": "PV-GENERAL-01"}])
        self.assertTrue(any("not retrieved" in issue for issue in issues))

    def test_possible_dosage_is_flagged(self):
        issues = validate_report("Not a confirmed diagnosis. Spray 20 ml per litre.", [])
        self.assertTrue(any("dosage" in issue.lower() for issue in issues))


class EmbeddingMathTests(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
