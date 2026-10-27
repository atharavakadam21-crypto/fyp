"""Dependency-free smoke tests for the starter PlantVision retrieval corpus."""
import unittest

from rag_knowledge import KNOWLEDGE_DOCUMENTS, retrieve_knowledge


class RetrievalKnowledgeTests(unittest.TestCase):
    def test_corpus_is_not_empty(self):
        self.assertGreaterEqual(len(KNOWLEDGE_DOCUMENTS), 5)

    def test_potato_early_blight_retrieves_specific_passage(self):
        results = retrieve_knowledge("Potato early blight brown concentric leaf spots prevention", top_k=3)
        self.assertTrue(results)
        self.assertIn("PV-POTATO-EARLY-01", [item["id"] for item in results])

    def test_viral_query_retrieves_viral_precautions(self):
        results = retrieve_knowledge("viral disease mosaic curling plant leaf prevention", top_k=3)
        self.assertTrue(results)
        self.assertIn("PV-VIRAL-01", [item["id"] for item in results])

    def test_empty_query_returns_no_results(self):
        self.assertEqual(retrieve_knowledge("   "), [])

    def test_results_include_traceable_metadata(self):
        result = retrieve_knowledge("apple scab fruit lesions", top_k=1)[0]
        for field in ("id", "title", "source", "text", "score"):
            self.assertIn(field, result)


if __name__ == "__main__":
    unittest.main()
