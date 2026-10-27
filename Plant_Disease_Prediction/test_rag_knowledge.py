"""Unit tests for PlantVision AI's dependency-free retrieval baseline."""
import unittest

from rag_knowledge import KNOWLEDGE_DOCUMENTS, retrieve_knowledge


class RetrievalKnowledgeTests(unittest.TestCase):
    def test_corpus_is_not_empty(self):
        self.assertGreaterEqual(len(KNOWLEDGE_DOCUMENTS), 5)

    def test_potato_early_blight_ranks_specific_passage_first(self):
        results = retrieve_knowledge("Potato early blight", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "PV-POTATO-EARLY-01")

    def test_potato_late_blight_ranks_specific_passage_first(self):
        results = retrieve_knowledge("Potato late blight", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "PV-POTATO-LATE-01")

    def test_apple_scab_ranks_specific_passage_first(self):
        results = retrieve_knowledge("Apple scab", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "PV-APPLE-SCAB-01")

    def test_viral_query_retrieves_viral_precautions(self):
        results = retrieve_knowledge("viral disease mosaic curling", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "PV-VIRAL-01")

    def test_empty_query_returns_no_results(self):
        self.assertEqual(retrieve_knowledge("   "), [])

    def test_generic_query_returns_no_results(self):
        self.assertEqual(retrieve_knowledge("plant disease symptoms prevention treatment"), [])

    def test_uncovered_disease_returns_no_results(self):
        self.assertEqual(retrieve_knowledge("grape black rot"), [])

    def test_unrelated_query_returns_no_results(self):
        self.assertEqual(retrieve_knowledge("quantum physics orbital velocity"), [])

    def test_healthy_query_returns_only_general_guidance(self):
        results = retrieve_knowledge("Tomato healthy", top_k=3)
        self.assertEqual([item["id"] for item in results], ["PV-GENERAL-01"])

    def test_results_include_traceable_metadata(self):
        result = retrieve_knowledge("apple scab", top_k=1)[0]
        for field in ("id", "title", "source", "text", "score"):
            self.assertIn(field, result)

    def test_top_k_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    retrieve_knowledge("apple scab", top_k=value)


if __name__ == "__main__":
    unittest.main()
