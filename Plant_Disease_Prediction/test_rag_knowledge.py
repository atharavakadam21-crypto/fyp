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

    def test_results_include_traceable_source_urls(self):
        result = retrieve_knowledge("apple scab", top_k=1)[0]
        for field in ("id", "title", "source", "source_title", "source_url", "source_publisher", "text", "score"):
            self.assertIn(field, result)
        self.assertTrue(result["source_url"].startswith("https://"))

    def test_source_backed_crop_disease_queries(self):
        cases = {
            "Grape black rot": "PV-GRAPE-BLACK-ROT-01",
            "Orange Huanglongbing citrus greening": "PV-CITRUS-HLB-01",
            "Pepper bacterial spot": "PV-PEPPER-BACTERIAL-SPOT-01",
            "Tomato early blight": "PV-TOMATO-EARLY-BLIGHT-01",
            "Tomato late blight": "PV-TOMATO-LATE-BLIGHT-01",
            "Tomato yellow leaf curl virus": "PV-TOMATO-TYLCV-01",
            "Squash powdery mildew": "PV-SQUASH-POWDERY-01",
        }
        for query, expected_id in cases.items():
            with self.subTest(query=query):
                results = retrieve_knowledge(query, top_k=3)
                self.assertTrue(results)
                self.assertEqual(results[0]["id"], expected_id)

    def test_similar_but_different_disease_is_not_misrepresented(self):
        results = retrieve_knowledge("Grape Esca Black Measles", top_k=3)
        self.assertEqual(results, [])

    def test_every_document_has_source_metadata(self):
        for doc in KNOWLEDGE_DOCUMENTS:
            with self.subTest(document=doc["id"]):
                self.assertTrue(doc.get("source_title"))
                self.assertTrue(doc.get("source_publisher"))
                self.assertTrue(doc.get("source_url", "").startswith("https://"))

    def test_top_k_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    retrieve_knowledge("apple scab", top_k=value)


if __name__ == "__main__":
    unittest.main()
