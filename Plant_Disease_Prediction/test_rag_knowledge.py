"""Unit tests for PlantVision AI's dependency-free retrieval baseline."""
import json
import unittest
from unittest.mock import patch

from rag_embeddings import cosine_similarity, embed_texts
from rag_knowledge import KNOWLEDGE_DOCUMENTS, _EMBEDDING_CACHE, retrieve_knowledge
from rag_evaluation import REQUIRED_HEADINGS, validate_report


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

    def test_newly_covered_disease_retrieves_specific_passage(self):
        results = retrieve_knowledge("grape black rot", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "PV-GRAPE-BLACK-ROT-01")

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

    def test_esca_retrieves_esca_not_black_rot(self):
        results = retrieve_knowledge("Grape Esca Black Measles", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "PV-GRAPE-ESCA-01")

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


    def test_cosine_similarity_ranks_aligned_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_similarity_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1])

    @patch("rag_embeddings.urllib.request.urlopen")
    def test_openai_embedding_response_is_parsed(self, mocked_urlopen):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }).encode("utf-8")
        mocked_urlopen.return_value = response
        vectors = embed_texts("OpenAI", "test-key", ["first", "second"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])

    @patch("rag_embeddings.urllib.request.urlopen")
    def test_gemini_embedding_response_is_parsed(self, mocked_urlopen):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "embeddings": [{"values": [0.25, 0.75]}]
        }).encode("utf-8")
        mocked_urlopen.return_value = response
        self.assertEqual(embed_texts("Gemini", "test-key", ["query"]), [[0.25, 0.75]])

    def test_semantic_retrieval_ranks_matching_document_first(self):
        _EMBEDDING_CACHE.clear()
        def fake_embed(provider, api_key, texts):
            if len(texts) == len(KNOWLEDGE_DOCUMENTS):
                return [
                    [1.0, 0.0] if "Potato late blight" in text
                    else [0.0, 1.0]
                    for text in texts
                ]
            return [[1.0, 0.0]]
        with patch.dict("os.environ", {"PLANTVISION_SEMANTIC_RETRIEVAL": "1"}):
            with patch("rag_embeddings.embed_texts", side_effect=fake_embed):
                results = retrieve_knowledge(
                    "Potato late blight", top_k=3, provider="OpenAI", api_key="test-key"
                )
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "PV-POTATO-LATE-01")
        self.assertEqual(results[0]["retrieval_method"], "semantic")
        _EMBEDDING_CACHE.clear()

    def test_semantic_failure_uses_marked_lexical_fallback(self):
        with patch.dict("os.environ", {"PLANTVISION_SEMANTIC_RETRIEVAL": "1"}):
            with patch("rag_embeddings.embed_texts", side_effect=RuntimeError("offline")):
                results = retrieve_knowledge(
                    "Apple scab", top_k=1, provider="OpenAI", api_key="test-key"
                )
        self.assertEqual(results[0]["id"], "PV-APPLE-SCAB-01")
        self.assertEqual(results[0]["retrieval_method"], "lexical_fallback")


    def test_valid_report_passes_structure_and_citation_checks(self):
        report = "\n".join(
            f"## {index}. {heading}\nHelpful guidance [PV-APPLE-SCAB-01]."
            for index, heading in enumerate(REQUIRED_HEADINGS, start=1)
        ) + "\nThis is not a confirmed diagnosis."
        issues = validate_report(
            report,
            [{"id": "PV-APPLE-SCAB-01"}],
        )
        self.assertEqual(issues, [])

    def test_missing_sections_are_reported(self):
        issues = validate_report("## What the prediction means\nNot a diagnosis.", [])
        self.assertTrue(any("Missing required report section" in issue for issue in issues))

    def test_unknown_citation_id_is_reported(self):
        report = "\n".join(f"## {heading}" for heading in REQUIRED_HEADINGS)
        report += "\nThis is not a confirmed diagnosis. [PV-UNKNOWN-99]"
        issues = validate_report(report, [{"id": "PV-APPLE-SCAB-01"}])
        self.assertTrue(any("not retrieved" in issue for issue in issues))

    def test_missing_citation_is_reported_when_evidence_was_retrieved(self):
        report = "\n".join(f"## {heading}" for heading in REQUIRED_HEADINGS)
        report += "\nThis is not a confirmed diagnosis."
        issues = validate_report(report, [{"id": "PV-APPLE-SCAB-01"}])
        self.assertTrue(any("no passage citations" in issue for issue in issues))

    def test_pesticide_dosage_pattern_is_flagged(self):
        report = "\n".join(f"## {heading}" for heading in REQUIRED_HEADINGS)
        report += "\nThis is not a confirmed diagnosis. Mix spray with 5 ml per litre."
        issues = validate_report(report, [])
        self.assertTrue(any("dosage detected" in issue for issue in issues))

    def test_empty_report_is_rejected(self):
        self.assertEqual(validate_report(" ", []), ["The generated report is empty."])


if __name__ == "__main__":
    unittest.main()
