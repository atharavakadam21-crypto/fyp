"""Offline tests for PlantVision prediction confidence safeguards."""
import unittest

from prediction_safety import (
    AMBIGUOUS_REVIEW,
    KNOWN_CLASS,
    LOW_CONFIDENCE,
    assess_prediction,
)


class PredictionSafetyTests(unittest.TestCase):
    def test_confident_and_separated_prediction_is_known(self):
        result = assess_prediction([0.88, 0.07, 0.05], threshold=0.65, min_margin=0.10)
        self.assertEqual(result["status"], KNOWN_CLASS)
        self.assertAlmostEqual(result["margin"], 0.81)

    def test_low_top_score_is_flagged(self):
        result = assess_prediction([0.42, 0.35, 0.23], threshold=0.65)
        self.assertEqual(result["status"], LOW_CONFIDENCE)

    def test_close_top_two_scores_require_review(self):
        result = assess_prediction([0.71, 0.68, 0.02], threshold=0.65, min_margin=0.10)
        self.assertEqual(result["status"], AMBIGUOUS_REVIEW)
        self.assertAlmostEqual(result["margin"], 0.03)

    def test_rejects_invalid_scores(self):
        for scores in ([0.5], [0.5, float("nan")], [-0.1, 1.1], [[0.5, 0.5]]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    assess_prediction(scores)

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            assess_prediction([0.8, 0.2], threshold=1.2)


if __name__ == "__main__":
    unittest.main()
