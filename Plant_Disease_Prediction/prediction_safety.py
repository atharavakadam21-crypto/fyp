"""Confidence and ambiguity checks for PlantVision classifier outputs.

A high top-1 score alone can be misleading when the runner-up is nearly tied.
This deterministic review flag does not claim calibrated probabilities or
replace expert diagnosis.
"""
import numpy as np

KNOWN_CLASS = "KNOWN CLASS"
LOW_CONFIDENCE = "UNKNOWN / LOW CONFIDENCE"
AMBIGUOUS_REVIEW = "AMBIGUOUS / REVIEW"


def assess_prediction(scores, threshold=0.65, min_margin=0.10):
    """Return status and top-1/top-2 scores for a class probability vector."""
    probabilities = np.asarray(scores, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("scores must be a one-dimensional array with at least two classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("scores must contain only finite values")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("scores must be probabilities in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 <= min_margin <= 1:
        raise ValueError("min_margin must be between 0 and 1")

    top_two = np.sort(probabilities)[-2:][::-1]
    top_score = float(top_two[0])
    runner_up_score = float(top_two[1])
    margin = top_score - runner_up_score

    if top_score < threshold:
        status = LOW_CONFIDENCE
    elif margin < min_margin:
        status = AMBIGUOUS_REVIEW
    else:
        status = KNOWN_CLASS

    return {
        "status": status,
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "margin": float(margin),
        "threshold": float(threshold),
        "min_margin": float(min_margin),
    }
