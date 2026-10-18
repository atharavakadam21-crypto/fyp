# Accuracy discrepancy investigation

**Status: OPEN — reproduction blocked in the remote GitHub editing environment.**

Previously recorded results are approximately 9.14% and 9.22% on the external set (2,374 images). The difference is small but material; it is not safe to assign a cause without per-image outputs from both scripts and the same checkpoint. The scripts have separate code paths for file inclusion and skipped-image accounting. Potential causes to inspect include denominator differences, corrupt/unreadable images, unmapped folders, model-path differences, and preprocessing changes across historical runs. None is confirmed as the cause.

The old scripts now have deprecation headers. `evaluate.py` is the canonical entry point and writes one row per readable image, explicit skipped-image details, mapped-only accuracy, and all-readable-image accuracy (unmapped labels count as incorrect in the latter).

To finish the investigation locally, run the old scripts against the same candidate checkpoint and external dataset, preserve each predictions CSV before the second run can overwrite any output, then join on normalized filepath and compare predicted index, confidence, mapped true label, and inclusion/skipped status. Compare denominator and class mapping before attributing the discrepancy to resize behavior.

No new accuracy value or discrepancy cause is claimed by this report.
