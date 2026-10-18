# Blockers and workarounds

## T3 — Run legacy evaluations and compare per-image outputs
- **Blocker:** The GitHub editing session can edit repository files but does not have the local external dataset at `D:\\FYP\\External_Dataset\\Test_OOD` or the local Keras checkpoints available for execution. The historical 9.14% vs 9.22% discrepancy cannot be reproduced or causally diagnosed here.
- **Workaround:** Pull this branch, run both retained legacy scripts with the same model/dataset, save each predictions CSV to separate names, then compare filepath, true/mapped label, prediction, confidence, skipped files, and denominator. Use `evaluate.py` for future reporting.
- **Do not infer:** Resize or denominator differences are plausible checks, not a verified cause.

## Validation execution
- **Blocker:** Unit tests, dataset evaluation, calibration, and candidate-vs-production comparisons require the local dataset and model binaries; these are not stored in the source checkout available to the remote editor.
- **Workaround:** Run the committed scripts locally and retain their generated reports. Reports explicitly mark measurements as pending rather than fabricating results.
