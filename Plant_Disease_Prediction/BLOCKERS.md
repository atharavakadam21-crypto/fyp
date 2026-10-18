# Blockers and workarounds

## T3 — Run legacy evaluations and compare per-image outputs
- **Blocker:** The GitHub editing session can edit repository files but does not have the local external dataset at `D:\\FYP\\External_Dataset\\Test_OOD` or the local Keras checkpoints available for execution. The historical 9.14% vs 9.22% discrepancy cannot be reproduced or causally diagnosed here.
- **Workaround:** Pull this branch, run both retained legacy scripts with the same model/dataset, save each predictions CSV to separate names, then compare filepath, true/mapped label, prediction, confidence, skipped files, and denominator. Use `evaluate.py` for future reporting.
- **Do not infer:** Resize or denominator differences are plausible checks, not a verified cause.

## Validation execution
- **Blocker:** Unit tests, dataset evaluation, calibration, and candidate-vs-production comparisons require the local dataset and model binaries; these are not stored in the source checkout available to the remote editor.
- **Workaround:** Run the committed scripts locally and retain their generated reports. Reports explicitly mark measurements as pending rather than fabricating results.

## T4 — Complete the exact 30-label external mapping
- **Blocker:** The local `Test_OOD` directory is not visible in the remote GitHub editing environment, so its exact 30 folder names cannot be enumerated. Guessing the missing labels would risk silently conflating disease definitions.
- **Workaround:** The YAML contains canonical identity mappings and four aliases already evidenced in the legacy evaluator; the report flags incomplete inventory. On the machine holding the dataset, list its subfolders and reconcile every one to a valid model class or explicit `null` before presenting mapped-only accuracy.

## T5 — Generate calibration plots and ECE metrics
- **Blocker:** The validation images and Keras checkpoint are local-only and cannot be executed in this remote GitHub editor. Numeric ECE and PNG artifacts therefore cannot be truthfully generated here.
- **Workaround:** Run `python calibration.py` with the original validation directory. The script fits temperature on one deterministic half and reports on the other; it saves the required PNG/CSV/JSON artifacts without touching the model file.

## T6 — Inspect serialized production/candidate checkpoints
- **Blocker:** The `.keras` checkpoint binaries are not available to the remote editor; the notebook's saved model summary cannot prove the current files have identical layers or weights.
- **Workaround:** Run `scripts/inspect_model.py` for production and candidate locally. The script emits layer configs, input/output shapes, parameter count, and weight names/shapes without writing to either checkpoint.

## T7 — Execute promotion gate
- **Blocker:** The gate needs the local validation dataset, external dataset, both checkpoint files, and a reconciled external label map. Those files are not available here, and external label inventory is still incomplete.
- **Workaround:** Run `python scripts/promotion_gate.py` locally after completing T4. It fails closed on missing data or failed checks and does not promote or overwrite any model.
