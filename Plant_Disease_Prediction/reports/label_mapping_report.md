# External label mapping report

**Status: PARTIALLY COMPLETE — actual external folder inventory was not available to this remote editor.**

## Mapping coverage
- Canonical model labels: 38 explicit identity mappings.
- Alternate external labels observed in the existing evaluator: 4 explicit mappings.
- Known alternate labels mapped: `Cherry___healthy`, `Corn___Cercospora_leaf_spot`, `Corn___Common_rust`, and `Corn___Northern_Leaf_Blight`.
- Unmapped external labels: **unknown until the 30 actual external folder names are enumerated**. Do not interpret the 38 identity entries as proof that all 30 external folders are mapped.
- Null mappings: none guessed. Add `null` only after confirming a label has no equivalent among the 38 model classes.
- Many-to-one: each of the four alternate labels maps to a canonical model label that also has an identity entry. This is intentional alias handling, not evidence that the external dataset contains duplicate classes.

## Manual review flags
- `Corn___Cercospora_leaf_spot` maps to `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`. The canonical label combines Cercospora leaf spot and Gray leaf spot wording; confirm the external annotation definition before treating it as equivalent.
- The three remaining aliases are spelling/name normalization based on existing project code. Confirm the dataset's label documentation before making scientific equivalence claims.
- Other disease-definition differences cannot be responsibly identified without the external label list and dataset documentation.

## Validation
`evaluate.py` validates every non-null mapping target against `class_names.json` before inference and fails loudly on invalid targets. It reports:
1. **Mapped-class accuracy** — only images with a valid target class.
2. **Overall accuracy** — all readable images, counting labels without a model equivalent as incorrect.

## Required finalization
Run from the project directory:
```powershell
Get-ChildItem "D:\FYP\External_Dataset\Test_OOD" -Directory | Select-Object -ExpandProperty Name
```
Compare those 30 names against `label_mapping.yaml`, then explicitly add any missing label as a verified class mapping or `null`. Do not silently infer an equivalent disease.
