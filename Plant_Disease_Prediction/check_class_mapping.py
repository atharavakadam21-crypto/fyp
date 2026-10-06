import os
from pathlib import Path

TRAIN_ROOT = Path(
    os.getenv(
        "TRAIN_DIR",
        "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    )
)
EXTERNAL_ROOT = Path(
    os.getenv(
        "EXTERNAL_DATASET",
        r"D:\FYP\External_Dataset\Test_OOD",
    )
)

EXTERNAL_TO_MODEL = {
    "Cherry___healthy": "Cherry_(including_sour)___healthy",
    "Corn___Cercospora_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
}

model_classes = sorted(
    p.name for p in TRAIN_ROOT.iterdir() if p.is_dir()
)
external_classes = sorted(
    p.name for p in EXTERNAL_ROOT.iterdir() if p.is_dir()
)

print(f"MODEL CLASSES: {len(model_classes)}")
for i, name in enumerate(model_classes):
    print(f"{i:02d}: {name}")

print(f"\nEXTERNAL CLASSES: {len(external_classes)}")
mapped_external = set()

for name in external_classes:
    mapped = EXTERNAL_TO_MODEL.get(name, name)
    mapped_external.add(mapped)
    status = "PASS" if mapped in model_classes else "MISSING"
    print(f"{status}: {name} -> {mapped}")

missing = [
    name for name in external_classes
    if EXTERNAL_TO_MODEL.get(name, name) not in model_classes
]

print("\nEXTERNAL CLASSES WITHOUT A MODEL MAPPING:")
if missing:
    for name in missing:
        print(f"  {name}")
else:
    print("None")

absent_from_external = [
    name for name in model_classes
    if name not in mapped_external
]

print("\nMODEL CLASSES ABSENT FROM EXTERNAL DATASET:")
for name in absent_from_external:
    print(f"  {name}")

if missing:
    raise SystemExit(
        "\nMapping check failed. Do not fine-tune until every external class is mapped."
    )

print("\nPASS: every external class maps to a known 38-class model output.")
