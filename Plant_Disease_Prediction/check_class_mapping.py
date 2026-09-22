import os

train_path = r"D:\FYP\machineLearning\Plant_Disease_Prediction\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
external_path = r"D:\FYP\External_Dataset\Test_OOD"

model_classes = sorted([
    folder for folder in os.listdir(train_path)
    if os.path.isdir(os.path.join(train_path, folder))
])

external_classes = sorted([
    folder for folder in os.listdir(external_path)
    if os.path.isdir(os.path.join(external_path, folder))
])

print(f"MODEL CLASSES: {len(model_classes)}")
for i, name in enumerate(model_classes):
    print(f"{i:02d}: {name}")

print(f"\nEXTERNAL CLASSES: {len(external_classes)}")
for name in external_classes:
    print(name)

print("\nEXTERNAL CLASSES NOT FOUND IN MODEL:")
missing = [name for name in external_classes if name not in model_classes]
if missing:
    for name in missing:
        print(name)
else:
    print("None")

print("\nMODEL CLASSES NOT FOUND IN EXTERNAL DATASET:")
missing_external = [name for name in model_classes if name not in external_classes]
if missing_external:
    for name in missing_external:
        print(name)
else:
    print("None")
