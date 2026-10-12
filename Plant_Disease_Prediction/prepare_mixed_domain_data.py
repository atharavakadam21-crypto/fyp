"""Prepare a leakage-conscious PlantVillage + PlantDoc dataset.

This script does not train a model and never modifies the source datasets or
production model. It extracts only confidently mapped PlantDoc images from the
ZIP's train split. PlantDoc validation/test directories are not used.

Example:
  python prepare_mixed_domain_data.py
  python prepare_mixed_domain_data.py --plantdoc-zip "C:/Users/me/Downloads/PlantDoc-Dataset-master.zip"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
FIELD_VALIDATION_FRACTION = 0.15

# Explicit whitelist only. Generic labels such as "Cherry leaf", "Tomato leaf",
# and "Bell_pepper leaf" are excluded because they do not prove a healthy label.
LABEL_MAP = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn rust leaf": "Corn_(maize)___Common_rust_",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf black rot": "Grape___Black_rot",
}


def normalize_label(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip().lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash_bytes(data: bytes) -> str | None:
    """Small dependency-free perceptual hash for exact/near-identical images."""
    try:
        from io import BytesIO
        with Image.open(BytesIO(data)) as im:
            im = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(im.getdata())
        bits = []
        for y in range(8):
            row = y * 9
            for x in range(8):
                bits.append("1" if pixels[row + x] > pixels[row + x + 1] else "0")
        return f"{int(''.join(bits), 2):016x}"
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent,
        help="Project directory containing the original training dataset.",
    )
    parser.add_argument(
        "--plantdoc-zip", type=Path,
        default=Path.home() / "Downloads" / "PlantDoc-Dataset-master.zip",
        help="Path to the downloaded PlantDoc ZIP.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: data/mixed_domain_v1).",
    )
    parser.add_argument(
        "--skip-perceptual-dedupe", action="store_true",
        help="Skip dHash checks against original images (faster, less thorough).",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    zip_path = args.plantdoc_zip.resolve()
    output = (args.output or root / "data" / "mixed_domain_v1").resolve()
    original_train = (
        root / "New Plant Diseases Dataset(Augmented)"
        / "New Plant Diseases Dataset(Augmented)" / "train"
    )

    if not original_train.is_dir():
        raise FileNotFoundError(
            f"Original training folder not found: {original_train}\n"
            "Run this script from the project root or pass --project-root."
        )
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"PlantDoc ZIP not found: {zip_path}\n"
            "Pass its location with --plantdoc-zip."
        )
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Rename/remove this generated output folder before rerunning; "
            "the script will not overwrite data silently."
        )

    classes = sorted(p.name for p in original_train.iterdir() if p.is_dir())
    if len(classes) != 38:
        raise ValueError(f"Expected 38 original classes, found {len(classes)}.")
    class_set = set(classes)
    invalid = sorted(set(LABEL_MAP.values()) - class_set)
    if invalid:
        raise ValueError("Mapped labels not in original 38 classes: " + repr(invalid))

    output.mkdir(parents=True)
    plantdoc_train_out = output / "plantdoc_train"
    field_val_out = output / "field_validation"
    plantdoc_train_out.mkdir()
    field_val_out.mkdir()

    rng = random.Random(SEED)
    original_rows: list[dict] = []
    original_sha = set()
    original_dhash: list[int] = []
    original_dhash_set: set[int] = set()

    print("Indexing original images for duplicate checks...")
    original_files = [
        p for p in original_train.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for i, image_path in enumerate(original_files, 1):
        try:
            data = image_path.read_bytes()
            digest = sha256_bytes(data)
            original_sha.add(digest)
            if not args.skip_perceptual_dedupe:
                perceptual = dhash_bytes(data)
                if perceptual:
                    value = int(perceptual, 16)
                    original_dhash.append(value)
                    original_dhash_set.add(value)
            # Infer class from its immediate class directory under train.
            label = image_path.relative_to(original_train).parts[0]
            if label not in class_set:
                continue
            original_rows.append({
                "path": str(image_path.resolve()),
                "label": label,
                "source": "original",
                "split": "train",
            })
        except OSError:
            continue
        if i % 10000 == 0:
            print(f"  indexed {i:,}/{len(original_files):,} images")

    # dHash values are sorted for a cheap local neighborhood search.
    original_dhash.sort()
    # Index hashes by their top 8 bits so near-duplicate checks do not scan
    # all ~70k original images for every PlantDoc image.
    dhash_buckets: dict[int, list[int]] = defaultdict(list)
    for value in original_dhash:
        dhash_buckets[value >> 56].append(value)
    print(f"Original training images indexed: {len(original_rows):,}")

    candidates: dict[str, list[dict]] = defaultdict(list)
    audit_rows: list[dict] = []
    exact_duplicates = 0
    perceptual_duplicates = 0
    corrupt_images = 0
    unmapped = 0
    seen_plantdoc_sha: set[str] = set()

    print("Reading only PlantDoc train images from ZIP...")
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = info.filename.replace("\\", "/")
            parts = member.split("/")
            if Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Require the path segment train/<class>/<image>; never select test
            # or validation data.
            train_positions = [i for i, part in enumerate(parts[:-1])
                               if part.lower() == "train"]
            if not train_positions:
                continue
            idx = train_positions[-1]
            if idx + 2 != len(parts) - 1:
                # Expect exactly train/class/image; ignore nested/unusual layouts.
                continue

            source_label = parts[idx + 1]
            canonical = LABEL_MAP.get(normalize_label(source_label))
            if canonical is None:
                unmapped += 1
                audit_rows.append({
                    "source_label": source_label,
                    "canonical_label": "EXCLUDE",
                    "reason": "Unmapped/ambiguous source label",
                    "zip_member": member,
                })
                continue

            try:
                data = archive.read(info)
                # Verify image bytes before writing them into the dataset.
                from io import BytesIO
                with Image.open(BytesIO(data)) as im:
                    im.verify()
            except Exception:
                corrupt_images += 1
                audit_rows.append({
                    "source_label": source_label,
                    "canonical_label": canonical,
                    "reason": "Unreadable/corrupt image",
                    "zip_member": member,
                })
                continue

            digest = sha256_bytes(data)
            if digest in original_sha or digest in seen_plantdoc_sha:
                exact_duplicates += 1
                audit_rows.append({
                    "source_label": source_label,
                    "canonical_label": canonical,
                    "reason": "Exact duplicate",
                    "zip_member": member,
                })
                continue
            seen_plantdoc_sha.add(digest)

            if not args.skip_perceptual_dedupe:
                perceptual = dhash_bytes(data)
                if perceptual:
                    value = int(perceptual, 16)
                    # Search only buckets whose top 8 bits are within Hamming
                    # distance 2, then verify full 64-bit distance <= 4.
                    # This keeps the perceptual check practical for 70k images.
                    prefix = value >> 56
                    nearby = (
                        candidate
                        for bucket, values in dhash_buckets.items()
                        if (prefix ^ bucket).bit_count() <= 2
                        for candidate in values
                    )
                    if any((value ^ candidate).bit_count() <= 4 for candidate in nearby):
                        perceptual_duplicates += 1
                        audit_rows.append({
                            "source_label": source_label,
                            "canonical_label": canonical,
                            "reason": "Perceptual near-duplicate of original",
                            "zip_member": member,
                        })
                        continue

            candidates[canonical].append({
                "bytes": data,
                "digest": digest,
                "member": member,
                "source_label": source_label,
                "suffix": Path(parts[-1]).suffix.lower() or ".jpg",
            })

    train_rows = list(original_rows)
    field_val_rows: list[dict] = []
    plantdoc_train_counts = Counter()
    field_val_counts = Counter()

    for canonical, items in sorted(candidates.items()):
        rng.shuffle(items)
        if len(items) < 2:
            for item in items:
                audit_rows.append({
                    "source_label": item["source_label"],
                    "canonical_label": canonical,
                    "reason": "Too few images for a safe train/validation split",
                    "zip_member": item["member"],
                })
            continue

        n_val = max(1, round(len(items) * FIELD_VALIDATION_FRACTION))
        n_val = min(n_val, len(items) - 1)
        split_items = [
            ("field_validation", items[:n_val], field_val_out, field_val_rows),
            ("train", items[n_val:], plantdoc_train_out, train_rows),
        ]

        for split_name, selected, destination, rows in split_items:
            for item in selected:
                filename = f"{item['digest'][:16]}{item['suffix']}"
                destination_path = destination / canonical / filename
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                destination_path.write_bytes(item["bytes"])
                rows.append({
                    "path": str(destination_path.resolve()),
                    "label": canonical,
                    "source": "PlantDoc",
                    "split": split_name,
                })
                if split_name == "train":
                    plantdoc_train_counts[canonical] += 1
                else:
                    field_val_counts[canonical] += 1

    columns = ["path", "label", "source", "split"]
    write_csv(output / "train_manifest.csv", train_rows, columns)
    write_csv(output / "field_validation_manifest.csv", field_val_rows, columns)
    write_csv(output / "label_mapping_audit.csv", audit_rows,
              ["source_label", "canonical_label", "reason", "zip_member"])
    (output / "class_names.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    counts = []
    original_counts = Counter(row["label"] for row in original_rows)
    for label in classes:
        counts.append({
            "class": label,
            "original_train_images": original_counts[label],
            "plantdoc_train_images": plantdoc_train_counts[label],
            "field_validation_images": field_val_counts[label],
        })
    write_csv(output / "class_counts.csv", counts,
              ["class", "original_train_images", "plantdoc_train_images",
               "field_validation_images"])

    print("\n========== PREPARATION COMPLETE ==========")
    print(f"Output: {output}")
    print(f"Original train images: {len(original_rows):,}")
    print(f"PlantDoc train images added: {sum(plantdoc_train_counts.values()):,}")
    print(f"PlantDoc field-validation images: {sum(field_val_counts.values()):,}")
    print(f"Classes with mapped PlantDoc images: {len(candidates)}")
    print(f"Unmapped/ambiguous images excluded: {unmapped:,}")
    print(f"Exact duplicates excluded: {exact_duplicates:,}")
    print(f"Perceptual near-duplicates excluded: {perceptual_duplicates:,}")
    print(f"Corrupt images excluded: {corrupt_images:,}")
    print("Production model and source datasets were not modified.")
    print("Review class_counts.csv and label_mapping_audit.csv before training.")


if __name__ == "__main__":
    main()
