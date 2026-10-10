"""Create a visual contact sheet of candidate-model misclassifications.

Reads image_predictions.csv created by analyze_mixed_domain_errors.py. No model
inference is performed. It is a visual audit aid, not an automatic relabeler.
"""
from __future__ import annotations

import argparse
import csv
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("error_analysis_field_mixed/image_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("error_analysis_field_mixed/misclassified_contact_sheet.jpg"))
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=320, help="Width of each thumbnail cell.")
    parser.add_argument("--pair-filter", type=str, default="", help="Optional substring required in actual->predicted pair, case-insensitive.")
    args = parser.parse_args()

    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    if args.max_images < 1 or args.columns < 1 or args.width < 160:
        parser.error("--max-images and --columns must be positive; --width must be >= 160")

    with args.predictions.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    wrong = [row for row in rows if row.get("correct", "").strip().lower() in {"false", "0", "no"}]
    if args.pair_filter:
        needle = args.pair_filter.casefold()
        wrong = [r for r in wrong if needle in f'{r["true_label"]} -> {r["predicted_label"]}'.casefold()]
    wrong.sort(key=lambda r: float(r.get("confidence") or 0.0), reverse=True)
    wrong = wrong[:args.max_images]
    if not wrong:
        raise ValueError("No misclassified images matched the selected filter.")

    padding = 10
    label_height = 94
    cell_h = int(args.width * 0.86) + label_height
    rows_count = (len(wrong) + args.columns - 1) // args.columns
    sheet = Image.new("RGB", (args.columns * args.width, rows_count * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for idx, row in enumerate(wrong):
        x = (idx % args.columns) * args.width
        y = (idx // args.columns) * cell_h
        image_path = Path(row["path"])
        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((args.width - 2 * padding, int(args.width * 0.86) - 2 * padding))
                tile = Image.new("RGB", (args.width, int(args.width * 0.86)), "white")
                tile.paste(image, ((args.width - image.width) // 2, (int(args.width * 0.86) - image.height) // 2))
                sheet.paste(tile, (x, y))
        except Exception as exc:
            draw.text((x + padding, y + padding), f"Could not open image: {exc}", fill="black", font=small_font)

        actual = textwrap.wrap("Actual: " + row["true_label"], width=36)
        predicted = textwrap.wrap("Predicted: " + row["predicted_label"], width=36)
        label_y = y + int(args.width * 0.86) + 2
        draw.text((x + padding, label_y), f"#{idx + 1}  confidence={float(row['confidence']):.3f}", fill="black", font=font)
        line_y = label_y + 20
        for line in actual[:2]:
            draw.text((x + padding, line_y), line, fill="black", font=small_font)
            line_y += 15
        for line in predicted[:2]:
            draw.text((x + padding, line_y), line, fill="darkred", font=small_font)
            line_y += 15
        draw.rectangle((x, y, x + args.width - 1, y + cell_h - 1), outline="gray", width=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=90)
    pairs = Counter((r["true_label"], r["predicted_label"]) for r in wrong)
    print(f"Misclassified images shown: {len(wrong)}")
    print(f"Contact sheet saved: {args.output.resolve()}")
    print("Pairs represented:")
    for (actual, predicted), count in pairs.most_common(12):
        print(f"{count:>3}  {actual} -> {predicted}")


if __name__ == "__main__":
    main()
