"""Lightweight, dependency-minimal image quality diagnostics for PlantVision AI."""
import numpy as np
from PIL import Image


def assess_image_quality(image):
    """Measure brightness, contrast and sharpness; return advisory warnings.

    These heuristics are deliberately advisory only: foliage, backgrounds and
    camera conditions vary widely, so they must not automatically reject a photo.
    """
    if image is None:
        raise ValueError("image is required")
    resampling = getattr(Image, "Resampling", Image)
    gray = image.convert("L").resize((256, 256), resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32)
    brightness = float(np.mean(pixels))
    contrast = float(np.std(pixels))

    # Variance of a 4-neighbour discrete Laplacian: a simple edge/sharpness proxy.
    center = pixels[1:-1, 1:-1]
    laplacian = (
        pixels[:-2, 1:-1] + pixels[2:, 1:-1]
        + pixels[1:-1, :-2] + pixels[1:-1, 2:]
        - 4.0 * center
    )
    sharpness = float(np.var(laplacian))

    warnings = []
    if brightness < 45:
        warnings.append("The photo looks very dark; add more even natural light.")
    elif brightness > 215:
        warnings.append("The photo looks very bright; avoid glare and direct flash.")
    if contrast < 18:
        warnings.append("The image has low contrast; place one leaf against a simple contrasting background.")
    if sharpness < 20:
        warnings.append("The image may be blurry; hold the camera steady and refocus on the leaf.")

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "warnings": warnings,
        "status": "review" if warnings else "ok",
    }
