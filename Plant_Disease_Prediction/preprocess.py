"""Canonical PlantVision preprocessing shared by inference and evaluation.

The original training notebook uses TensorFlow image_dataset_from_directory with
RGB, image_size=(128, 128), bilinear interpolation, and no pixel rescaling.
Therefore returned float32 pixels remain in [0, 255]; no mean/std normalization
is applied.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMAGE_SIZE = (128, 128)
COLOR_MODE = "RGB"
INTERPOLATION = "bilinear"


def preprocess_image(image, image_size=IMAGE_SIZE) -> np.ndarray:
    """Return one RGB float32 HWC tensor matching the original training input.

    Args:
        image: PIL image or HWC NumPy image.
        image_size: (height, width), defaulting to the trained 128x128 input.

    Returns:
        float32 array with shape (height, width, 3), raw pixel range [0, 255].
    """
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image).convert(COLOR_MODE)
        array = np.asarray(image)
    else:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Expected an HWC image, got shape {array.shape}")
        if array.shape[-1] == 4:
            array = array[..., :3]
        elif array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {array.shape[-1]}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode=COLOR_MODE)

    tensor = tf.convert_to_tensor(np.asarray(image), dtype=tf.float32)
    tensor = tf.image.resize(
        tensor,
        size=tuple(image_size),
        method=INTERPOLATION,
        antialias=False,
    )
    return tensor.numpy().astype(np.float32, copy=False)
