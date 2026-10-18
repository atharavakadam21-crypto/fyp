"""Regression tests for the app/evaluation preprocessing contract."""
import numpy as np
from PIL import Image

from preprocess import preprocess_image
from evaluate_external_dataset import load_image as evaluation_preprocess


def test_app_and_evaluation_preprocessing_match_for_three_images(tmp_path):
    # Deterministic synthetic RGB samples avoid requiring the private dataset.
    for index, shape in enumerate(((73, 91), (128, 128), (257, 143))):
        height, width = shape
        y, x = np.indices((height, width))
        pixels = np.stack(
            (
                (x * 3 + index * 17) % 256,
                (y * 5 + index * 29) % 256,
                (x + y * 7 + index * 41) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        image_path = tmp_path / f"sample_{index}.png"
        Image.fromarray(pixels, mode="RGB").save(image_path)

        with Image.open(image_path) as image:
            app_tensor = preprocess_image(image)
        eval_tensor = evaluation_preprocess(image_path)

        assert app_tensor.shape == (128, 128, 3)
        assert app_tensor.dtype == np.float32
        assert eval_tensor.dtype == np.float32
        assert np.allclose(app_tensor, eval_tensor, rtol=0.0, atol=1e-6)
