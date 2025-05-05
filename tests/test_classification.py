"""
Package tests defines unit tests for hot dog classification logic.
"""

import os
import unittest
from PIL import Image

from app import is_image_hot_dog


class HotDogClassificationTests(unittest.TestCase):
    """
    HotDogClassificationTests verifies the core hot dog classification logic
    against known sample images in JPEG and HEIF formats.
    """

    def setUp(self):
        current_directory = os.path.dirname(__file__)
        images_directory = os.path.join(current_directory, "images")
        self.hot_dog_image_paths = [
            os.path.join(images_directory, "hotdog_1.jpg"),
            os.path.join(images_directory, "hotdog_2.jpg"),
            os.path.join(images_directory, "hotdog_1.heic"),
            os.path.join(images_directory, "hotdog_2.heic"),
        ]
        self.not_hot_dog_image_paths = [
            os.path.join(images_directory, "not_hotdog_1.jpg"),
            os.path.join(images_directory, "not_hotdog_2.jpg"),
            os.path.join(images_directory, "not_hotdog_1.heic"),
            os.path.join(images_directory, "not_hotdog_2.heic"),
        ]

    def test_hot_dog_images_are_detected(self):
        for image_path in self.hot_dog_image_paths:
            pil_image = Image.open(image_path).convert("RGB")
            self.assertTrue(
                is_image_hot_dog(pil_image),
                f"{image_path} should be classified as hot dog",
            )

    def test_non_hot_dog_images_are_not_detected(self):
        for image_path in self.not_hot_dog_image_paths:
            pil_image = Image.open(image_path).convert("RGB")
            self.assertFalse(
                is_image_hot_dog(pil_image),
                f"{image_path} should be classified as not hot dog",
            )


if __name__ == "__main__":
    unittest.main()
