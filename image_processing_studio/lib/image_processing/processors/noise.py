import numpy

from cv2.typing import MatLike
from typing import Dict

from lib.image_processing.image_processor import ImageProcessor


class Salt(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        amount = parameters.get("amount", 0.01)
        noisy_image = image.copy()

        num_salt = int(amount * image.size)

        salt_coords = [
            numpy.random.randint(0, i - 1, num_salt) for i in image.shape[:2]
        ]

        noisy_image[salt_coords[0], salt_coords[1]] = 255

        return noisy_image


class Pepper(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        amount = parameters.get("amount", 0.01)
        noisy_image = image.copy()

        num_pepper = int(amount * image.size)

        pepper_coords = [
            numpy.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]
        ]

        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image
