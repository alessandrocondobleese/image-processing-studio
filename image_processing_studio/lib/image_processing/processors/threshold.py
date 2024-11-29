import cv2 as opencv

import numpy

from cv2.typing import MatLike
from typing import Dict

from lib.image_processing.image_processor import ImageProcessor
from lib.image_processing.image import image_is_color


class Binary(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        threshold = parameters.get("threshold")
        if image_is_color(image):
            result_image = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
            _, result_image = opencv.threshold(
                result_image, threshold, 255, opencv.THRESH_BINARY
            )
            return result_image

        _, result_image = opencv.threshold(image, threshold, 255, opencv.THRESH_BINARY)
        return result_image


class Otsu(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        if image_is_color(image):
            gray_image = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
            _, result_image = opencv.threshold(
                gray_image, 0, 255, opencv.THRESH_BINARY + opencv.THRESH_OTSU
            )
            return result_image
        
        else:
            gray_image = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
            _, result_image = opencv.threshold(
                gray_image, 0, 255, opencv.THRESH_BINARY + opencv.THRESH_OTSU
            )
            return result_image
        
class Invert(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        result_image = 255 - image
        return result_image


class Mean(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        threshold = numpy.mean(image)
        thresholded_image = (image >= threshold).astype(numpy.uint8) * 255
        return thresholded_image
    
class Canny(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        lower_threshold = parameters.get("lower_threshold")
        upper_threshold = parameters.get("upper_threshold")
        return opencv.Canny(lower_threshold, upper_threshold)