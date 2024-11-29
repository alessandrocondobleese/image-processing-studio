import numpy
import cv2 as opencv

from cv2.typing import MatLike
from typing import Dict
from scipy.stats import mode

from lib.image_processing.image_processor import ImageProcessor
from lib.image_processing.image import image_is_color, image_as_GRAY


class Average(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        size = parameters.get("size")
        size = (size, size)
        result_image = opencv.blur(image, size)
        return result_image


class Median(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        size = parameters.get("size")
        result_image = opencv.medianBlur(image, size)
        return result_image


class Mode(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        size = parameters.get("size")

        kernel = opencv.getStructuringElement(opencv.MORPH_RECT, (size, size))

        filtered_image = numpy.zeros_like(image, dtype=numpy.uint8)

        margin = size // 2

        for i in range(margin, image.shape[0] - margin):
            for j in range(margin, image.shape[1] - margin):
                # Extract the region defined by the kernel
                window = image[i - margin : i + margin + 1, j - margin : j + margin + 1]
                mode_value = mode(window[kernel == 1], axis=None).mode[0]
                filtered_image[i, j] = mode_value

        return filtered_image


class Minimum(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        size = parameters.get("size")
        size = (size, size)
        shape = opencv.MORPH_CROSS
        kernel = opencv.getStructuringElement(shape, size)
        result_image = opencv.erode(image, kernel)
        return result_image


class Maximum(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        size = parameters.get("size")
        size = (size, size)
        shape = opencv.MORPH_CROSS
        kernel = opencv.getStructuringElement(shape, size)
        result_image = opencv.dilate(image, kernel)
        return result_image


class Gaussian:
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        sigma = parameters.get("sigma")
        size = parameters.get("size")
        size = (size, size)
        result_image = opencv.GaussianBlur(image, size, sigma)
        return result_image


class Kirsch(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        kernels = [
            numpy.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=numpy.float32),
            numpy.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=numpy.float32),
            numpy.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=numpy.float32),
            numpy.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=numpy.float32),
            numpy.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=numpy.float32),
            numpy.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=numpy.float32),
            numpy.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=numpy.float32),
            numpy.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=numpy.float32),
        ]

        image = image_as_GRAY(image) if image_is_color(image) else image

        responses = [opencv.filter2D(image, -1, kernel) for kernel in kernels]

        kirsch_output = numpy.max(responses, axis=0)

        return kirsch_output
