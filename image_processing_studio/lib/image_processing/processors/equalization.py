import numpy
import cv2 as opencv

from cv2.typing import MatLike
from typing import Dict

from lib.image_processing.image_processor import ImageProcessor
from lib.image_processing.image import image_is_color


class Rayleigh(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        alpha = parameters.get("alpha", 200)

        if image_is_color(image):
            hsv_image = opencv.cvtColor(image, opencv.COLOR_BGR2HSV)
            h, s, v = opencv.split(hsv_image)
            histogram = opencv.calcHist([v], [0], None, [256], (0, 256))
            p_g = histogram / v.size

        else:
            histogram = opencv.calcHist([image], [0], None, [256], (0, 256))
            p_g = histogram / image.size

        P_g = p_g.cumsum()

        g_min = 0
        epsilon = 1e-6

        F_g = g_min + numpy.sqrt(
            numpy.absolute(2 * (alpha**2) * numpy.log(epsilon + 1 / (1 + P_g)))
        )

        F_g = numpy.ceil(F_g)
        F_g = numpy.clip(F_g, 0, 255)
        F_g = F_g.astype("uint8")

        if image_is_color(image):
            equalized_v = F_g[v]

            result_image = opencv.merge([h, s, equalized_v])
            result_image = opencv.cvtColor(result_image, opencv.COLOR_HSV2BGR)

            return result_image
        else:
            result_image = F_g[image]

        return result_image


class Uniform(ImageProcessor):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        if image_is_color(image):
            b, g, r = opencv.split(image)

            b_eq = opencv.equalizeHist(b)
            g_eq = opencv.equalizeHist(g)
            r_eq = opencv.equalizeHist(r)

            return opencv.merge((b_eq, g_eq, r_eq))

        return opencv.equalizeHist(opencv.cvtColor(image, opencv.COLOR_BGR2GRAY))
