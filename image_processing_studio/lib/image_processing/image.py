import numpy
import pandas
import cv2 as opencv

from cv2.typing import MatLike


def image_from_buffer(buffer: bytes) -> MatLike:
    image_bytes = numpy.asarray(bytearray(buffer), dtype="uint8")
    image = opencv.imdecode(image_bytes, opencv.IMREAD_COLOR)

    return image


def image_is_grayscale(image: MatLike) -> MatLike:
    if len(image.shape) == 2 or image.shape[2] == 1:
        return True

    blue_channel, green_channel, red_channel = opencv.split(image)

    blue_and_green_are_equals = (blue_channel == green_channel).all()
    blue_and_red_are_equals = (green_channel == red_channel).all()

    return blue_and_green_are_equals and blue_and_red_are_equals


def image_is_color(image: MatLike) -> MatLike:
    return not image_is_grayscale(image)


def image_as_RGB(image: MatLike) -> MatLike:
    return opencv.cvtColor(image, opencv.COLOR_BGR2RGB)


def image_as_GRAY(image: MatLike) -> MatLike:
    return opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)


def image_channels_histograms(image: MatLike) -> pandas.DataFrame:
    if image_is_grayscale(image):
        image_histogram = opencv.calcHist([image], [0], None, [256], [0, 256])
        image_histogram = pandas.DataFrame(image_histogram, columns=["gray"])
        return image_histogram

    image_channels = opencv.split(image)
    image_channels_names = ("blue", "green", "red")
    image_histograms = {}
    for image_channel_index, image_channel_name in enumerate(image_channels_names):
        image_channel_histogram = opencv.calcHist(
            [image_channels[image_channel_index]], [0], None, [256], [0, 256]
        )
        image_histograms[image_channel_name] = image_channel_histogram.flatten()

    image_histograms = pandas.DataFrame(image_histograms)
    return image_histograms


def image_channel_histogram_stats(
    histogram: pandas.DataFrame | None,
) -> pandas.DataFrame:
    histogram_counts = histogram["frequency"].values

    intensities = histogram["index"].values

    total_pixels = histogram_counts.sum()

    p = histogram_counts / total_pixels

    mean = numpy.sum(intensities * p)

    variance = numpy.sum(((intensities - mean) ** 2) * p)
    std_dev = numpy.sqrt(variance)

    skewness = numpy.sum(((intensities - mean) ** 3) * p) / ((std_dev**3))

    return {"mean": mean, "standard_deviation": std_dev, "skewness": skewness}
