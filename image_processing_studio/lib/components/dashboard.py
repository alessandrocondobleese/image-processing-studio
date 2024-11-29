import marimo as mo
import cv2 as opencv
import lib.image_processing.image as imagelib

from typing import Optional, List


def dashboard(image, image_channel_histogram, image_histogram_stats=Optional[List[tuple]]):
    stats_section = mo.md("**Selecciona un canal**").center()
    if image_histogram_stats is not None:
        stats_section = mo.hstack(
            [
                mo.stat(
                    image_histogram_stat[0],
                    label=image_histogram_stat[1],
                )
                for image_histogram_stat in image_histogram_stats
            ],
            justify="space-around",
            widths="equal",
            gap=0.375,
        )

    dashboard = mo.hstack(
        [
            mo.image(imagelib.image_as_RGB(image), rounded=True),
            mo.vstack(
                [
                    image_channel_histogram,
                    stats_section,
                ]
            ),
        ],
        widths="equal",
        gap=0.75,
    )

    return dashboard

