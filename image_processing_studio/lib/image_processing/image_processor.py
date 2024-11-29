from typing import Protocol, Dict
from cv2.typing import MatLike

class ImageProcessor(Protocol):
    def execute(self, image: MatLike, parameters: Dict) -> MatLike:
        ...