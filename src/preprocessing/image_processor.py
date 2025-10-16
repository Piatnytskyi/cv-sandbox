import numpy as np
from typing import Tuple, Optional
from ..models import Image


class ImageProcessor:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.original_image: Optional[Image] = None
        
    def prepare_for_clustering(self, image: Image) -> np.ndarray:
        image_array = image.data
        
        self.original_image = image.copy()
        
        pixels = image_array.reshape(-1, image_array.shape[2])
        
        if self.normalize:
            pixels = pixels.astype(np.float64) / 255.0
        
        return pixels
    
    @staticmethod
    def create_gradient(width: int = 400, height: int = 300) -> Image:
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                image_array[y, x] = [
                    int(255 * x / width),
                    int(255 * y / height),
                    int(255 * (1 - x / width))
                ]
        
        return Image(image_array)
    
    def reconstruct_image(
        self,
        labels: np.ndarray,
        cluster_centers: np.ndarray,
        shape: Optional[Tuple[int, ...]] = None
    ) -> Image:
        if shape is None:
            if self.original_image is None:
                raise ValueError("No shape provided and no original_image stored")
            shape = self.original_image.shape
        
        quantized_pixels = cluster_centers[labels]

        if self.normalize:
            quantized_pixels = (quantized_pixels * 255).astype(np.uint8)
        else:
            quantized_pixels = quantized_pixels.astype(np.uint8)
        
        quantized_array = quantized_pixels.reshape(shape)
        
        return Image(quantized_array)

