import numpy as np
from typing import Tuple, Optional
from ..models import Image


class ImageProcessor:
    def __init__(self, normalize: bool = True, random_state: Optional[int] = None):
        self.normalize = normalize
        self.original_image: Optional[Image] = None
        self._rng = np.random.RandomState(random_state) if random_state is not None else np.random
        
    def prepare_for_clustering(self, image: Image) -> np.ndarray:
        image_array = image.data
        
        self.original_image = image.copy()
        
        pixels = image_array.reshape(-1, image_array.shape[2])
        
        if self.normalize:
            pixels = pixels.astype(np.float64) / 255.0
        
        return pixels
    
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
    
    def apply_gaussian_noise(self, image: Image, mean: float = 0.0, std: float = 25.0) -> Image:
        noisy_data = image.data.astype(np.float64)
        
        noise = self._rng.normal(mean, std, noisy_data.shape)
        noisy_data = noisy_data + noise
        
        noisy_data = np.clip(noisy_data, 0, 255).astype(np.uint8)
        
        return Image(noisy_data)
    
    def apply_impulse_noise(self, image: Image, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> Image:
        noisy_data = image.data.copy()
        
        total_pixels = noisy_data.shape[0] * noisy_data.shape[1]
        
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [self._rng.randint(0, i, num_salt) for i in noisy_data.shape[:2]]
        noisy_data[salt_coords[0], salt_coords[1], :] = 255
        
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [self._rng.randint(0, i, num_pepper) for i in noisy_data.shape[:2]]
        noisy_data[pepper_coords[0], pepper_coords[1], :] = 0
        
        return Image(noisy_data)
