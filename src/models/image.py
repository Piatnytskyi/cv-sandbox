import numpy as np
from PIL import Image as PILImage
from typing import Optional


class Image:
    def __init__(self, data: Optional[np.ndarray] = None):
        if data is not None and data.ndim != 3:
            raise ValueError(f"Expected image array with 3 dimensions, got shape {data.shape}")
        self._data = data
        
    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            raise ValueError("No image data available")
        return self._data
    
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @classmethod
    def load(cls, image_path: str) -> 'Image':
        try:
            pil_image = PILImage.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image_array = np.array(pil_image)
            return cls(image_array)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{image_path}' not found")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def save(self, output_path: str) -> None:
        pil_image = PILImage.fromarray(self.data.astype(np.uint8))
        pil_image.save(output_path)
    
    def get_info(self) -> dict:
        data = self.data
        return {
            "shape": data.shape,
            "height": data.shape[0],
            "width": data.shape[1],
            "channels": data.shape[2] if data.ndim > 2 else 1,
            "total_pixels": data.shape[0] * data.shape[1],
            "dtype": str(data.dtype),
            "min_value": int(data.min()),
            "max_value": int(data.max()),
        }
    
    def copy(self) -> 'Image':
        return Image(self.data.copy())
