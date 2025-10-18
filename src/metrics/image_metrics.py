import numpy as np
from typing import Tuple
from ..models import Image

class ImageMetrics:    
    @staticmethod
    def _flatten_images(image1: Image, image2: Image) -> Tuple[np.ndarray, np.ndarray]:
        if image1.shape != image2.shape:
            raise ValueError(f"Images must have the same shape. Got {image1.shape} and {image2.shape}")
        
        flat1 = image1.data.reshape(-1, image1.data.shape[2])
        flat2 = image2.data.reshape(-1, image2.data.shape[2])
        
        return flat1, flat2
    
    @staticmethod
    def _compute_pixel_matches(image1: Image, image2: Image, tolerance: int = 0) -> Tuple[int, int, int, int]:
        flat1, flat2 = ImageMetrics._flatten_images(image1, image2)
        
        if tolerance == 0:
            matches = np.all(flat1 == flat2, axis=1)
        else:
            differences = np.abs(flat1.astype(np.int32) - flat2.astype(np.int32))
            matches = np.all(differences <= tolerance, axis=1)
        
        true_positives = np.sum(matches)
        false_negatives = np.sum(~matches)
        
        true_negatives = 0
        false_positives = 0
        
        return true_positives, false_positives, true_negatives, false_negatives
    
    @staticmethod
    def precision(image1: Image, image2: Image, tolerance: int = 0) -> float:
        tp, fp, tn, fn = ImageMetrics._compute_pixel_matches(image1, image2, tolerance)
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    @staticmethod
    def recall(image1: Image, image2: Image, tolerance: int = 0) -> float:
        tp, fp, tn, fn = ImageMetrics._compute_pixel_matches(image1, image2, tolerance)
        
        if tp + fn == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(image1: Image, image2: Image, tolerance: int = 0) -> float:
        prec = ImageMetrics.precision(image1, image2, tolerance)
        rec = ImageMetrics.recall(image1, image2, tolerance)
        
        if prec + rec == 0:
            return 0.0
        
        return 2 * (prec * rec) / (prec + rec)

    @staticmethod
    def mse(image1: Image, image2: Image) -> float:
        flat1, flat2 = ImageMetrics._flatten_images(image1, image2)
        
        squared_diff = (flat1.astype(np.float64) - flat2.astype(np.float64)) ** 2
        return float(np.mean(squared_diff))
    
    @staticmethod
    def psnr(image1: Image, image2: Image, max_pixel_value: float = 255.0) -> float:
        mse_value = ImageMetrics.mse(image1, image2)
        
        if mse_value == 0:
            return float('inf')
        
        return float(20 * np.log10(max_pixel_value / np.sqrt(mse_value)))
