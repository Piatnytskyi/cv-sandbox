import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import numpy as np
from ..models import Image

class ResultVisualizer:   
    def __init__(self, style: str = 'default', dpi: int = 150):
        self.style = style
        self.dpi = dpi
        plt.style.use(self.style)
        
    def visualize_comparison(
        self,
        original: Image,
        processed: Image,
        title_original: str = 'Original Image',
        title_processed: str = 'Processed Image',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].imshow(original.data)
        axes[0].set_title(title_original, fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(processed.data)
        axes[1].set_title(title_processed, fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def visualize_predictions(
        self,
        images: np.ndarray,
        pred_labels: List[str],
        true_labels: List[str],
        title: str = 'Sample Predictions',
        figsize: Tuple[int, int] = (15, 6),
        cmap: str = 'gray'
    ) -> plt.Figure:
        n_samples = len(images)
        cols = n_samples // 2
        
        fig, axes = plt.subplots(2, cols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i in range(n_samples):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            if len(images[i].shape) == 3 and images[i].shape[2] == 1:
                image = images[i].reshape(images[i].shape[0], images[i].shape[1])
            else:
                image = images[i]
            
            # If cmap is specified and image is 3D, convert to grayscale
            if cmap is not None and len(image.shape) == 3:
                image = image.mean(axis=2)
            
            ax.imshow(image, cmap=cmap)
            
            color = 'green' if pred_labels[i] == true_labels[i] else 'red'
            ax.set_title(f"Pred: {pred_labels[i]}\nTrue: {true_labels[i]}", 
                        fontsize=10, color=color)
            ax.axis('off')
        
        plt.tight_layout()
        
        return fig
       
    def show(self) -> None:
        plt.show()
    
    def close_all(self) -> None:
        plt.close('all')