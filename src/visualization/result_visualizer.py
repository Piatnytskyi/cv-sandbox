import matplotlib.pyplot as plt
from typing import Tuple
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
       
    def show(self) -> None:
        plt.show()
    
    def close_all(self) -> None:
        plt.close('all')